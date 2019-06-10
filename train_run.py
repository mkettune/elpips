import tensorflow as tf
import numpy as np

import time
import os
import argparse
import sys

import elpips
import train_dataset


parser = argparse.ArgumentParser()

parser.add_argument('--metric', type=str, required=True, help='elpips_vgg, elpips_squeeze_maxpool, lpips_vgg or lpips_squeeze')
parser.add_argument('--name', type=str, required=True, help='name for training instance')
parser.add_argument('--use_dropout', action='store_true', help='use dropout for lpips weights when training')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training')
parser.add_argument('--display_freq', type=int, default=5000, help='show training results in console every [display_freq] training images. Must be a multiple of batch_size.')
parser.add_argument('--save_latest_freq', type=int, default=10000, help='save current weights every [save_latest_freq] training images. Must be a multiple of batch_size.')

parser.add_argument('--nepoch', type=int, default=5, help='number of epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=5, help='number of additional epochs at linearly decaying learning rate')
parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--datasets', type=str, nargs='+', default=['train/traditional','train/cnn','train/mix'], help='datasets to train on: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')


opt = parser.parse_args()


# Load model.

# No input transformations for training the metrics.
# (Input transformations could be used of course.)
model_config = elpips.get_config(opt.metric, batch_size=opt.batch_size)
model_config.enable_dropout = False
model_config.enable_offset = False
model_config.enable_flip = False
model_config.enable_swap = False
model_config.enable_color_permutation = False
model_config.enable_color_multiplication = False
model_config.enable_scale = False


# Implement the small network which tries to predict human 2afc results based on the estimated distances d(a, ref) and d(b, ref).

def conv2d_1x1(name, input, input_feature_count, output_feature_count, W=None, b=None):
	W = tf.get_variable(
		name=name+"_W",
		shape=[1, 1, input_feature_count, output_feature_count] if W is None else None,
		dtype=tf.float32,
		initializer=tf.contrib.layers.variance_scaling_initializer() if W is None else W,
		trainable=True
	)
	b = tf.get_variable(
		name=name+"_b",
		shape=[1, 1, 1, output_feature_count] if b is None else None,
		dtype=tf.float32,
		initializer=tf.zeros_initializer() if b is None else b,
		trainable=True
	)
	layer = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b
	return layer, W, b

def BCERankingLoss(d0s, d1s, judges, chn_mid=32, eps=0.1):
	params = {'W1':None, 'b1':None, 'W2':None, 'b2':None, 'W3':None, 'b3':None}
	
	with tf.variable_scope('dist2logit', reuse=tf.AUTO_REUSE):
		layer = tf.stack([d0s, d1s, d0s-d1s, d0s/(eps+d1s), d1s/(eps+d0s)], axis=1)
		layer = tf.reshape(layer, [-1, 1, 1, 5])
	
		layer, W1, b1 = conv2d_1x1('conv1', layer, 5, chn_mid, params['W1'], params['b1'])
		layer = tf.nn.leaky_relu(layer, 0.2)
		
		layer, W2, b2 = conv2d_1x1('conv2', layer, chn_mid, chn_mid, params['W2'], params['b2'])
		layer = tf.nn.leaky_relu(layer, 0.2)

		layer, W3, b3 = conv2d_1x1('conv3', layer, chn_mid, 1, params['W3'], params['b3'])
		
		losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=judges, logits=layer[:,0,0,0])
		loss = tf.reduce_mean(losses)
		
	return loss, {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2, 'W3':W3, 'b3':b3}


# Construct the graph.

with tf.device('/gpu:0'):
	model = elpips.Metric(model_config, back_prop=True, trainable='lpips', use_lpips_dropout=opt.use_dropout)

	tf_X_image = tf.placeholder(tf.float32, [opt.batch_size, 64, 64, 3])
	tf_Y_image = tf.placeholder(tf.float32, [opt.batch_size, 64, 64, 3])
	tf_ref_image = tf.placeholder(tf.float32, [opt.batch_size, 64, 64, 3])
	tf_judge = tf.placeholder(tf.float32, [opt.batch_size])
	

	print("Compiling distance evaluation")

	tf_d1, tf_d2 = model.forward((tf_X_image, tf_Y_image), tf_ref_image)

	# Forcing prediction of human 2afc values to be symmetric.
	print("Compiling BCE Ranking Loss")
	tf_loss1, bce_params = BCERankingLoss(tf_d1, tf_d2, tf_judge)
	tf_loss2, bce_params = BCERankingLoss(tf_d2, tf_d1, 1.0 - tf_judge)
	tf_loss = 0.5 * (tf_loss1 + tf_loss2)

	# Learning rate schedule.
	original_learning_rate = opt.lr
	tf_learning_rate = tf.get_variable('learning_rate', dtype=tf.float32, initializer=original_learning_rate, trainable=False)

	tf_step = tf.get_variable('step', dtype=tf.int32, initializer=0, trainable=False)
	tf_increase_step = tf.assign(tf_step, tf_step + 1)
	tf_actual_learning_rate = tf_learning_rate * tf.clip_by_value((tf.cast(tf_step, tf.float32) - 50.0) / 50.0, 0.0, 1.0)
	
	# Optimizer.
	print("Compiling Adam optimizer")
	tf_optimizer = tf.train.AdamOptimizer(tf_actual_learning_rate, 0.9, 0.999,) # Original code used beta1 = 0.5. Probably not significant?
	tf_minimize = tf_optimizer.minimize(tf_loss, var_list=tf.trainable_variables())

	# Linearly decaying learning rate in the end.
	tf_decay_learning_rate = tf.assign(tf_learning_rate, tf_learning_rate - original_learning_rate / tf.cast(opt.nepoch_decay, tf.float32))

	# Run-time evaluation of training accuracy.
	tf_d2_lt_d1 = tf.cast(tf.less(tf_d2, tf_d1), tf.float32)
	tf_compute_accuracy = tf.reduce_mean(tf_d2_lt_d1 * tf_judge + (1.0 - tf_d2_lt_d1) * (1.0 - tf_judge))
	
	tf_smooth_accuracy_c = tf.get_variable('smooth_accuracy_c', dtype=tf.float32, initializer=0.0, trainable=False)
	tf_smooth_accuracy_w = tf.get_variable('smooth_accuracy_w', dtype=tf.float32, initializer=0.0, trainable=False)
	with tf.control_dependencies([
		tf.assign(tf_smooth_accuracy_c, 0.999 * tf_smooth_accuracy_c + tf_compute_accuracy),
		tf.assign(tf_smooth_accuracy_w, 0.999 * tf_smooth_accuracy_w + 1.0)
	]):
		tf_smooth_accuracy = tf_smooth_accuracy_c / tf_smooth_accuracy_w
		
	# Run-time evaluation of training loss.
	tf_smooth_loss_c = tf.get_variable('smooth_loss_c', dtype=tf.float32, initializer=0.0, trainable=False)
	tf_smooth_loss_w = tf.get_variable('smooth_loss_w', dtype=tf.float32, initializer=0.0, trainable=False)
	with tf.control_dependencies([
		tf.assign(tf_smooth_loss_c, 0.995 * tf_smooth_loss_c + tf_loss),
		tf.assign(tf_smooth_loss_w, 0.995 * tf_smooth_loss_w + 1.0)
	]):
		tf_smooth_loss = tf_smooth_loss_c / tf_smooth_loss_w

	# Project LPIPS weights to be non-negative.
	tf_clamp_weights = [tf.assign(W, tf.maximum(0.0, W)) for W in model.network.linear_weight_as_dict.values()]

	
def updateLearningRate(sess, nepoch_decay):
	old_lr = sess.run(tf_learning_rate)
	new_lr = sess.run(tf_decay_learning_rate)
	print('update lr decay: %f -> %f' % (old_lr, new_lr))



# Create data loader.
print("Creating data loader.")
data_loader = train_dataset.DataLoader(opt.datasets, dataset_mode='2afc', batch_size=opt.batch_size, serial_batches=False)


# Start session.
print("Starting session")
saver = tf.train.Saver(tf.trainable_variables())

gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)


def saveWeights(sess):
	# Save the LPIPS weights.
	linear_weight_as_dict = sess.run(model.network.linear_weight_as_dict)
	linear_weight_as_dict = {key: np.maximum(0.0, value) for key, value in linear_weight_as_dict.items()}
	np.save('./saves/{}-latest.npy'.format(opt.name), linear_weight_as_dict)
	
	# # Save the small network weights too.
	# bce = sess.run(bce_params)
	# np.save('./saves/{}-latest-bce.npy'.format(opt.name), bce)


with tf.Session(config=session_config) as sess:
	sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

	# Modifying the graph is not allowed after this.
	tf.get_default_graph().finalize()
	
	total_steps = 0

	for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
		epoch_start_time = time.time()
		for i, data in enumerate(data_loader.epoch()):			
			total_steps += opt.batch_size
				
			lr, smooth_loss, smooth_accuracy, accuracy, loss, _ = sess.run([tf_actual_learning_rate, tf_smooth_loss, tf_smooth_accuracy, tf_compute_accuracy, tf_loss, tf_minimize], feed_dict={
				tf_X_image: data['p0'],
				tf_Y_image: data['p1'],
				tf_ref_image: data['ref'],
				tf_judge: data['judge']
			})
			
			sess.run([tf_clamp_weights, tf_increase_step])

			if total_steps % opt.display_freq == 0:
				print("{}: lr {}  smooth loss: {}. loss: {}.  accuracy: {}".format(i, lr, smooth_loss, loss, 100.0 * smooth_accuracy))
			
			if total_steps % opt.save_latest_freq == 0:
				print('saving the latest model (epoch %d, total_steps %d)' %
					  (epoch, total_steps))
				saver.save(sess, './saves/{}-latest.ckpt'.format(opt.name))
				saveWeights(sess)
				
		print('saving the model at the end of epoch %d, iters %d' %
			  (epoch, total_steps))
		saver.save(sess, './saves/{}-latest.ckpt'.format(opt.name), global_step=0)
		saver.save(sess, './saves/{}.ckpt'.format(opt.name), global_step=i)
		saveWeights(sess)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

		if epoch > opt.nepoch:
			updateLearningRate(sess, opt.nepoch_decay)
