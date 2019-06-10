import tensorflow as tf
import numpy as np

import argparse
import os
import pdb

import elpips
import train_dataset


def score_2afc_dataset(data_loader, func):
	'''Computes the 2AFC score for the dataset defined by 'data_loader' using the distance function 'func'.
	
	OUTPUTS
		[0] - 2AFC score in [0,1], fraction of time 'func' agrees with human evaluators
		[1] - dictionary with following elements
		          d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
		          gts - N array in [0,1], preferred patch selected by human evaluators
		                (closer to "0" for left patch p0, "1" for right patch p1,
		                 "0.6" means 60 percent of people preferred right patch, 40 percent preferred left)
		          scores - N array in [0,1], corresponding to what percentage function agreed with humans
	CONSTS
		N - number of test triplets in data_loader
	'''
	d0s = []
	d1s = []
	gts = []

	for (i,data) in enumerate(data_loader.epoch()):
		d0, d1 = func(data['p0'], data['p1'], data['ref'])
		d0s.append(d0)
		d1s.append(d1)
		gts.extend(data['judge'])

	d0s = np.concatenate(d0s)
	d1s = np.concatenate(d1s)
	gts = np.asarray(gts)
	
	scores = (d0s < d1s) * (1.0 - gts) + (d1s < d0s) * gts + (d1s == d0s) * 0.5

	return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))


parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, required=True, help='elpips_vgg, elpips_squeeze_maxpool, lpips_vgg or lpips_squeeze')
parser.add_argument('--datasets', type=str, nargs='+', default=['val/traditional','val/cnn','val/color','val/superres','val/deblur','val/frameinterp'], help='datasets to test - [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--batch_size', type=int, default=50, help='batch size to test at once. Default: 50')
parser.add_argument('--custom_weights', type=str, help='custom LPIPS weights for the used (E-)LPIPS network')
parser.add_argument('--average_over', type=int, default=50, help='number of evaluations for E-LPIPS distance comparisons. Small values bias the reported 2AFC score below the real accuracy. Recommended: 50 or more. Default: 50')

opt = parser.parse_args()


# Load model.
model_config = elpips.get_config(opt.metric, batch_size=opt.batch_size, n=opt.average_over)
model_config.enable_scale = False # Not much room for scaling in 64x64.

custom_weights = None if not opt.custom_weights else np.load(opt.custom_weights).item()
model = elpips.Metric(model_config, back_prop=False, custom_lpips_weights=custom_weights)


# Create session.
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
with tf.Session(config=session_config) as sess:
	sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

	# Create a wrapper to call the model.
	tf_X_image = tf.placeholder(tf.float32)
	tf_Y_image = tf.placeholder(tf.float32)
	tf_ref_image = tf.placeholder(tf.float32)
	
	tf_X = tf_X_image
	tf_Y = tf_Y_image
	tf_ref = tf_ref_image
	
	model_loss1, model_loss2 = model.forward((tf_X, tf_Y), tf_ref)
	
	tf.get_default_graph().finalize()
	
	evaluate_model = lambda image1, image2, image3: sess.run([model_loss1, model_loss2], feed_dict={tf_X_image: image1, tf_Y_image: image2, tf_ref_image: image3})

	# Run the 2AFC tests.
	for dataset in opt.datasets:
		data_loader = train_dataset.DataLoader([dataset], dataset_mode='2afc', batch_size=opt.batch_size)

		score, results_verbose = score_2afc_dataset(data_loader, evaluate_model)

		# Print results.
		print('  Dataset [%s]: %.2f' % (dataset, 100.0 * score))
