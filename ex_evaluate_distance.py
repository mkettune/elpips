# Evaluates the E-LPIPS distance between two images.
#
# Stops iteration when absolute and relative error requirements have been fulfilled, or the maximum number of samples has been reached.
#
# Usage:
#    python ex_evaluate_distance.py image1 image2

import argparse

import tensorflow as tf
import numpy as np
import imageio

import elpips


# Command line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, nargs=2, help='two images whose distance needs to be evaluated')
parser.add_argument('--metric', type=str, default='elpips_vgg', help='elpips_vgg (default), elpips_squeeze_maxpool (experimental)')
parser.add_argument('--max_relative_error', type=float, default=0.025, help='set tolerance in stopping condition 1.96 σ < µ * max_relative_error. Default: 0.025')
parser.add_argument('--max_absolute_error', type=float, default=0.01, help='set tolerance in stopping condition 1.96 σ < max_absolute_error. Default: 0.01')
parser.add_argument('--max_iterations', type=int, default=5000, help='maximum number of iterations regardless of maximum absolute and relative error. Default: 2000')
parser.add_argument('--batch_size', type=int, default=10, help='number of copies of the images to process at once. Decrease if memory is an issue. Default: 10.')
args = parser.parse_args()

if args.metric not in ('elpips_vgg', 'elpips_squeeze_maxpool'):
	raise Exception('Unsupported metric')


BATCH_SIZE = args.batch_size

	
# Load images.
image1 = imageio.imread(args.image[0])[:,:,0:3].astype(np.float32) / 255.0
image2 = imageio.imread(args.image[1])[:,:,0:3].astype(np.float32) / 255.0

assert image1.shape == image2.shape


# Create the distance metric.
if args.metric == 'elpips_vgg':
	config = elpips.elpips_vgg(batch_size=BATCH_SIZE, n=1)
elif args.metric == 'elpips_squeeze_maxpool':
	confi = elpips.elpips_squeeze_maxpool(batch_size=BATCH_SIZE, n=1)	
else:
	raise Exception('Unsupported metric')

config.set_scale_levels_by_image_size(image1.shape[0], image1.shape[1])	
metric = elpips.Metric(config, back_prop=False)

	
# Create the computation graph.
print("Creating computation graph.")
tf_image1 = tf.placeholder(tf.float32)
tf_image2 = tf.placeholder(tf.float32)

# Extend single images into small minibatches to take advantage of the implementation's Latin Hypercube Sampling.
tf_input1 = tf.tile(tf_image1, [BATCH_SIZE, 1, 1, 1])
tf_input2 = tf.tile(tf_image2, [BATCH_SIZE, 1, 1, 1])

tf_evaluate_distance = metric.forward(tf_input1, tf_input2)


# Run.	
print("Running graph.")
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=session_config) as sess:
	sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
	
	# Run E-LPIPS.
	last_report = 0
	distances = []
	
	print("Evaluating E-LPIPS until abs_error < {} and rel_error < {}, but at most {} iterations.".format(args.max_absolute_error, args.max_relative_error, args.max_iterations))
	
	for sample_index in range(0, 0 + args.max_iterations, BATCH_SIZE):
		distances_in_minibatch = sess.run(tf_evaluate_distance, feed_dict={
			tf_image1: np.expand_dims(image1, axis=0), # convert into NHWC tensors
			tf_image2: np.expand_dims(image2, axis=0)
		})	
		distances.extend(distances_in_minibatch.tolist())

		# Stop early if absolute and relative errors are small.
		if len(distances) >= 10:
			mean = np.mean(distances)
			stddev_of_mean = np.std(distances, ddof=1) / np.sqrt(len(distances))
			
			relative_error_satistfied = 1.96 * stddev_of_mean < args.max_relative_error * mean
			absolute_error_satistfied = 1.96 * stddev_of_mean < args.max_absolute_error
			if relative_error_satistfied and absolute_error_satistfied:
				break

			# Report.
			processed_sample_count = sample_index + BATCH_SIZE
			if processed_sample_count >= last_report + 50:
				last_report = processed_sample_count
				
				relative_bound = 1.96 * stddev_of_mean / (1e-12 + mean)
				absolute_bound = 1.96 * stddev_of_mean
				print("   [Processed samples: {}.  Current estimate: {} +- {} ({:.4f}%)]".format(processed_sample_count, mean, absolute_bound, 100.0 * relative_bound))

	# Report final results.
	print("Distance ({}): {}".format(args.metric, mean))

	mean = np.mean(distances)
	stddev_of_mean = np.std(distances, ddof=1) / np.sqrt(len(distances))			
	relative_bound = 1.96 * stddev_of_mean / (1e-12 + mean)
	absolute_bound = 1.96 * stddev_of_mean
	print("  +- {} or {:.4f}%  (bounds: 1.96 σ)".format(absolute_bound, 100.0 * relative_bound))
	