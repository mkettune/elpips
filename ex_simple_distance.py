# Evaluates the E-LPIPS distance between two images. Simple version.
#
# Usage:
#    python ex_simple_distance.py image1 image2
#
# To evaluate LPIPS distance:
#    python ex_simple_distance.py image1 image2 --metric=[elpips_vgg|lpips_vgg|lpips_squeeze]
#
#
# Notes:
# 	Back prop is disabled here for performance since we are not optimizing through the metric.
#
# 	When back prop is required: 
#       - Decrease 'n' in elpips.elpips_vgg(...) to 1-3 to save memory.
#       - Setting n=1 is a good default when noisy estimates are acceptable.
#
#   When dealing with small images:
#       - Limit the maximum downscaling level: image_size / max_scale_level should never be under 64.
#
#         To do this:
#             config = elpips.elpips_vgg(...);
#             config.set_scale_levels_by_image_size(image_h, image_w)
#
#   Example for e.g. neural network training:
# 	    config = elpips.elpips_vgg(batch_size=BATCH_SIZE, n=1)
# 	    config.set_scale_levels_by_image_size(CROP_SIZE, CROP_SIZE)
# 	    metric = elpips.Metric(config, back_prop=True)
#       loss = metrics.forward(prediction, ground_truth)

import argparse

import tensorflow as tf
import numpy as np
import imageio

import elpips


# Command line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, nargs=2, help='two images whose distance needs to be evaluated')
parser.add_argument('--metric', type=str, default='elpips_vgg', help='elpips_vgg (default), lpips_vgg or lpips_squeeze')
parser.add_argument('-n', type=int, default=200, help='number of samples to use for E-LPIPS. Default: 200')
args = parser.parse_args()

if args.metric not in ('elpips_vgg', 'lpips_vgg', 'lpips_squeeze'):
	raise Exception('Unsupported metric')

	
# Load images.
image1 = imageio.imread(args.image[0])[:,:,0:3].astype(np.float32) / 255.0
image2 = imageio.imread(args.image[1])[:,:,0:3].astype(np.float32) / 255.0

assert image1.shape == image2.shape


# Create the distance metric.
if args.metric == 'elpips_vgg':
	# Use E-LPIPS averages over n samples.
	metric = elpips.Metric(elpips.elpips_vgg(batch_size=1, n=args.n), back_prop=False)	
elif args.metric == 'lpips_vgg':
	# Use LPIPS-VGG.
	metric = elpips.Metric(elpips.lpips_vgg(1), back_prop=False)
elif args.metric == 'lpips_squeeze':
	# Use LPIPS-SQUEEZENET.
	metric = elpips.Metric(elpips.lpips_squeeze(1), back_prop=False)
else:
	raise Exception('Unspported metric')

	
# Create the computation graph.
print("Creating computation graph.")
tf_image1 = tf.placeholder(tf.float32)
tf_image2 = tf.placeholder(tf.float32)
tf_evaluate_distance = metric.forward(tf_image1, tf_image2)
	
	
# Run.	
print("Running graph.")
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=session_config) as sess:
	sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
	
	# Run E-LPIPS.
	distances_in_minibatch = sess.run(tf_evaluate_distance, feed_dict={
		tf_image1: np.expand_dims(image1, axis=0), # convert to NHWC tensors
		tf_image2: np.expand_dims(image2, axis=0)
	})	

	print("Distance ({}): {}".format(args.metric, distances_in_minibatch[0]))

