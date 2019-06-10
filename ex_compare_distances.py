# Compares which one of two images is closer to a given referece image.
#
# Usage:
#    python ex_compare_distances.py reference_image image1 image2
#
#    python ex_compare_distances.py --metric=[elpips_vgg|lpips_vgg|lpips_squeeze] reference_image image1 image2
#


import argparse

import tensorflow as tf
import numpy as np
import imageio

import elpips


# Command line parameters.
parser = argparse.ArgumentParser()
parser.add_argument('reference_image', type=str, nargs=1, help='reference image to compare against')
parser.add_argument('image', type=str, nargs=2, help='two images that are compared to the reference image')
parser.add_argument('--metric', type=str, default='elpips_vgg', help='elpips_vgg (default), lpips_vgg or lpips_squeeze')
parser.add_argument('-n', type=int, default=30, help='number of samples to use for E-LPIPS. Default: 30')
args = parser.parse_args()

if args.metric not in ('elpips_vgg', 'lpips_vgg', 'lpips_squeeze'):
	raise Exception('Unsupported metric')

	
# Load images.
reference_image = imageio.imread(args.reference_image[0])[:,:,0:3].astype(np.float32) / 255.0
image1 = imageio.imread(args.image[0])[:,:,0:3].astype(np.float32) / 255.0
image2 = imageio.imread(args.image[1])[:,:,0:3].astype(np.float32) / 255.0

assert image1.shape == reference_image.shape
assert image2.shape == reference_image.shape


# Create the distance metric.
if args.metric == 'elpips_vgg':
	# Use E-LPIPS-VGG averages over n samples.
	config = elpips.elpips_vgg(batch_size=1, n=args.n)
	config.set_scale_levels_by_image_size(reference_image.shape[0], reference_image.shape[1])	
	metric = elpips.Metric(config, back_prop=False)
elif args.metric == 'lpips_vgg':
	# Use LPIPS-VGG.
	metric = elpips.Metric(elpips.lpips_vgg(1), back_prop=False)
elif args.metric == 'lpips_squeeze':
	# Use LPIPS-SQUEEZENET.
	metric = elpips.Metric(elpips.lpips_squeeze(1), back_prop=False)
else:
	raise Exception('Unknown metric')


# Create the computation graph.
print("Creating computation graph.")
tf_reference_image = tf.placeholder(tf.float32)
tf_image1 = tf.placeholder(tf.float32)
tf_image2 = tf.placeholder(tf.float32)
tf_evaluate_distances_with_correlated_noise = metric.forward((tf_image1, tf_image2), tf_reference_image)
	
	
# Run.	
print("Running graph.")
gpu_options = tf.GPUOptions(allow_growth=True)
session_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

with tf.Session(config=session_config) as sess:
	sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
	
	# Comparisons of E-LPIPS distances are much more precise when they have been evaluated simultaneously, with e same random transforms
	# and dropout.
	correlated_distances_in_minibatch = sess.run(tf_evaluate_distances_with_correlated_noise, feed_dict={
		tf_reference_image: np.expand_dims(reference_image, axis=0), # convert into NHWC tensors
		tf_image1: np.expand_dims(image1, axis=0),
		tf_image2: np.expand_dims(image2, axis=0)
	})	
	
	batch1_distances = correlated_distances_in_minibatch[0]
	image1_distance = batch1_distances[0]
	
	batch2_distances = correlated_distances_in_minibatch[1]
	image2_distance = batch2_distances[0]
	
	if image1_distance < image2_distance:
		print("Image '{}' is closer! (*{:.05f}* - {:.05f} = {:.05f} < 0)".format(args.image[0], image1_distance, image2_distance, image1_distance - image2_distance))
	elif image1_distance > image2_distance:
		print("Image '{}' is closer! ({:.05f} - *{:.05f}* = {:.05f} > 0)".format(args.image[1], image1_distance, image2_distance, image1_distance - image2_distance))
	else:
		print("Images are equally close! ({:.05f} - {:.05f} = {:.05f})".format(args.image[1], image1_distance, image2_distance, image1_distance - image2_distance))

