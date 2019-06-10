"""Generic inference script that inferences a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('master', '',
                           'The address of the TensorFlow master to use.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', 'train_logs/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string('labels_path', '', 'Result path.')
tf.app.flags.DEFINE_string('images_list', '', 'Test image list.')
tf.app.flags.DEFINE_string('images_dir', '.', 'Test image directory.')
tf.app.flags.DEFINE_integer('num_classes', 5, 'Number of classes.')
tf.app.flags.DEFINE_string('model_name', 'inception_v3',
                           'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None,
    'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer('infer_image_size', None, 'Eval image size')
FLAGS = tf.app.flags.FLAGS


def main(_):
    """main."""
    if FLAGS.images_list:
        images_names = [line.split()[0] for line in open(FLAGS.images_list)]
    else:
        images_names = os.listdir(FLAGS.images_dir)
    if FLAGS.labels_path:
        labels = [line.strip() for line in open(FLAGS.labels_path)]
    else:
        labels = None
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(FLAGS.model_name,
                                                 num_classes=FLAGS.num_classes,
                                                 is_training=False)
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)
        infer_image_size = FLAGS.infer_image_size or network_fn.default_image_size
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path
        tensor_input = tf.placeholder(
            tf.float32, [None, infer_image_size, infer_image_size, 3])
        logits, _ = network_fn(tensor_input)
        logits = tf.nn.top_k(logits, 1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print(time.ctime())
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            for image_name in images_names:
                image_path = os.path.join(FLAGS.images_dir, image_name)
                image = open(image_path, 'rb').read()
                image = tf.image.decode_jpeg(image, channels=3)
                processed_image = image_preprocessing_fn(
                    image, infer_image_size, infer_image_size)
                processed_image = sess.run(processed_image)
                images = np.array([processed_image])
                prediction = sess.run(logits, feed_dict={
                    tensor_input: images
                }).indices[0][0]
                if labels:
                    prediction = labels[prediction]
                print('{} {}'.format(image_name, prediction))
        print(time.ctime())


if __name__ == '__main__':
    tf.app.run()
