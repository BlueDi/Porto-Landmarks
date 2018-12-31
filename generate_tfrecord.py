from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
import dataset_utils
from collections import namedtuple, OrderedDict


flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def class_text_to_int(row_label):
  '''Define classes to be detected'''
  if row_label == 'arrabida':
    return 1
  elif row_label == 'camara':
    return 2
  elif row_label == 'clerigos':
    return 3
  elif row_label == 'musica':
    return 4
  elif row_label == 'serralves':
    return 5
  else:
    None


def split(df, group):
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
  with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)
  width, height = image.size

  filename = group.filename.encode('utf8')
  image_format = b'jpg'
  xmins = []
  xmaxs = []
  ymins = []
  ymaxs = []
  classes_text = []
  classes = []

  for index, row in group.object.iterrows():
    xmins.append(row['xmin'] / width)
    xmaxs.append(row['xmax'] / width)
    ymins.append(row['ymin'] / height)
    ymaxs.append(row['ymax'] / height)
    classes_text.append(row['class'].encode('utf8'))
    classes.append(class_text_to_int(row['class']))

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_utils.int64_feature(height),
    'image/width': dataset_utils.int64_feature(width),
    'image/filename': dataset_utils.bytes_feature(filename),
    'image/source_id': dataset_utils.bytes_feature(filename),
    'image/encoded': dataset_utils.bytes_feature(encoded_jpg),
    'image/format': dataset_utils.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_utils.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_utils.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_utils.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_utils.float_list_feature(ymaxs),
    'image/object/class/text': dataset_utils.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_utils.int64_list_feature(classes),
  }))
  return tf_example


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  path = os.path.join(os.getcwd(), FLAGS.image_dir)
  examples = pd.read_csv(FLAGS.csv_input)
  grouped = split(examples, 'filename')
  for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

  writer.close()
  output_path = os.path.join(os.getcwd(), FLAGS.output_path)
  print('Successfully created the TFRecords: {}'.format(output_path))


tf.app.run()