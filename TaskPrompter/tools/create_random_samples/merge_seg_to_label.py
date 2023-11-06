"""Create lmdb from CityScapes."""
import argparse
import glob
import os
import os.path as osp
import random
import time


parser = argparse.ArgumentParser(
    description='Create video (images) into lmdb database.')
parser.add_argument(
    '-d', '--data_path', help='location of input image folder.', type=str)
parser.add_argument(
    '-fl',
    '--file_name_for_label',
    help='file index for multi_label.',
    type=str)
parser.add_argument(
    '-fs',
    '--file_name_for_seg',
    help='file index for multi_label.',
    type=str)

parser.add_argument(
    '-il', '--index_label', help='the number of labelled images of label.', type=str)
parser.add_argument(
    '-is', '--index_seg', help='the index of labelled images of segmentation.', type=str)

parser.add_argument(
    '-o', '--output_dir', help='the output folder.', type=str)
parser.add_argument(
    '-g', '--group', help='the name of final labelled images.', default=0, type=int)


args = parser.parse_args()


def random_sample():
  # read in labelled data
  labelled_path = os.path.join(args.data_path, args.file_name_for_label)
  with open(labelled_path) as f:
    label_set = f.readlines()

  seg_path = os.path.join(args.data_path, args.file_name_for_seg)
  with open(seg_path) as f:
    seg_set = f.readlines()

  intersection = list(set(label_set) | set(seg_set))
  length_of_labelled = len(intersection)


  output_folder = os.path.join(args.data_path, args.output_dir)
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  output_file = os.path.join(
      output_folder, 'random%d_%s_%s_%d.txt' %
      (length_of_labelled, args.index_label, args.index_seg, args.group))
  print('the length of overall labelled data = %d' % length_of_labelled)
  with open(output_file, 'w+') as f:
    for l in intersection:
      f.writelines(l)


if __name__ == '__main__':
  random_sample()
