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
    '-f', '--file_index', help='file index for the data split .', type=str)
parser.add_argument(
    '-n', '--final_number', help='the name of final labelled images.', type=int)

parser.add_argument(
    '-o', '--output_folder', help='the name of final output folder.', type=str)

args = parser.parse_args()


def random_sample():
  # read in labelled data
  labelled_path = os.path.join(args.data_path, 'init_gt', 'labelled_set_{}.txt'.format(args.file_index))
  with open(labelled_path) as f:
    labelled_set = f.readlines()
    num_labelled = len(labelled_set)

  # read in unlabelld data
  unlabelled_path = os.path.join(args.data_path, 'init_gt', 'unlabelled_set_{}.txt'.format(args.file_index))
  with open(unlabelled_path) as f:
    unlabelled_set = f.readlines()
    num_unlabelled = len(unlabelled_set)

  new_picked = args.final_number - num_labelled
  assert new_picked > 0 and new_picked <= num_unlabelled

  random.shuffle(unlabelled_set)
  new_items = unlabelled_set[:new_picked]
  new_labelled_set = labelled_set + new_items

  if not os.path.exists(os.path.join(args.data_path, args.output_folder)):
    os.makedirs(os.path.join(args.data_path, args.output_folder))
  file_labelled = os.path.join(args.data_path, args.output_folder,
                               'random{}_{}_{}.txt'.format(args.final_number, args.file_index, int(time.time())))

  with open(file_labelled, 'w+') as f:
    for l in new_labelled_set:
      f.writelines(l)

if __name__ == '__main__':
  random_sample()
