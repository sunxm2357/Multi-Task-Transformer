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
    '-o', '--output_path', help='location of output image folder.', type=str)
parser.add_argument(
    '-n', '--num_init_gt', help='the number of images in the initial gt.', type=int)
parser.add_argument(
    '-f', '--full_list', help='the full list of training index.', type=str)

args = parser.parse_args()


def create_init_set():
  with open(args.full_list) as f:
      lines = f.readlines()
  random.shuffle(lines)
  # create lmdb
  args.output_path = os.path.join(args.output_path, 'init_gt')
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  labelled_set = lines[:args.num_init_gt]
  unlabelled_set = lines[args.num_init_gt:]
  rand_n = int(time.time())
  file_labelled = os.path.join(
      args.output_path, 'labelled_set_{}_{}.txt'.format(args.num_init_gt,
                                                        rand_n))
  file_unlabelled = os.path.join(
      args.output_path,
      'unlabelled_set_{}_{}.txt'.format(args.num_init_gt, rand_n))

  with open(file_unlabelled, 'w+') as f:
    for l in unlabelled_set:
      f.writelines(l)

  with open(file_labelled, 'w+') as f:
    for l in labelled_set:
      f.writelines(l)

if __name__ == '__main__':
  create_init_set()
