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
    '-o', '--output_path', help='location of output image folder.', type=str)
parser.add_argument(
    '-n', '--num_init_gt', help='location of output image folder.', type=int)


args = parser.parse_args()


def create_init_set():
  data_list = osp.join(args.data_path, 'train.txt')
  with open(data_list) as f:
    full_list = f.readlines()

  # create lmdb
  args.output_path = os.path.join(args.output_path, 'init_gt')
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  random.shuffle(full_list)
  labelled_set = full_list[:args.num_init_gt]
  unlabelled_set = full_list[args.num_init_gt:]
  rand_n = int(time.time())
  file_labelled = os.path.join(
      args.output_path, 'labelled_set_{}_{}.txt'.format(args.num_init_gt,
                                                        rand_n))
  file_unlabelled = os.path.join(
      args.output_path,
      'unlabelled_set_{}_{}.txt'.format(args.num_init_gt, rand_n))

  with open(file_unlabelled, 'w+') as f:
    f.writelines(unlabelled_set)

  with open(file_labelled, 'w+') as f:
    f.writelines(labelled_set)

if __name__ == '__main__':
  create_init_set()

