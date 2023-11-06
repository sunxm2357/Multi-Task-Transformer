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
  datafolder = osp.join(args.data_path, 'train', 'image')
  file_list = [f for f in glob.glob(datafolder + '/*.npy')]

  # create lmdb
  args.output_path = os.path.join(args.output_path, 'init_gt')
  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  full_list = []
  for image in file_list:
    name = os.path.basename(image)
    img_index = name.split('.')[0]
    full_list.append(img_index)
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
    for l in unlabelled_set:
      f.writelines(l+'\n')

  with open(file_labelled, 'w+') as f:
    for l in labelled_set:
      f.writelines(l+'\n')

if __name__ == '__main__':
  create_init_set()
