"""python PASCAL_VOC_joint.py -d /projectnb/ivc-ml/sunxm/datasets/cityscapes/lists/ -f 300_1691347206 -n 390 -o 690_300 -g 1"""
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
    '-f',
    '--file_indices',
    help='file index for the data split .',
    type=str,
    nargs='+')
parser.add_argument(
    '-n', '--new_number', help='the name of final labelled images.', type=int)

parser.add_argument(
    '-g', '--group', help='the name of final labelled images.', type=int)

parser.add_argument(
    '-o',
    '--output_folder',
    help='the output folder',
    type=str)


args = parser.parse_args()


def random_sample():
  # read in labelled data
  labelled_sets = []
  num_labelleds = []
  overall_unlabelled_set = None
  for file_index in args.file_indices:
    labelled_path = os.path.join(args.data_path, 'init_gt',
                                 'labelled_set_{}.txt'.format(file_index))
    with open(labelled_path) as f:
      labelled_set = f.readlines()
      labelled_sets.append(labelled_set)
      num_labelled = len(labelled_set)
      num_labelleds.append(num_labelled)
    # read in unlabelld data
    unlabelled_path = os.path.join(args.data_path, 'init_gt',
                                   'unlabelled_set_{}.txt'.format(file_index))
    with open(unlabelled_path) as f:
      unlabelled_set = f.readlines()
      if overall_unlabelled_set is None:
        overall_unlabelled_set = unlabelled_set
      else:
        overall_unlabelled_set = list(
            set(overall_unlabelled_set) & set(unlabelled_set))

  assert args.new_number <= len(overall_unlabelled_set)

  random.shuffle(overall_unlabelled_set)
  new_items = overall_unlabelled_set[:args.new_number]

  for f_id, file_index in enumerate(args.file_indices):
    new_labelled_set = labelled_sets[f_id] + new_items

    if not os.path.exists(os.path.join(args.data_path, args.output_folder)):
      os.makedirs(os.path.join(args.data_path, args.output_folder))
    file_labelled = os.path.join(
        args.data_path, args.output_folder,
        'random{}_{}_{}.txt'.format(num_labelleds[f_id] + args.new_number,
                                    file_index, args.group))

    with open(file_labelled, 'w+') as f:
      for l in new_labelled_set:
        f.writelines(l)

if __name__ == '__main__':
  random_sample()
