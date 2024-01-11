class CityScapes(torch.utils.data.Dataset):
    def __init__(self, dataroot, data_split, task, transform, num_class=None, text_file=None):
        print(self.name())
        if isinstance(task, str):
            task = [task]
        json_file = os.path.join(dataroot, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.groups = info[data_split]
        self.data_split = data_split
        if text_file is not None:
            text_file_abs = os.path.join(dataroot, text_file)
            with open(text_file_abs, 'rb') as f:
                lines = f.readlines()
            groups = []
            for line in lines:
                idx = int(line.strip())
                group = ['train/image/%d.npy' % idx, 'train/depth/%d.npy'%idx, 'train/label_2/%d.npy' % idx,
                         'train/label_7/%d.npy' % idx, 'train/label_19/%d.npy' % idx]
                groups.append(group)
            self.groups = groups
        self.task = task

        self.crop_h = 224
        self.crop_w = 224
        self.transform = transform

    def __len__(self):
        return len(self.groups)
        # return 16
        # return 6

    # @staticmethod
    def __scale__(self, img, labels):
        """
             Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        new_labels = []
        for label in labels:
            label = np.expand_dims(
                cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST),
                axis=-1)
            new_labels.append(label)
        return img_new, new_labels

    @staticmethod
    def __mirror__(img, labels):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            labels = [label[:, ::-1] for label in labels]
        return img, labels

    def __random_crop_and_pad_image_and_labels__(self,
                                                 img,
                                                 labels,
                                                 crop_h,
                                                 crop_w,
                                                 ignore_label=-1.0):
        # combining
        img_and_label = [img] + labels
        combined = np.concatenate(img_and_label, axis=2)
        image_shape = img.shape
        label_cdims = [label.shape[-1] for label in labels]
        # padding to the crop size
        pad_shape = [
            max(image_shape[0], crop_h),
            max(image_shape[1], crop_w), combined.shape[-1]
        ]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0]) // 2, (
                pad_shape[1] - image_shape[1]) // 2
        combined_pad[offset_h:offset_h + image_shape[0],
        offset_w:offset_w + image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0,
                                             crop_offset_h + 1), np.random.randint(
            0, crop_offset_w + 1)
        combined_crop = combined_pad[start_h:start_h + crop_h,
                        start_w:start_w + crop_w]
        # separating
        img_cdim = image_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        label_crops = []
        label_cdim_count = 0

        for label_cdim in label_cdims:
            label_crop = deepcopy(combined_crop[:, :, img_cdim + label_cdim_count:img_cdim + label_cdim_count +
                                                               label_cdim]).astype('float')
            label_cdim_count += label_cdim
            label_crops.append(label_crop)

        return img_crop, label_crops

    def __getitem__(self, item):
        img_path, depth_path, label2_path, label7_path, label19_path = self.groups[item]
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        labels = [None for _ in self.task]
        if 'depth' in self.task:
            depth = np.load(os.path.join(self.dataroot, depth_path))
            depth_index = self.task.index('depth')
            labels[depth_index] = depth
        if 'seg' in self.task:
            label19 = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
            seg_index = self.task.index('seg')

            labels[seg_index] = label19
        if self.data_split == 'train':
            img, labels = self.__scale__(img, labels)
            img, labels = self.__mirror__(img, labels)
            # labels = [label[:, :, None] for label in labels]
            img, labels = self.__random_crop_and_pad_image_and_labels__(img, labels, self.crop_h, self.crop_w)

        image = Image.fromarray(np.uint8(img))
        image = self.transform(image)
        labels = [torch.from_numpy(label) for label in labels]
        if 'seg' in self.task:
            seg_index = self.task.index('seg')
            labels[seg_index] = labels[seg_index].long()

        labels = [label.permute(2, 0, 1) for label in labels]

        return image, labels, img_path

    def name(self):
        return 'CityScapes'


if __name__ == '__main__':
  root_path = '/projectnb/ivc-ml/sunxm/datasets/cityscapes'
  data_split = 'train'
  task = ['seg', 'depth']
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  import torchvision.transforms as transforms

  normalize = transforms.Normalize(mean=mean, std=std)
  transform = transforms.Compose([
      transforms.ToTensor(),
      normalize,
  ])

  text_file = 'lists/train.txt'

  dataset = CityScapes(root_path, data_split, task, transform, text_file=text_file)
  img, label, image_key = dataset[70]
  import pdb
  pdb.set_trace()