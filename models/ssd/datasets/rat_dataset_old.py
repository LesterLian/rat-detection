import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
from torch.utils.data import Dataset


class RATDataset(Dataset):
    def __init__(self,
                 root,
                 trainval_path,
                 test_path,
                 transform=None,
                 target_transform=None,
                 is_test=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            images_sets_file = test_path
        else:
            images_sets_file = trainval_path
        self.ids = RATDataset._read_image_ids(images_sets_file)
        self.class_names = ('BACKGROUND', 'rat')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def __len__(self):
        return len(self.ids)

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def _get_annotation(self, image_id):
        # annotation_file = self.root / f"Annotations/
        try:
            annotation_file = os.path.join(self.root, f"CHTXrat{image_id}.xml") # TODO CHECK THIS
            objects = ET.parse(annotation_file).findall("object")
        except:
            annotation_file = os.path.join(self.root, f"CHTXrat{image_id}.xml")
            objects = ET.parse(annotation_file).findall("object")
            print("load annotation error")

        boxes = []
        labels = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()

            if class_name in self.class_dict:
                bbox = object.find('bndbox')
                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])
                labels.append(self.class_dict[class_name])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def _read_image(self, image_id):
        try:
            image_file = "./data/VOCdevkit/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image_file = "./data/VOCdevkit/test/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            image = cv2.imread(str(image_file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("read image error")
        return image

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids
