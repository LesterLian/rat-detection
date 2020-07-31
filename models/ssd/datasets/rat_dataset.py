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
                 images_path,
                 xmls_path,

                 transform=None,
                 target_transform=None,
                 is_test=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.images = RATDataset._read_images_path(images_path)
        self.xmls = RATDataset._read_xmls_path(xmls_path)
        self.class_names = ('BACKGROUND', 'rat')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_path = self.images[index]
        xml_path = self.xmls[index]
        boxes, labels = self._get_annotation(xml_path)
        image = self._read_image(image_path)
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

    def _get_annotation(self, xml_path):
        objects = ET.parse(xml_path).findall("object")
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

    def _read_image(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def _read_images_path(images_path):
        images = []
        with open(images_path) as f:
            for line in f:
                images.append(line.rstrip())
        return images[:-1]

    @staticmethod
    def _read_xmls_path(xmls_path):
        xmls = []
        with open(xmls_path) as f:
            for line in f:
                xmls.append(line.rstrip())
        return xmls[:-1]
