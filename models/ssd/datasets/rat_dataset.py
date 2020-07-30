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

    def _get_annotation(self, image_id):
        # annotation_file = self.root / f"Annotations/
        try:
            annotation_file = os.path.join(self.root, f"CHTXrat{image_id}")

        @staticmethod
        def _read_image_ids(image_sets_file):
            ids = []
            with open(image_sets_file) as f:
                for line in f:
                    ids.append(line.rstrip())
            return ids
