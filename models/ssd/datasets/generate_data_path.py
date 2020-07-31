import glob
import os.path as osp
import sys
import config
import numpy as np
import xml.etree.ElementTree as ET


def main(root):
    if not osp.exists(root):
        print("dataset path is wrong.")
        sys.exit(1)

    xmls = glob.glob(f"{root}/*/*.xml")
    tmp_xmls = []
    for xml_path in xmls:
        objects = ET.parse(xml_path).findall("object")
        boxes = []
        for object in objects:
            bbox = object.find('bndbox')
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
        if len(boxes) != 0:
            tmp_xmls.append(xml_path)
    xmls = tmp_xmls
    images = [xml[:-3] + 'jpg' for xml in xmls]
    images = np.array(images)
    xmls = np.array(xmls)
    total_size = len(images)

    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    shuffled_idx = np.arange(total_size)
    np.random.shuffle(shuffled_idx)

    train_images = images[shuffled_idx[:train_size]]
    train_xmls = xmls[shuffled_idx[:train_size]]

    val_images = images[shuffled_idx[train_size:]]
    val_xmls = xmls[shuffled_idx[train_size:]]

    print(f"Train dataset size: {train_size}, Val dataset size: {val_size}")

    with open(f"{root}/train_images.txt", 'w') as path:
        for image in train_images:
            path.write(image + "\n")

    with open(f"{root}/train_xmls.txt", 'w') as path:
        for train_xml in train_xmls:
            path.write(train_xml + "\n")

    with open(f"{root}/val_images.txt", 'w') as path:
        for val_image in val_images:
            path.write(val_image + "\n")

    with open(f"{root}/val_xmls.txt", 'w') as path:
        for val_xml in val_xmls:
            path.write(val_xml + "\n")

    print('generate_data_path finished!')


if __name__ == '__main__':
    main(config.DATASET_ROOT_PATH)
