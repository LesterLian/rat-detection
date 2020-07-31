import glob
import os.path as osp
import sys
import config
import numpy as np


def main(root):
    if not osp.exists(root):
        print("dataset path is wrong.")
        sys.exit(1)

    images = glob.glob(f"{root}/*/*.jpg")
    xmls = glob.glob(f"{root}/*/*.xml")

    def map_f(x):
        return root + x[1:]

    map(map_f, images)
    map(map_f, xmls)

    images = np.array(images)
    xmls = np.array(xmls)
    assert len(images) == len(xmls)
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


# 12284.09
# 673.10
if __name__ == '__main__':
    main(config.DATASET_ROOT_PATH)
