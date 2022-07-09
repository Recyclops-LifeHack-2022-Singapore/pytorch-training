import os
import shutil
import random

SRC_DATASET = './data/compiled_dataset'
DST_DATASET = './data/dataset'
TRAIN_RATIO = 0.8

def main():
    create_dir(DST_DATASET)

    data_classes = os.listdir(SRC_DATASET)

    train_count = 0
    test_count = 0

    for data_class in data_classes:
        print("Parsing {}...".format(data_class))
        class_path = os.path.join(SRC_DATASET, data_class)

        # create destination dirs
        train_class_path = os.path.join(DST_DATASET, 'train', data_class)
        test_class_path = os.path.join(DST_DATASET, 'test', data_class)
        create_dir(os.path.dirname(train_class_path))
        create_dir(os.path.dirname(test_class_path))
        create_dir(train_class_path)
        create_dir(test_class_path)

        # shuffle all images of a class
        image_files = os.listdir(class_path)
        random.shuffle(image_files) 

        # split into train / test
        train_size = int(len(image_files) * TRAIN_RATIO)
        train_image_paths = image_files[:train_size]
        test_image_paths = image_files[train_size:]

        train_count += len(train_image_paths)
        test_count += len(test_image_paths)

        save_files(train_image_paths, class_path, train_class_path)
        save_files(test_image_paths, class_path, test_class_path)

    print("-"*20)
    print("Completed")
    print("Train Images: {}".format(train_count))
    print("Test Images: {}".format(test_count))
    print("Ratio: {}".format(train_count / (train_count + test_count)))
    print("-"*20)

def save_files(files, src, dst):
    for image_file in files:
        src_path = os.path.join(src, image_file)
        dst_path = os.path.join(dst, image_file)

        shutil.copy(src_path, dst_path)

def create_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == '__main__':
    main()