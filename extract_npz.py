import numpy as np
import os
from PIL import Image

NPZ_FILE = '../data_archive/recycle_data_shuffled.npz'
OUT_DIR = './data/extracted'

label_map = {
    0: 'boxes', 
    1: 'glass_bottles', 2: 'soda_cans', 
    3: 'crushed_soda_cans', 
    4: 'water_bottles'
}

def unpickle(file):
    with open(file, 'rb') as fo:
        data = np.load(fo)
        x, y = data['x_train'], data['y_train']
        return x, y

if __name__ == '__main__':
    data, labels = unpickle(NPZ_FILE)

    for i in range(data.shape[0]):
        image = data[i]
        label = labels[i]
        label_dir_path = os.path.join(OUT_DIR, label_map[label[0]])
        if not os.path.isdir(label_dir_path):
            os.mkdir(label_dir_path)

        im = Image.fromarray(image)
        im.save(os.path.join(label_dir_path, '{}_{}.jpg'.format(label, i)))

    print("Completed")