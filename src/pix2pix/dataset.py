import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from . import config

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        output_image = image[ : , : 256, : ]
        input_image = image[ : , 256 : , : ]

        aug = config.both_transform(image = input_image, image0 = output_image)
        input_image, output_image = aug["image"], aug["image0"]

        input_image = config.transform_only_input(image = input_image)["image"]
        output_image = config.transform_only_input(image = output_image)["image"]

        return input_image, output_image
