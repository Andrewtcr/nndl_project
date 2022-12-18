from PIL import Image
import os


DATASET_DIR = 'subclass_training_data/train'


# Try converting image files to all be jpg...
directory = os.fsencode(DATASET_DIR)

for folder in os.listdir(directory):
    folder_name = os.fsdecode(folder)
    for file in os.listdir(os.fsencode(f'{DATASET_DIR}/{folder_name}')):
        # print(os.fsdecode(file))

        filename = os.fsdecode(file)
        img_path = f'{DATASET_DIR}/{folder_name}/{filename}'

        if img_path[-4:] == '.png':
            im = Image.open(img_path)
            rgb_im = im.convert('RGB')
            rgb_im.save(f'{img_path[:-4]}.jpg')

            os.remove(img_path)