import csv
import os
import shutil


DATASET_DIR = 'subclass_training_data'
Y_TRAIN_PATH = 'neuralnet_subclasses_competition_data/y_train.csv'
Y_GEN_PATH = 'neuralnet_subclasses_competition_data/y_generated.csv'
X_TRAIN_PATH = 'neuralnet_subclasses_competition_data/train_shuffle'
X_GEN_PATH = 'neuralnet_subclasses_competition_data/x_generated'


if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

labels = None
index = 0
with open(Y_TRAIN_PATH, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Labels are in the header
    labels = next(csvreader)

    i = 0
    for row in csvreader:
        label = labels[row.index('1')]

        label_path = f'{DATASET_DIR}/{label}'

        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        shutil.copyfile(f'{X_TRAIN_PATH}/{i}.jpg', f'{label_path}/{i}.jpg')

        i += 1
    index = i

with open(Y_GEN_PATH, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    # Labels are in the header
    labels = next(csvreader)

    i = 0
    for row in csvreader:
        label = labels[row.index('1')]

        label_path = f'{DATASET_DIR}/{label}'

        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        shutil.copyfile(f'{X_GEN_PATH}/{i}.png', f'{label_path}/{index}.png')

        i += 1
        index += 1
