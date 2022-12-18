from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import Image, Dataset
import os


DATASET_DIR = 'subclass_training_data/train'


# Hangs like hell and i dont know why
# dataset = load_dataset('imagefolder', data_dir='D:/Homework Assignments/NNDL/project/subclass_training_data', cache_dir='./')

training_images = []
training_labels = []
label_names = []

directory = os.fsencode(DATASET_DIR)

for folder in os.listdir(directory):
    folder_name = os.fsdecode(folder)
    label_names.append(folder_name)
    for file in os.listdir(os.fsencode(f'{DATASET_DIR}/{folder_name}')):
        # print(os.fsdecode(file))

        filename = os.fsdecode(file)
        img_path = f'{DATASET_DIR}/{folder_name}/{filename}'

        training_images.append(img_path)
        training_labels.append(label_names.index(folder_name))

dataset = Dataset.from_dict({'image': training_images, 'label': training_labels}).cast_column("image", Image())

dataset.shuffle(seed=42)
print(dataset[0])


### FROM VIT CARD https://huggingface.co/google/vit-base-patch16-224 ###
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = dataset[0]['image']

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

