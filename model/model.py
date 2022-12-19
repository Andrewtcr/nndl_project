from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
import numpy as np
from datasets import Image, Dataset, load_metric
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)                        
from torch.utils.data import DataLoader
import torch
import os


DATASET_DIR = './data/subclass_training_data/train'


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
print(label_names)


### FROM VIT CARD https://huggingface.co/google/vit-base-patch16-224 ###
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# image = dataset[0]['image']

splits = dataset.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']

# print(train_ds['label'])

id2label = {id:label for id, label in enumerate(label_names)}
label2id = {label:id for id, label in id2label.items()}

print(id2label)
print(label2id)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=89,
                                                  id2label=id2label,
                                                  label2id=label2id)
# inputs = feature_extractor(images=image, return_tensors="pt")

print(feature_extractor.size)
resize_seq = (feature_extractor.size['height'], feature_extractor.size['width'])

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(resize_seq),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(resize_seq),
            CenterCrop(resize_seq),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

# Set the transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

# print(train_ds[:2])

metric_name = "accuracy"

args = TrainingArguments(
    'nndl_checkpoints_etc',
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    logging_dir='logs',
    remove_unused_columns=False,
)
     
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=feature_extractor,
)

trainer.train()


# train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)

# batch = next(iter(train_dataloader))
# for k,v in batch.items():
#   if isinstance(v, torch.Tensor):
#     print(k, v.shape)



# outputs = model(**inputs)
# logits = outputs.logits
# # model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

