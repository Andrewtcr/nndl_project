import zipfile
import torch
from PIL import Image

# transforms = torch.nn.Sequential(
#     transforms.CenterCrop(10),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# )
# scripted_transforms = torch.jit.script(transforms)

# class experimental_dataset(Dataset):

#     def __init__(self, data, transform):
#         self.data = data
#         self.transform = transform

#     def __len__(self):
#         return len(self.data.shape[0])

#     def __getitem__(self, idx):
#         item = self.data[idx]
#         item = self.transform(item)
#         return item

#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#     ])




def geometric(zip_in: str, zip_out: str, img_ext: str='.jpg'):
    '''
    Gets images from a zipped folder, applies geometric transformations, and saves the new images in an output folder.
    @zip_in: path of input zip.
    @zip_out: path of output zip.
    @img_ext: extension of the image files within the zip.
    '''

    input_zip = zipfile.ZipFile(zip_in)
    input_info = input_zip.infolist()

    output_zip = zipfile.ZipFile(zip_out, 'a')

    last_img = None
    for name in input_info:
        if name.orig_filename[-4:] != img_ext:
            continue

        # Get the image to be augmented
        img = Image.open(input_zip.open(name))

        # Perform rotation
        
        
        last_img = img
    
    last_img.show()


if __name__ == '__main__':
    geometric('test_image.zip', 'test_augmented.zip')