import os
import shutil
import random

dataset_path = "/teamspace/studios/this_studio/AerialSegmentation/Semantic segmentation dataset"
new_dataset_path = "/teamspace/studios/this_studio/AerialSegmentation"
train_path = os.path.join(new_dataset_path, "train")
val_path = os.path.join(new_dataset_path, "val")

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

train_image_path = os.path.join(train_path, "images")
train_mask_path = os.path.join(train_path, "masks")

val_image_path = os.path.join(val_path, "images")
val_mask_path = os.path.join(val_path, "masks")

os.makedirs(train_image_path, exist_ok=True)
os.makedirs(val_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(val_mask_path, exist_ok=True)


tile_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

n_train_images = 8
n_val_images = 1


def copy(train_status):
    if train_status:
        images = train_images
        path_image = train_image_path
        path_mask = train_mask_path
    else:
        images = val_images
        path_image = val_image_path
        path_mask = val_mask_path

    

    for image in images:
        tile_image_name = f'{tile_folder}_{image}'
        shutil.copy(os.path.join(images_path, image), os.path.join(path_image, tile_image_name))

        mask_name = image.split('.')[0]+'.png'
        tile_mask_name = f'{tile_folder}_{mask_name}'
        shutil.copy(os.path.join(masks_path, mask_name), os.path.join(path_mask,  tile_mask_name))
        

for tile_folder in tile_folders:
    images_path = os.path.join(dataset_path, tile_folder, 'images')
    masks_path = os.path.join(dataset_path, tile_folder, 'masks')

    images = os.listdir(images_path)
    masks = os.listdir(masks_path)

    random.shuffle(images)
    random.shuffle(masks)

    train_images = images[:n_train_images]
    val_images = images[n_train_images:]

    copy(train_status=True)
    copy(train_status=False)


shutil.rmtree(dataset_path)

print(f"Data organization and split completed successfully. Total Training Files is {len(os.listdir(train_image_path))} and Validation Files is {len(os.listdir(val_image_path))}")
