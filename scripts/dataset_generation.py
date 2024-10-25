import os
from typing import Any
import typer
import glob
import cv2
import numpy as np
import albumentations as A

app = typer.Typer()


banknote_transforms = A.Compose([
                                A.RandomBrightnessContrast(p=0.3),
                                A.CoarseDropout(p = .7,max_height=25,max_width=25,max_holes=5),
                                A.HueSaturationValue(p=0.3),
                            ])

banknote_mask_transforms = A.Compose([
                                    A.Perspective(p=.35),
                                    A.Rotate(p=.5,limit=20),])


def get_all_variations(img):
    rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_180 = cv2.rotate(img, cv2.ROTATE_180)
    rot_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    half_1 = img[:, :img.shape[1]//2]
    half_2 = img[:, img.shape[1]//2:]
    half_1_90 = cv2.rotate(half_1, cv2.ROTATE_90_CLOCKWISE)
    half_1_180 = cv2.rotate(half_1, cv2.ROTATE_180)
    half_1_270 = cv2.rotate(half_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    half_2_90 = cv2.rotate(half_2, cv2.ROTATE_90_CLOCKWISE)
    half_2_180 = cv2.rotate(half_2, cv2.ROTATE_180)
    half_2_270 = cv2.rotate(half_2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return (img, rot_90, rot_180, rot_270, half_1, half_1_90, half_1_180, half_1_270, half_2, half_2_90, half_2_180, half_2_270)


def make_variations(original_banknotes_dir, new_banknotes_dir , class_name ):
    n = 0
    input_dir = os.path.join(original_banknotes_dir, str(class_name))
    out_dir = os.path.join(new_banknotes_dir , str(class_name))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    files = glob.glob(os.path.join(input_dir, '*.jpg'))

    for file in files:
        img = cv2.imread(file)
        vars_list = get_all_variations(img)
        for variation in vars_list:
            fname = os.path.join(out_dir, str(n)+'.jpg')
            cv2.imwrite(fname, variation)
            n += 1


def transform_mask_banknote(banknote_pre_transforms,
                            banknote_post_transforms,
                            banknote_img,
                            img_shape,
                            banknote_size,
                            scale):

    transformed_banknote = banknote_pre_transforms(image=banknote_img)['image']

    shape = transformed_banknote.shape

    banknote_size = (
        int(banknote_size[0] * scale), int(banknote_size[1] * scale))
    half_banknote_size = (
        int(banknote_size[0] * scale), int(banknote_size[1] / 2 * scale))

    if (shape[0] < shape[1]):  # the banknote is horizontal
        if (shape[0]/shape[1] > 0.7):  # the banknote is half visible
            transformed_banknote = cv2.resize(
                transformed_banknote, (half_banknote_size[1], half_banknote_size[0]))
        else:  # the banknote is fully visible
            transformed_banknote = cv2.resize(
                transformed_banknote, (banknote_size[1], banknote_size[0]))

    else:  # the banknote is vertical
        if (shape[1]/shape[0] > 0.7):  # banknote is half visible
            transformed_banknote = cv2.resize(
                transformed_banknote, (half_banknote_size[0], half_banknote_size[1]))

        else:  # the banknote is fully visible
            transformed_banknote = cv2.resize(
                transformed_banknote, (banknote_size[0], banknote_size[1]))

    r_x = np.random.randint(img_shape[0]-transformed_banknote.shape[0])
    r_y = np.random.randint(img_shape[1]-transformed_banknote.shape[1])

    temp_img = np.zeros((img_shape[0], img_shape[1], 3),
                             dtype=np.uint8)
    mask = np.ones((img_shape[0], img_shape[1], 3),
                   dtype=np.uint8)

    temp_img[r_x:r_x+transformed_banknote.shape[0], r_y:r_y+transformed_banknote.shape[1]] = transformed_banknote
    mask[r_x:r_x+transformed_banknote.shape[0], r_y:r_y+transformed_banknote.shape[1]] = 0
    temp = banknote_post_transforms(image=temp_img, mask=mask)
    temp_img = temp['image']
    mask = temp['mask'] #this mask is to be used in the overlay function.
    return temp_img, mask


def overlay(banknote,
            background,
            img_size,
            banknote_size,
            scale,
            banknote_transforms,
            banknote_mask_transforms):
    """
    Overlays a transformed banknote image on a background image.

    Args:
        banknote (numpy.ndarray): The transformed banknote image.
        background (numpy.ndarray): The background image.
        img_size (tuple): The desired size of the output image.
        scale (float): The scale factor to apply to the banknote image.
        banknote_transforms: The transforms to apply to the banknote image.
        banknote_mask_transforms: The transforms to apply to the banknote mask.

    Returns:
        numpy.ndarray: The resulting image with the banknote overlaid on the background.
    """

    background = cv2.resize(background, (img_size[1], img_size[0]))
    banknote, mask = transform_mask_banknote(
        banknote_transforms, banknote_mask_transforms, banknote, img_size, banknote_size, scale)

    resultant_img = (mask * background) + banknote

    return resultant_img

@app.command()
def generate_dataset(dataset_path: str,
                     banknotes_vars_path: str,
                     banknotes_path: str,
                     backgrounds_path: str,
                     images_per_class: int,
                     img_height: int,
                     img_width: int,
                     banknote_height: int,
                     banknote_width: int,
                     upper_scale_bound: int,
                     lower_scale_bound: int):

    banknote_transforms = A.Compose([
                                    A.RandomBrightnessContrast(p=0.3),
                                    A.CoarseDropout(p = .7,max_height=25,max_width=25,max_holes=5),
                                    A.HueSaturationValue(p=0.3),
                                ])

    banknote_mask_transforms = A.Compose([
                                        A.Perspective(p=.35),
                                        A.Rotate(p=.5,limit=20),])
    img_size = (img_height, img_width)
    banknote_size = (banknote_height, banknote_width)

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.exists(banknotes_vars_path):
        os.mkdir(banknotes_vars_path)
    backgrounds_paths = glob.glob(os.path.join(backgrounds_path, '*.jpg'))
    for folder in os.listdir(banknotes_path):
        i = 0
        n = 0
        c = 0
        make_variations(banknotes_path, banknotes_vars_path, folder)
        files = glob.glob(os.path.join(banknotes_vars_path, folder, '*.jpg'))
        class_dir = os.path.join(dataset_path, folder)
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for j in range(images_per_class):
            if (i >= len(backgrounds_paths)):
                i = 0
            if(c >= len(files)):
                c = 0
            banknote = cv2.imread(files[c])
            background = cv2.imread(backgrounds_paths[i])
            scale = np.random.uniform(lower_scale_bound, upper_scale_bound)
            image = overlay(banknote, background, img_size, banknote_size, scale,
                            banknote_transforms, banknote_mask_transforms)
            cv2.imwrite(os.path.join(class_dir, str(n)+'.jpg'), image)
            n += 1
            i += 1
            c += 1




if __name__ == '__main__':
    app()