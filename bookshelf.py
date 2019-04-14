import glob
import sys

import cv2
import numpy as np
from skimage.util.shape import view_as_blocks

import cnn

input_train_path = "data/dataset1/train/*.JPG"
label_train_path = "data/dataset1/train/masks/*.JPG"
input_test_path = "data/dataset1/test/*.JPG"
label_test_path = "data/dataset1/test/masks/*.JPG"
patch_dim = (32, 32)
patch_channels = 3
# minimum ratio of foreground to background pixels to label as foreground
fg_threshold = 0.7


def image_to_patches(im):
    """takes an RGB image and divides into blocks of size patch_dim,
    reshapes into an array with all patches"""
    patches_matrix = view_as_blocks(im, block_shape=(
        patch_dim[0], patch_dim[1], patch_channels))
    patches_array = np.reshape(patches_matrix, (patches_x * patches_y,
                                                patch_dim[0], patch_dim[1], patch_channels), order='C')
    return patches_array


def mask_to_patches(im):
    """takes an RGB image and divides into blocks of size patch_dim,
    reshapes into an array with all patches"""
    patches_matrix = view_as_blocks(im, block_shape=(
        patch_dim[0], patch_dim[1]))
    patches_array = np.reshape(patches_matrix, (patches_x * patches_y,
                                                patch_dim[0], patch_dim[1]), order='C')
    return patches_array


def patches_to_labels(mask_patches):
    """returns a patches_x by patches_y matrix corresponding to all patch labels"""
    labels = np.zeros(patches_x * patches_y)
    for n in range(0, patches_x * patches_y):
        labels[n] = calc_label(mask_patches[n])
    return labels


def calc_label(mask_patch):
    patch_sum = mask_patch.sum() / (patch_dim[0] * patch_dim[1])
    if patch_sum > fg_threshold:
        return 1
    else:
        return 0


def read_and_format_image(file):
    img = cv2.imread(file)
    img = img.astype('float32') / 255
    return img


def read_and_format_mask(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = im_bw.astype('float32') / 255
    return img


def outlier_free_set(patches, labels):
    outlier_indices = []
    for n in range(0, len(labels)):
        # TODO: update in [calc_label] if this method is to be used
        if labels[n] == -1:
            outlier_indices.append(n)

    pa = np.delete(patches, outlier_indices, axis=0)
    lb = np.delete(labels, outlier_indices, axis=0)
    pa = np.reshape(pa, (len(lb), 32, 32, 3))
    return pa, lb


def main():
    np.set_printoptions(threshold=sys.maxsize)
    img_train = [read_and_format_image(file) for file in glob.glob(input_train_path)]
    lbl_train = [read_and_format_mask(file) for file in glob.glob(label_train_path)]
    img_test = [read_and_format_image(file)for file in glob.glob(input_test_path)]
    lbl_test = [read_and_format_mask(file) for file in glob.glob(label_test_path)]

    cnn.build_model()
    global height, width, channels, patches_x, patches_y

    for n in range(0, len(img_train)):
        height, width, channels = img_train[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        im_patches = image_to_patches(img_train[n])
        height, width = lbl_train[n].shape
        channels = 1
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        labels = patches_to_labels(mask_to_patches(lbl_train[n]))
        # im_patches, labels = outlier_free_set(im_patches, labels)
        print('Training sample:', n+1)
        cnn.train(im_patches, labels)

    for n in range(0, len(img_test)):
        height, width, channels = img_test[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        im_patches = image_to_patches(img_test[n])
        height, width = lbl_test[n].shape
        channels = 1
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        labels = patches_to_labels(mask_to_patches(lbl_test[n]))
        print('Testing sample:', n+1)
        cnn.test(im_patches, labels)


if __name__ == '__main__':
    main()
