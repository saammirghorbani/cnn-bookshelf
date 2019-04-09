from skimage.util.shape import view_as_blocks
import numpy as np
import cv2
import train as cnn

im = cv2.imread("data/dataset1/train/shelf1.JPG")
im_mask = cv2.imread("data/dataset1/train/masks/shelf1_mask.JPG")
im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

height, width, channels = im.shape
patch_dim = (32, 32)
patch_channels = 3
patches_x = int(width / patch_dim[0])
patches_y = int(height / patch_dim[1])
# minimum ratio of foreground to background pixels to label as foreground
fg_treshold = 0.8


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
    # maybe we should normalize by 255 earlier?
    patch_sum = mask_patch.sum() / (255 * patch_dim[0] * patch_dim[1])
    if patch_sum > fg_treshold:
        return 1
    else:
        return 0


def main():
    im_patches = image_to_patches(im)
    labels = patches_to_labels(mask_to_patches(im_mask))
    cnn.train(im_patches, labels)


if __name__ == '__main__':
    main()
