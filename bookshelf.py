from skimage.util.shape import view_as_blocks
from matplotlib import pyplot as plt
import numpy as np
import cv2

im = cv2.imread("data/dataset1/train/shelf1.JPG")
im_mask = cv2.imread("data/dataset1/train/masks/shelf1_mask.JPG")
height, width, channels = im.shape
patch_dim = (32, 32)
patch_channels = 1
patches_x = int(width / patch_dim[0])
patches_y = int(height / patch_dim[1])
# minimum ratio of foreground to background pixels to label as foreground
fg_treshold = 0.8


def image_to_patches(im):
    """takes an RGB image and divides into blocks of size patch_dim"""
    return view_as_blocks(im, block_shape=(patch_dim[0], patch_dim[1], patch_channels))


def patches_to_labels(mask_patches):
    """returns a patches_x by patches_y matrix corresponding to all patch labels"""
    labels = np.zeros((patches_x, patches_y))
    for i in range(0, patches_x):
        for j in range(0, patches_y):
            labels[i][j] = calc_label(mask_patches[j, i, 0])
    return labels


def calc_label(mask_patch):
    # maybe we should normalize by 255 earlier?
    sum = mask_patch.sum() / (255 * patch_dim[0] * patch_dim[1])
    if(sum > fg_treshold):
        return 1
    else:
        return 0


def get_pixel_label(x, y, labels):
    """Debug method that takes a pixel position and returns the label"""
    return labels[int(x / patch_dim[0]), int(y / patch_dim[1])]


def main():
    im_patches = image_to_patches(im)
    labels_mat = patches_to_labels(image_to_patches(im_mask))
    # DEBUG:
    # print full matrix
    # np.set_printoptions(threshold=sys.maxsize)
    #print(get_pixel_label(2463, 3031, labels_mat))
    # print image
    # plt.imshow(im)
    # plt.show()
    # print mask
    # plt.imshow(labels_mat, cmap="gray")
    # plt.show()
    # should i map input and labels together or just keep two
    # separate structures?


if __name__ == '__main__':
    main()
