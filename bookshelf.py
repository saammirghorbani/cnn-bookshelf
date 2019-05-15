import glob
import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt
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
# minimum ratio of foreground to background pixels to label as background
bg_threshold = 0.3


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


def patches_to_labels_train(patches):
    """returns a patches_x * patches_y array corresponding to all patch labels except unreliable ones"""
    fg_count = 0
    bg_count = 0
    labels = np.zeros(patches_x * patches_y)
    for n in range(0, patches_x * patches_y):
        fg_pixel_ratio = patches[n].sum() / (patch_dim[0] * patch_dim[1])
        if fg_pixel_ratio > fg_threshold:
            labels[n] = 1
            fg_count += 1
        elif fg_pixel_ratio < bg_threshold:
            labels[n] = 0
            bg_count += 1
        else:
            labels[n] = -1
    return labels, fg_count, bg_count


def patches_to_labels_test(patches):
    """returns a patches_x * patches_y array corresponding to all patch labels"""
    labels = np.zeros(patches_x * patches_y)
    for n in range(0, patches_x * patches_y):
        fg_pixel_ratio = patches[n].sum() / (patch_dim[0] * patch_dim[1])
        if fg_pixel_ratio > fg_threshold:
            labels[n] = 1
        else:
            labels[n] = 0
    return labels


def read_and_format_image(file):
    img = cv2.imread(file)
    img = img.astype('float32') / 255
    return img


def read_and_format_mask(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = im_bw.astype('float32') / 255
    return img


def prune_unreliable_samples(patches, labels):
    unrel_indices = []
    for n in range(0, len(labels)):
        if labels[n] == -1:
            unrel_indices.append(n)
    pa = np.delete(patches, unrel_indices, axis=0)
    lb = np.delete(labels, unrel_indices, axis=0)
    pa = np.reshape(pa, (len(lb), patch_dim[0], patch_dim[1], patch_channels))
    return pa, lb


def balance_labels(patches, labels, fg_count, bg_count):
    index = 0
    removable_indices = []
    while bg_count > fg_count:
        if labels[index] == 0:
            removable_indices.append(index)
            bg_count -= 1
        index += 1
    pa = np.delete(patches, removable_indices, axis=0)
    lb = np.delete(labels, removable_indices, axis=0)
    pa = np.reshape(pa, (len(lb), patch_dim[0], patch_dim[1], patch_channels))
    return pa, lb


def plot_results(image, predictions, labels, sample_nr):
    output = np.reshape(predictions, [patches_y, patches_x])
    lbl = np.reshape(labels, [patches_y, patches_x])
    fig, axs = plt.subplots(nrows=1, ncols=3)

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Input')

    axs[1].imshow(output, cmap='gray')
    axs[1].set_title('Output')

    axs[2].imshow(lbl, cmap='gray')
    axs[2].set_title('Ground Truth')

    fig.suptitle('Testing sample ' + str(sample_nr), fontsize=16)


def train_and_save():
    global height, width, channels, patches_x, patches_y
    img_train = [read_and_format_image(file) for file in glob.glob(input_train_path)]
    lbl_train = [read_and_format_mask(file) for file in glob.glob(label_train_path)]
    cnn.build_model()
    for n in range(0, len(img_train)):
        height, width, channels = img_train[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        im_patches = image_to_patches(img_train[n])
        height, width = lbl_train[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        labels, fg_count, bg_count = patches_to_labels_train(mask_to_patches(lbl_train[n]))
        im_patches, labels = prune_unreliable_samples(im_patches, labels)
        im_patches, labels = balance_labels(im_patches, labels, fg_count, bg_count)
        print('Training sample:', n+1)
        cnn.train(im_patches, labels)
    cnn.save_model()


def test_model(model):
    global height, width, channels, patches_x, patches_y
    img_test = [read_and_format_image(file)for file in glob.glob(input_test_path)]
    lbl_test = [read_and_format_mask(file) for file in glob.glob(label_test_path)]
    for n in range(0, len(img_test)):
        height, width, channels = img_test[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        im_patches = image_to_patches(img_test[n])
        height, width = lbl_test[n].shape
        patches_x = int(width / patch_dim[0])
        patches_y = int(height / patch_dim[1])
        labels = patches_to_labels_test(mask_to_patches(lbl_test[n]))
        print('Testing sample:', n + 1)
        predictions = cnn.test(im_patches, labels, model)
        plot_results(img_test[n], predictions, labels, n + 1)
    plt.show()


def main():
    np.set_printoptions(threshold=sys.maxsize)
    train_and_save()  # comment out if you want to use previously saved model
    model = cnn.load_model()
    test_model(model)


if __name__ == '__main__':
    main()
