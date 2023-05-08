import numpy as np
import matplotlib.pyplot as plt
# import ipympl
import imageio.v3 as iio
import skimage
from skimage.filters import gaussian, median, difference_of_gaussians, laplace
from skimage.feature import canny
from skimage import exposure

# fig = plt.figure()
# fig.add_subplot(2, 2, 1)
# plt.imshow(image_original)
# # print(image)
# # image_original.shape
# gray_image = skimage.color.rgb2gray(image_original)
# fig.add_subplot(2, 2, 2)
# plt.imshow(gray_image)
# filtered_image = median(gray_image)
# fig.add_subplot(2, 2, 3)
# plt.imshow(filtered_image)
# image_canny = canny(filtered_image, 1.5)
# image = image_canny.copy()
# fig.add_subplot(2, 2, 4)
# plt.imshow(image, cmap='gray')
# plt.show(block=True)

def preprocess_image(original_image, verbose=True):
    gray_image = skimage.color.rgb2gray(original_image)
    # fig, ax = plt.subplots()
    # plt.axis('off')
    # plt.imshow(gray_image, cmap='gray')
    filtered_image = difference_of_gaussians(gray_image, 1, 12)
    if verbose:
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.imshow(filtered_image, cmap='gray')
    image_edges = laplace(filtered_image, 8)
    image = image_edges.copy()
    if verbose:
        fig, ax = plt.subplots()
        plt.axis('off')
        plt.imshow(image, cmap='gray')
        plt.autoscale(tight=True)
        plt.show()

    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    img_eq = exposure.equalize_hist(image)
    img_eq[img_eq>0.25] = 1
    img_eq[img_eq<=0.25] = 0
    img_eq = 1 - img_eq
    if verbose:
        fig, ax = plt.subplots()
        plt.imshow(img_eq, cmap='gray')
    image = img_eq
    print(image.shape)
    return image

def remove_islands(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size), np.uint8)
    kernel[:,0] = 1
    kernel[0,:] = 1
    kernel[:,-1] = 1
    kernel[-1,:] = 1
    # print(kernel)


    for i in range(0, image.shape[0]-kernel_size):
        for j in range(0, image.shape[1]-kernel_size):
            border = image[i:i+kernel_size, j:j+kernel_size]
            if((border*kernel).sum() == 0):
                image[i:i+kernel_size, j:j+kernel_size] = 0


def get_image(path, verbose=False):
    image_original = iio.imread(uri=path)
    image = preprocess_image(image_original, verbose=verbose)
    remove_islands(image)
    remove_islands(image, 10)
    # fig, ax = plt.subplots()
    # plt.imshow(image)
    return image
