import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

epsilon = 0.000000001

def get_differential_filter():
    # To do
    filter_x = (1/3)*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_y = (1/3)*np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # To do
    number_of_extra_rows = filter.shape[0]//2
    number_of_extra_cols = filter.shape[1]//2
    im_filtered = np.zeros((im.shape[0], im.shape[1]))
    im_to_process = np.zeros((im.shape[0]+(2*number_of_extra_rows), im.shape[1]+(2*number_of_extra_cols)))
    im_to_process[number_of_extra_rows:-1*number_of_extra_rows, number_of_extra_cols:-1*number_of_extra_cols] = im
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_filtered[i,j] = np.sum(filter*im_to_process[i:i+filter.shape[0], j:j+filter.shape[1]])
    return im_filtered

def normalizeAngle(a):
    if a<0:
        return a+np.pi
    else:
        return a

def get_gradient(im_dx, im_dy):
    # To do
    grad_mag = np.zeros((im_dx.shape[0], im_dx.shape[1]))
    grad_angle = np.zeros((im_dx.shape[0], im_dx.shape[1]))
    grad_mag =  np.sqrt(np.square(im_dx) + np.square(im_dy))
    grad_angle = np.arctan2(im_dy, im_dx)
    vfunc = np.vectorize(normalizeAngle)
    grad_angle = vfunc(grad_angle)
    print(grad_angle.min())
    print(grad_angle.max())
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    row_cells = grad_mag.shape[0]//cell_size
    col_cells = grad_mag.shape[1]//cell_size
    bin_length = 6
    ori_histo = np.zeros((row_cells, col_cells, bin_length))
    for i in range(grad_mag.shape[0]):
        for j in range(grad_mag.shape[1]):
            cell_row_index = i//cell_size
            cell_col_index = j//cell_size
            bin_index = math.ceil(grad_angle[i, j]/(np.pi/bin_length)) - 1
            ori_histo[cell_row_index, cell_col_index, bin_index] += 1
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    normalized_rows = ori_histo.shape[0]-(block_size-1)
    normalized_cols = ori_histo.shape[1]-(block_size-1)
    normalized_bins = ori_histo.shape[2]*(block_size**2)
    ori_histo_normalized = np.zeros((normalized_rows, normalized_cols, normalized_bins))
    for i in range(normalized_rows):
        for j in range(normalized_cols):
            temp_histo = ori_histo[i:i+block_size, j:j+block_size, :]
            temp_histo = np.reshape(temp_histo, (normalized_bins))
            ori_histo_normalized[i, j, :] = temp_histo / np.sqrt(temp_histo.sum()**2 + epsilon)
    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    # To do
    filter_x, filter_y = get_differential_filter()
    print("Shapes: ")
    print("image "+str(im.shape))
    im = filter_image(im, np.array([[0.05472157, 0.11098164, 0.05472157], [0.11098164, 0.22508352, 0.11098164], [0.05472157, 0.11098164, 0.05472157]]))
    gradient_x = filter_image(im, filter_x)
    gradient_y = filter_image(im, filter_y)
    print("gradients "+str(gradient_x.shape))
    gradient_mag, gradient_angle = get_gradient(gradient_x, gradient_y)
    print("gradient mag "+str(gradient_mag.shape))
    print("gradient angle "+str(gradient_angle.shape))
    histo = build_histogram(gradient_mag, gradient_angle, 8)
    print("original histogram "+str(histo.shape))
    hog = get_block_descriptor(histo, 2)
    print("normalized histo "+str(hog.shape))
    visualize_hog(im, histo, 8)
    return hog


def visualize_hog(im, ori_histo, cell_size):
    norm_constant = 1e-3
    num_bins = ori_histo.shape[2]
    height, width = im.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2:width:cell_size], np.r_[cell_size/2:height:cell_size])

    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2)  # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len  # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    # im = cv2.imread('cameraman.tif', 0)
    im = cv2.imread('einstein.jpg', 0)
    # im = filter_image(im, np.array([[0.05472157, 0.11098164, 0.05472157], [0.11098164, 0.22508352, 0.11098164], [0.05472157, 0.11098164, 0.05472157]]))
    dx, dy = get_differential_filter()
    im_dx = filter_image(im, dx)
    im_dy = filter_image(im, dy)
    gradient_mag, gradient_angle = get_gradient(im_dx, im_dy)
    # hog = extract_hog(im)
    plt.imshow(gradient_angle)
    plt.show()
    # print(filter_image(np.ones((256, 256)), filter_x))
    # print(filter_image(np.ones((256, 256)), filter_y))


