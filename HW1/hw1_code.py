import cv2
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.000000001

def get_differential_filter():
    filter_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    filter_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)
    return filter_x, filter_y

def filter_image(im, filter):
    number_of_extra_rows = filter.shape[0]//2
    number_of_extra_cols = filter.shape[1]//2
    im_filtered = np.zeros((im.shape[0], im.shape[1]), float)
    im_to_process = np.zeros((im.shape[0]+(2*number_of_extra_rows), im.shape[1]+(2*number_of_extra_cols)))
    im_to_process[number_of_extra_rows:-1*number_of_extra_rows, number_of_extra_cols:-1*number_of_extra_cols] = im
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            im_filtered[i,j] = sum(sum(filter*im_to_process[i:i+filter.shape[0], j:j+filter.shape[1]]))
    return im_filtered

def get_gradient(im_dx, im_dy):
    grad_mag = np.zeros((im_dx.shape[0], im_dx.shape[1]), float)
    grad_angle = np.zeros((im_dx.shape[0], im_dx.shape[1]), float)
    grad_mag =  np.sqrt(np.square(im_dx) + np.square(im_dy))
    grad_angle = np.arctan2(im_dy, im_dx) * (180 / np.pi)
    return grad_mag, grad_angle

def get_bin_index(grad_angle):
    if grad_angle<0:
        grad_angle += 180

    if grad_angle<15 or grad_angle>=165:
        return 0
    elif grad_angle<45:
        return 1
    elif grad_angle<75:
        return 2
    elif grad_angle<105:
        return 3
    elif grad_angle<135:
        return 4
    elif grad_angle<165:
        return 5

def build_histogram(grad_mag, grad_angle, cell_size):
    row_cells = grad_mag.shape[0]//cell_size
    col_cells = grad_mag.shape[1]//cell_size
    bin_length = 6
    ori_histo = np.zeros((row_cells, col_cells, bin_length))
    for i in range(grad_mag.shape[0]):
        for j in range(grad_mag.shape[1]):
            cell_row_index = i//cell_size
            cell_col_index = j//cell_size
            bin_index = get_bin_index(grad_angle[i, j])
            ori_histo[cell_row_index, cell_col_index, bin_index] += grad_mag[i, j]
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
            ori_histo_normalized[i, j, :] = temp_histo / (np.linalg.norm(temp_histo)+epsilon)
    block_desciptor = np.reshape(ori_histo_normalized, (normalized_rows*normalized_cols*normalized_bins))
    return block_desciptor

def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0
    diff_x, diif_y = get_differential_filter()
    dI_dx = filter_image(im, diff_x)
    dI_dy = filter_image(im, diif_y)
    gradient_mag, gradient_angle = get_gradient(dI_dx, dI_dy)
    ori_histo = build_histogram(gradient_mag, gradient_angle, 8)
    hog = get_block_descriptor(ori_histo, 2)

    # visualize to verify
    visualize_hog(im, hog, 8, 2)

    return hog

def visualize_histo(im, ori_histo, cell_size):
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

# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


if __name__=='__main__':
    im = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
    hog = extract_hog(im)