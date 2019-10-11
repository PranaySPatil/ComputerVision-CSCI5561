import cv2
import numpy as np
# import cupy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from scipy import interpolate

def find_affine_transform(a, b):
    a_transpose = np.transpose(a)
    a_transpose_a_inverse = np.linalg.inv(np.dot(a_transpose, a))
    affine_matrix = np.dot(np.dot(a_transpose_a_inverse, np.transpose(a)), b)
    return affine_matrix

def find_match(img1, img2):
    # To do
    sift = cv2.xfeatures2d.SIFT_create()
    #sift2 = cv2.SIFT()

    matcher = cv2.BFMatcher()
    img1_key_points = sift.detect(img1)
    img2_key_points = sift.detect(img2)

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    nbrs = NearestNeighbors(n_neighbors=2).fit(des1)
    distances, indices = nbrs.kneighbors(des2, 2)
    x1 = []
    x2 = []
    for i in range(len(distances)):
        if distances[i][0] < 0.75 * distances[i][1]:
            x2.append(kp2[i].pt)
            x1.append(kp1[indices[i][0]].pt)

    return np.array(x1), np.array(x2)

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    number_of_random_samples = 3
    max_inliers = 0
    best_transform = None
    best_inlier_x1 = None
    best_inlier_x2 = None
    for iteration in range(ransac_iter):
        a = np.zeros((6,6))
        b = np.zeros((6))
        a_test = np.zeros((2,6))
        b_test = np.zeros((2))
        inlier_x1 = []
        inlier_x2 = []
        inliers_count = 0
        for i in range(number_of_random_samples):
            sample_index = random.randint(0, x1.shape[0]-1)
            a[2*i] = np.array([x1[sample_index][0], x1[sample_index][1], 1, 0, 0, 0])
            a[2*i+1] = np.array([0, 0, 0, x1[sample_index][0], x1[sample_index][1], 1])
            b[2*i] = x2[sample_index][0]
            b[2*i+1] = x2[sample_index][1]
        try:
            affine_transform = find_affine_transform(a, b)
        except:
            continue
        for i in range(x1.shape[0]-1):
            a_test[0] = np.array([x1[i][0], x1[i][1], 1, 0, 0, 0])
            a_test[1] = np.array([0, 0, 0, x1[i][0], x1[i][1], 1])
            b_test[0] = x2[i][0]
            b_test[1] = x2[i][1]
            error_matrix = b_test - np.dot(a_test, affine_transform)#np.insert(x2[i], 2, 1) - np.dot(affine_transform, np.insert(x1[i], 2, 1))
            error = np.hypot(error_matrix[0], error_matrix[1])
            if error < ransac_thr:
                inliers_count += 1
                inlier_x1.append(x1[i])
                inlier_x2.append(x2[i])
        print("Iteration "+str(iteration+1)+", #inliners "+str(inliers_count))
        if max_inliers<inliers_count:
            max_inliers = inliers_count
            best_transform = affine_transform
            best_inlier_x1 = inlier_x1
            best_inlier_x2 = inlier_x2
    print("#####################################\\nMax Inliners")
    print(max_inliers)
    affine_matrix = np.zeros((3,3))
    affine_matrix[0] = best_transform[0:3]
    affine_matrix[1] = best_transform[3:6]
    affine_matrix[2] = np.array([0, 0, 1])
    return affine_matrix, np.array(best_inlier_x1), np.array(best_inlier_x2)

def warp_image(img, A, output_size):
    # To do
    img_warped = np.zeros(output_size)
    img_warped2 = np.zeros(output_size)
    inverse_transform = np.linalg.inv(A)
    for i in range(output_size[0]):
        for j in range(output_size[1]):
            x = np.ones((3,1))
            x[0][0] = j
            x[1][0] = i
            x_prime = np.dot(A, x)
            x2 = math.floor(x_prime[0][0])
            y2 = math.floor(x_prime[1][0])
            if x2>=0 and x2<img.shape[1] and y2>=0 and y2<img.shape[0]:
                img_warped[i][j] = img[y2][x2]
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         x = np.ones((3,1))
    #         x[0][0] = j
    #         x[1][0] = i
    #         x_prime = np.dot(inverse_transform, x)
    #         x2 = math.floor(x_prime[0][0])
    #         y2 = math.floor(x_prime[1][0])
    #         if x2>=0 and x2<output_size[1] and y2>=0 and y2<output_size[0]:
    #             img_warped2[y2][x2] = img[i][j]
    return img_warped

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

def align_image(template, target, A):
    # To do
    template = template/255
    target = target/255
    filter_x, filter_y = get_differential_filter()
    dp = np.zeros((6, 1))
    I_dx = filter_image(template, filter_x)
    I_dy = filter_image(template, filter_y)
    dw_dp = np.zeros((2, 6))
    steepest_distance_images = np.zeros((6, template.shape[0], template.shape[1]))
    dI_x_dw_dp = np.zeros((template.shape[0]*template.shape[1], 6))
    dI_x_dw_dp2 = np.zeros((6, 6))
    hessian = np.zeros((template.shape[0], template.shape[1], 6, 6))
    index = 0
    errors = [[], []]
    for i in range(template.shape[0]):
        for j in range(template.shape[1]):
            dI = np.array([I_dx[i][j], I_dy[i][j]])
            # dw_dp = np.array([[i, 0, j, 0, 1, 0], [0, i, 0,j, 0, 1]])
            dw_dp = np.array([[j, i, 1, 0, 0, 0], [0, 0, 0, j, i, 1]])
            dI_x_dw_dp[index] = np.dot(dI, dw_dp)
            for k in range(dI_x_dw_dp.shape[1]):
                steepest_distance_images[k][i][j] = dI_x_dw_dp[index][k]
            index += 1
    # for i in range(template.shape[0]):
    #     for j in range(template.shape[1]):
    #         dI = np.array([I_dx[i][j], I_dy[i][j]])
    #         # dw_dp = np.array([[i, 0, j, 0, 1, 0], [0, i, 0,j, 0, 1]])
    #         dw_dp = np.array([[i, j, 1, 0, 0, 0], [0, 0, 0, i, j, 1]])
    #         dI_x_dw_dp2 = dI_x_dw_dp2 + np.dot(dI, dw_dp)
    # print(hessian)
    # fig, axes = plt.subplots(1, 6)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # for i in range(len(steepest_distance_images)):
    #     axes[i].axis('off')
    #     axes[i].imshow(steepest_distance_images[i],  cmap='gray')
    # plt.show()
    # fig, axes = plt.subplots(1, 6)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # for i in range(len(steepest_distance_images)):
    #     axes[i].axis('off')
    #     axes[i].imshow(steepest_distance_images[i])
    # plt.show()
    hessian = np.dot(np.transpose(dI_x_dw_dp), dI_x_dw_dp)
    # hessian2 = np.dot(np.transpose(dI_x_dw_dp2), dI_x_dw_dp2)
    hessian_inverse = np.linalg.inv(hessian)
    dp_norm = 1
    iteration = 1
    while dp_norm>0.001 and iteration<1000:
        warped_image = warp_image(target, A, template.shape)
        # plt.imshow(warped_image)
        # plt.show()
        error_image = warped_image - template
        # error_image = cv2.normalize(error_image, error_image, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # xmax, xmin = error_image.max(), error_image.min()
        # error_image = (error_image - xmin)/(xmax - xmin)
        error_value = np.linalg.norm(error_image)
        dp = np.dot(hessian_inverse, np.dot(np.transpose(dI_x_dw_dp), np.reshape(error_image, (template.shape[0]*template.shape[1], 1))))
        dp_norm = np.linalg.norm(dp)
        dp = np.array([dp.flatten()[0:3], dp.flatten()[3:6], [0,0,1]])
        dp[0][0] = dp[0][0] + 1
        dp[1][1] = dp[1][1] + 1
        A = np.dot(A, np.linalg.inv(dp))
        # A = A + np.linalg.inv(dp)
        if iteration%10 == 0:
            print("Iteration "+str(iteration))
            print(error_value)
            print('{:.30f}'.format(dp_norm))
        errors[0].append(error_value)
        errors[1].append(iteration)
        iteration += 1
        # plt.imshow(error_image)
        # plt.show()
    return A, errors


def track_multi_frames(template, target_list):
    A_list = []
    x1, x2 = find_match(template, target_list[0])
    A, x1, x2 = align_image_using_feature(x1, x2, 6, 10000)
    # A = np.load("affine_matrix.npy")
    for target in target_list:
        A_refined, errors = align_image(template, target, A)
        A_list.append(A_refined)
        A = A_refined
        template = warp_image(target, A_refined, template.shape)
    return A_list

def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray')
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors[1], errors[0])
        # plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()

def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
        target_list.append(target)
    # x1, x2 = find_match(template, target_list[0])
    # A, x1, x2 = align_image_using_feature(x1, x2, 6, 10000)
    # template = cv2.normalize(template, template, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # target_list[0] = cv2.normalize(target_list[0], target_list[0], 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # visualize_find_match(template, target_list[0], x1, x2)
    # np.save("affine_matrix.npy", A)
    A = np.load("affine_matrix.npy")
    # img_warped = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # img_warped, img_warped_2 = warp_image(target_list[0], A, template.shape)
    # plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    # plt.imshow(img_warped_2, cmap='gray', vmin=0, vmax=255)
    # plt.axis('off')
    # plt.show()
    # A_refined, errors = align_image(template, target_list[0], A)
    # visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)