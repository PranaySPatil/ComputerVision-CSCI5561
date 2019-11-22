import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    # TO DO
    mini_batch_x = []
    mini_batch_y = []
    k = 0

    for i in range (0, im_train.shape[1], batch_size):
        batch_x = im_train[:, i: min(im_train.shape[1], i + batch_size)]
        batch_label = np.zeros((10, batch_x.shape[1]))
        for i in range(i, min(im_train.shape[1], i + batch_size)):
            batch_label[label_train[0][i], i % batch_size] = 1
        
        mini_batch_x.append(batch_x)
        mini_batch_y.append(batch_label)
        k += 1

    return np.array(mini_batch_x), np.array(mini_batch_y)

def fc(x, w, b):
    # TO DO
    y = np.transpose(np.dot(np.transpose(x), np.transpose(w)))
    y = y + b
    # y = y/np.linalg.norm(y)

    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    dl_dx = np.dot(np.transpose(dl_dy), w)
    dl_dw = np.dot(dl_dy.reshape(dl_dy.shape[0], 1), x.reshape(1, x.shape[0]))
    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    l = (y - y_tilde) ** 2
    dl_dy = 2 * (y - y_tilde) * -1
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
    l = 0
    e_power_x = np.exp(x-max(x))
    # e_power_x = np.exp(x)
    soft_max_x = e_power_x/sum(e_power_x)
    # for i in range(10):
    #     l += (y[i] * np.log(soft_max_x[i]))
    try:
        l = y * np.log(soft_max_x + np.finfo(float).eps)
    except:
        pass
    l *= -1
    dl_dy = soft_max_x - y

    return l, dl_dy

def relu(x):
    # TO DO
    # relu_lambda = lambda t: t if t>0 else t*0.01
    relu_lambda = lambda t: max(0, t)
    v_func = np.vectorize(relu_lambda)
    y = v_func(x)

    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = []
    x_flat = x.flatten()
    dl_dy_flatten = dl_dy.flatten()
    for i in range(x_flat.shape[0]):
        if x_flat[i] > 0:
            dl_dx.append(dl_dy_flatten[i])
        else:
            # dl_dx.append(0.01)
            dl_dx.append(0)

    return np.array(dl_dx).reshape((x_flat.shape[0], 1))

def conv(x, w_conv, b_conv):
    # TO DO
    stride = 1
    im_h, im_w, im_c = x.shape
    w_h, w_w, w_c1, w_c2 = w_conv.shape
    y = np.zeros((im_h, im_w, w_c2))
    x_pad = np.pad(x,((1, 1),(1, 1), (0, 0)), 'constant')

    for c in range(w_c2):
        conv_filter = w_conv[ :, :, :, c].reshape(w_h, w_w)
        for i in range(im_h):
            for j in range(im_w):
                im = x_pad[i:i+w_h, j:j+w_w, :].reshape(w_h, w_w)
                y[i, j, c] = sum(sum(conv_filter * im)) + b_conv[c]

    # x_pad = np.pad(x,((1, 1),(1, 1), (0, 0)),'constant')
    # x_col = im2col(x_pad, w_h, w_w, stride)
    # filter_col = np.reshape(w_conv, (w_c2, -1))
    # y = x_col.dot(filter_col.T) + np.transpose(b_conv)
    # y = col2im(y, im_h, im_w, 1)
    # return y/np.linalg.norm(y)
    return y
    

def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    x_pad = np.pad(x,((1, 1),(1, 1), (0, 0)), 'constant')
    dl_dw = np.zeros(w_conv.shape)
    dl_db = np.zeros(b_conv.shape)
    im_h, im_w, im_c = x.shape
    w_h, w_w, w_c1, w_c2 = w_conv.shape
    stride = 1

    for c in range(w_c2):
        for i in range(im_h):
            for j in range(im_w):
                im = x_pad[i:i+w_h, j:j+w_w, :].reshape(w_h, w_w)
                dl_dw[:, :, :, c] += np.expand_dims(dl_dy[i, j, c] * im, axis=2)
                dl_db[c] += dl_dy[i, j, c]

    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    y = np.zeros((x.shape[0]//2, x.shape[1]//2, x.shape[2]))
    for y_c in range(x.shape[2]):
        y_x = 0
        for r in range(0, x.shape[0], 2):
            y_y = 0
            for c in range(0, x.shape[1], 2):
                y[y_x, y_y, y_c] = np.max(x[r:r+2,  c:c+2, y_c])  
                y_y += 1
            y_x += 1

    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    dl_dx = np.zeros(x.shape)

    for c in range(dl_dy.shape[2]):
        for i in range(dl_dy.shape[0]):
            for j in range(dl_dy.shape[1]):
                x_ = x[i*2:i*2+2, j*2:j*2+2, c]
                mask = (x_ == np.max(x_))
                dl_dx[i*2:i*2+2, j*2:j*2+2, c] = mask*dl_dy[i, j, c]

                # if x[i, j, c] == y[i//dl_dx.shape[0], j//dl_dx.shape[1], c]:
                #     dl_dx[i, j, c] = dl_dy[i//dl_dx.shape[0], j//dl_dx.shape[1], c]

    return dl_dx


def flattening(x):
    # TO DO
    y = x.flatten()
    return y.reshape((y.shape[0], 1))


def flattening_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy.reshape(x.shape)
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    # TO DO
    gamma = 0.8
    lambda_decay = 0.9
    mean = 0
    sigma = 5 
    w = np.random.normal(mean, sigma, (10, 196))
    b = np.random.normal(mean, sigma, (10, 1))
    k = 0
    batch_count = 1
    losses = []

    for iteration in range(1, 10000):
        print("Epoch #"+str(iteration))
        print("Batch count #"+str(k))
        if iteration % 1000 == 0:
            gamma = lambda_decay * gamma
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))
        batch_size = mini_batch_x.shape[0]
        L = 0

        for i in range(mini_batch_x[k].shape[1]):
            x = mini_batch_x[k,:,i]
            y = mini_batch_y[k,:,i]
            x = x.reshape((x.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_euclidean(y_tilde, y)
            L += np.linalg.norm(l)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dL_dw += dl_dw
            dL_db += dl_db
        
        losses.append(L/mini_batch_x[k].shape[1])
        k += 1
        k %= batch_size
        batch_count += 1
        w = w - ((gamma/batch_size) * dL_dw)
        b = b - ((gamma/batch_size) * dL_db) 

    plt.plot(list(range(1, 10000)), losses)
    plt.show()
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    gamma = 12
    lambda_decay = 1
    mean = 0
    sigma = 5 
    w = np.random.normal(mean, sigma, (10, 196))
    b = np.random.normal(mean, sigma, (10, 1))
    k = 0
    batch_count = 1
    losses = []

    for iteration in range(1, 20000):
        print("Epoch #"+str(iteration))
        print("Batch count #"+str(k))
        if iteration % 5000 == 0:
            gamma = lambda_decay * gamma
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))
        batch_size = mini_batch_x.shape[0]
        L = 0

        for i in range(mini_batch_x[k].shape[1]):
            x = mini_batch_x[k,:,i]
            y = mini_batch_y[k,:,i]
            x = x.reshape((x.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
            y_tilde = fc(x, w, b)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L += np.linalg.norm(l)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, x, w, b, y_tilde)
            dL_dw += dl_dw
            dL_db += dl_db
        
        k += 1
        k %= batch_size
        losses.append(L/mini_batch_x[k].shape[1])
        batch_count += 1
        w = w - ((gamma/batch_size) * dL_dw)
        b = b - ((gamma/batch_size) * dL_db) 

    plt.plot(list(range(1, 20000)), losses)
    plt.show()
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    gamma = 0.5
    lambda_decay = 0.4
    mean = 0
    sigma = 5
    w1 = np.random.normal(mean, sigma, (30, 196))
    w1 = w1/np.linalg.norm(w1)
    b1 = np.random.normal(mean, sigma, (30, 1))
    b1 = b1/np.linalg.norm(b1)
    w2 = np.random.normal(mean, sigma, (10, 30))
    w2 = w2/np.linalg.norm(w2)
    b2 = np.random.normal(mean, sigma, (10, 1))
    b2 = b2/np.linalg.norm(b2)
    k = 0
    batch_count = 1
    losses = []

    for iteration in range(1, 30000):
        print("Epoch #"+str(iteration))
        print("Batch count #"+str(batch_count))
        if iteration % 5000 == 0:
            gamma = lambda_decay * gamma
        dL_dw1 = np.zeros((30, 196))
        dL_db1 = np.zeros((30, 1))
        dL_dw2 = np.zeros((10, 30))
        dL_db2 = np.zeros((10, 1))
        batch_size = mini_batch_x.shape[0]
        L = 0

        for i in range(mini_batch_x[k].shape[1]):
            x = mini_batch_x[k,:,i]
            y = mini_batch_y[k,:,i]
            x = x.reshape((x.shape[0], 1))
            y = y.reshape((y.shape[0], 1))
            x2 = fc(x, w1, b1)
            x3 = relu(x2)
            y_tilde = fc(x3, w2, b2)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L += np.linalg.norm(l)
            dl_dx3, dl_dw2, dl_db2 = fc_backward(dl_dy, x3, w2, b2, y_tilde)
            dl_dx2 = relu_backward(dl_dx3, x2, x3)
            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dx2, x, w1, b1, x2)
            # dl_dx2 = relu_backward(dl_dy*dl_dx3, x2, x3)
            # dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy*dl_dx3*dl_dx2, x, w1, b1, x2)
            dL_dw1 += dl_dw1
            dL_db1 += dl_db1
            dL_dw2 += dl_dw2
            dL_db2 += dl_db2
        
        losses.append(L/mini_batch_x[k].shape[1])
        k += 1
        k %= batch_size
        batch_count += 1
        w1 = w1 - ((gamma/batch_size) * dL_dw1)
        b1 = b1 - ((gamma/batch_size) * dL_db1)
        w2 = w2 - ((gamma/batch_size) * dL_dw2)
        b2 = b2 - ((gamma/batch_size) * dL_db2)

    plt.plot(list(range(1, 30000)), losses)
    plt.show()
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    mean = 0
    sigma = 1
    w_conv = np.random.normal(mean, sigma, (3, 3, 1, 3))
    # w_conv = w_conv/np.linalg.norm(w_conv)
    b_conv = np.random.normal(mean, sigma, (3, 1))
    # b_conv = b_conv/np.linalg.norm(b_conv)
    w_fc = np.random.normal(mean, sigma, (10, 147))
    # w_fc = w_fc/np.linalg.norm(w_fc)
    b_fc = np.random.normal(mean, sigma, (10, 1))
    # b_fc = b_fc/np.linalg.norm(b_fc)
    gamma = 0.5
    m_L = 10
    losses = []
    lambda_decay = 0.9

    k = 0
    batch_count = 1

    for iteration in range(1, 25000):
        print("Epoch #"+str(iteration))
        # print("Batch count #"+str(batch_count))
        if iteration % 5000 == 0:
            gamma = lambda_decay * gamma
        dL_dw_conv = np.zeros((3, 3, 1, 3))
        dL_db_conv = np.zeros((3, 1))
        dL_dw_fc = np.zeros((10, 147))
        dL_db_fc = np.zeros((10, 1))
        L = 0
        batch_size = mini_batch_x.shape[0]

        for i in range(mini_batch_x[k].shape[1]):
            x = mini_batch_x[k,:,i]
            y = mini_batch_y[k,:,i]
            x = x.reshape((14, 14, 1), order='F')
            # x = x/np.linalg.norm(x)
            y = y.reshape((y.shape[0], 1))
            # y = y/np.linalg.norm(y)
            x1 = conv(x, w_conv, b_conv)
            x2 = relu(x1)
            x3 = pool2x2(x2)
            x4 = flattening(x3)
            y_tilde = fc(x4, w_fc, b_fc)
            l, dl_dy = loss_cross_entropy_softmax(y_tilde, y)
            L += np.linalg.norm(l)
            dl_dx4, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, x4, w_fc, b_fc, y_tilde)
            dl_dx3 = flattening_backward(dl_dx4, x3, x4)
            dl_dx2 = pool2x2_backward(dl_dx3, x2, x3)
            dl_dx = relu_backward(dl_dx2, x1, x2)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dx2, x, w_conv, b_conv, x1)

            dL_dw_conv += dl_dw_conv
            dL_db_conv += dl_db_conv
            dL_dw_fc += dl_dw_fc
            dL_db_fc += dl_db_fc

        print("Loss " + str(L/mini_batch_x[k].shape[1]))
        losses.append(L/mini_batch_x[k].shape[1])
        k += 1
        k %= batch_size
        batch_count += 1
        w_conv = w_conv - ((gamma/batch_size) * dL_dw_conv)
        b_conv = b_conv - ((gamma/batch_size) * dL_db_conv)
        w_fc = w_fc - ((gamma/batch_size) * dL_dw_fc)
        b_fc = b_fc - ((gamma/batch_size) * dL_db_fc)
        # if m_L >  L/mini_batch_x[k].shape[1]:
        #     m_L = min(m_L, L/mini_batch_x[k].shape[1])
        #     w_conv2, b_conv2, w_fc2, b_fc2 = w_conv, b_conv, w_fc, b_fc 

    plt.plot(list(range(1, 25000)), losses)
    plt.show()
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    # main.main_slp_linear()
    # main.main_slp()
    # main.main_mlp()
    main.main_cnn()



