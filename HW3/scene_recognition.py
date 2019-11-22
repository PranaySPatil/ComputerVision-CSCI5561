import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, size, stride):
    # To do
    n = (img.shape[0]-size+1) * (img.shape[1]-size+1)
    sift = cv2.xfeatures2d.SIFT_create()
    dense_feature = []
    feature_index = 0
    keypoints = []
    kp = [cv2.KeyPoint(x, y, stride) for y in range(0, img.shape[0], stride) 
                                    for x in range(0, img.shape[1], stride)]
    
    keypoints, dense_feature = sift.compute(img, kp)

    return dense_feature


def get_tiny_image(img, output_size):
    # To do
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, output_size)
    img_resized = (img_resized-np.mean(img_resized))/np.std(img_resized)
    
    return img_resized


def predict_knn(feature_train, label_train, feature_test, k):
    # To do
    label_test_pred = []
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(feature_train)
    distances, indices = nbrs.kneighbors(feature_test)
    for neibhbors in indices:
        predictions = [label_train[i] for i in neibhbors]
        label_test_pred.append(max(set(predictions), key=predictions.count))

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    feature_train = []
    feature_test = []
    confusion = np.zeros((15, 15))
    accuracy = 0
    img_size = 10
    k = 9

    for img_path in img_train_list:
        tiny_image = get_tiny_image(img_path, (img_size, img_size))
        feature_train.append(tiny_image.flatten())

    for img_path in img_test_list:
        tiny_image = get_tiny_image(img_path, (img_size, img_size))
        feature_test.append(tiny_image.flatten())

    label_test_pred = predict_knn(feature_train, label_train_list, feature_test, k)

    for i in range(len(label_test_pred)):
        predicted_index = label_classes.index(label_test_pred[i])
        actual_index = label_classes.index(label_test_list[i])
        confusion[actual_index, predicted_index] += 1
        if predicted_index == actual_index:
            accuracy += 1
    
    accuracy = accuracy/len(label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # To do
    print("calculating visual dictionary")
    kmeans = KMeans(n_clusters=dic_size)
    vocab = kmeans.fit(dense_feature_list)

    return vocab.cluster_centers_


def compute_bow(feature, vocab):
    # To do
    bow_feature = np.zeros((vocab.shape[0], 1))
    nbrs = NearestNeighbors(n_neighbors=1).fit(vocab)
    distances, indices = nbrs.kneighbors(feature)
    for i in indices:
        bow_feature[i] += 1

    return bow_feature/np.linalg.norm(bow_feature)


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    dense_feature_list = []
    feature_list = None
    dic_size = 200
    img = None
    print("calculating dense features list")
    i = 1

    for img_path in img_train_list:
        print(str(i) + " out of "+str(len(img_train_list)))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature_list = compute_dsift(img, 20, 10)
        dense_feature_list.extend(feature_list)
        i += 1

    visual_dict = build_visual_dictionary(dense_feature_list, dic_size)
    np.save("visualDict_knn.npy", visual_dict)
    print("saved dictionary")

    # load previously built dictionary
    # visual_dict = np.load("visualDict_knn.npy", allow_pickle=True)

    x_train = []
    x_test = []

    for img_path in img_train_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature_list = compute_dsift(img, 20, 10)
        x_train.append(compute_bow(feature_list, visual_dict))

    for img_path in img_test_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature_list = compute_dsift(img, 20, 10)
        x_test.append(compute_bow(feature_list, visual_dict))
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape((1500, dic_size))
    x_test = x_test.reshape((1500, dic_size))
    label_test_pred = []
    accuracy = 0
    confusion = np.zeros((15, 15))
    k = 8
    nbrs = NearestNeighbors(n_neighbors=k).fit(x_train)
    distances, indices = nbrs.kneighbors(x_test)

    for neibhbors in indices:
        predictions = [label_train_list[i] for i in neibhbors]
        label_test_pred.append(max(set(predictions), key=predictions.count))

    for i in range(len(label_test_pred)):
        actual_label_index = label_classes.index(label_test_list[i])
        predicted_label_index = label_classes.index(label_test_pred[i])
        confusion[actual_label_index][predicted_label_index] += 1
        if label_test_pred[i] == label_test_list[i]:
            accuracy += 1
    
    confusion = confusion / np.linalg.norm(confusion)
    accuracy = accuracy/len(label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test):
    # To do
    clf = LinearSVC()
    models = {}
    n_classes = 15
    predictions = np.ndarray((feature_test.shape[0], n_classes), dtype=float)

    for i in range(0, n_classes):
        labels_for_model = []
        for label in label_train:
            if label == i:
                labels_for_model.append(1)
            else:
                labels_for_model.append(0)
        models[i] = clf.fit(feature_train, labels_for_model)
        predictions[:,i] = models[i].decision_function(feature_test)
    
    label_test_pred = []

    for i in range(len(feature_test)):
        predicted_label = np.where(predictions[i] == np.amax(predictions[i]))
        label_test_pred.append(predicted_label[0][0])

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # To do
    # Reading the dict saved in classify_knn_bow, Uncomment till line 232 to build the dictionary again
    # dense_feature_list = []
    # feature_list = None
    # dic_size = 200
    # img = None
    # print("calculating dense features list")
    # i = 1
    # for img_path in img_train_list:
    #     print(str(i) + " out of "+str(len(img_train_list)))
    #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     feature_list = compute_dsift(img, 20, 10)
    #     dense_feature_list.extend(feature_list)
    #     i += 1

    # visual_dict = build_visual_dictionary(dense_feature_list, dic_size)

    visual_dict = np.load("visualDict_knn.npy", allow_pickle=True)
    dic_size = 200
    x_train = []
    x_test = []

    for img_path in img_train_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature_list = compute_dsift(img, 20, 10)
        x_train.append(compute_bow(feature_list, visual_dict))

    for img_path in img_test_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        feature_list = compute_dsift(img, 20, 10)
        x_test.append(compute_bow(feature_list, visual_dict))
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train.reshape((1500, dic_size))
    x_test = x_test.reshape((1500, dic_size))
    
    y_train = [label_classes.index(label) for label in label_train_list]
    label_test_pred = predict_svm(x_train, y_train, x_test)
    accuracy = 0
    confusion = np.zeros((15, 15))

    for i in range(len(label_test_pred)):
        actual_label_index = label_classes.index(label_test_list[i])
        predicted_label_index = label_test_pred[i]
        confusion[actual_label_index][predicted_label_index] += 1
        if actual_label_index == predicted_label_index:
            accuracy += 1

    accuracy = accuracy/len(label_test_pred)
    visualize_confusion_matrix(confusion, accuracy, label_classes)

    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
    
    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)




