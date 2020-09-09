import os
import csv
import numpy as np
import scipy.io as sio
import bct
import time

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from sklearn.utils import shuffle

from nilearn import connectome
from scipy import stats
#from sklearn.feature_selection import SelectFromModel
#from sklearn.linear_model import LassoCV
from sklearn import linear_model
import pickle
import tensorflow as tf
import os
import shutil

import json

from tempfile import mkdtemp

from tqdm import tqdm


import numpy as np

from joblib import Parallel, delayed

import depmeas
import numpy
import random
# Reading and computing the input data

# Selected pipeline
#pipeline = 'cpac'

manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)

# Input data variables
root_folder = '../../'
data_folder = os.path.join(root_folder, 'Data/Yeo_ROItimeseries/Yeo_signal_17')
# phenotype = os.path.join(root_folder, 'Data/Subject_thresholding_list_181020ver.csv')
phenotype = os.path.join(root_folder, 'Data/Subject_list_181024ver.csv')
# phenotype = os.path.join(root_folder, 'Data/Subject_list_181024ver_Rearranged.csv')
# phenotype = os.path.join(root_folder, 'Data/Subject_thresholding_list_181024ver.csv')

def load_ec_GCA(subject_list, data_fld):
    all_EC = []
    for subject in subject_list:
        # for i in os.listdir(data_fld):
        #     print(os.path.isfile(os.path.join(data_fld, i)))
        #     print(subject)
        #     print(i)
        # print('hi')
        flname = [i for i in os.listdir(data_fld) if
                  os.path.isfile(os.path.join(data_fld, i)) and subject in i]
        fl = os.path.join(data_fld, flname[0])
        EC = np.load(fl, allow_pickle=True)
        # timeseries = sio.loadmat(fl)['ROI']
        # for i in range(len(subject_list)):
        #     all_networks.append(fc[:,:,i])
        all_EC.append(EC)
    vec_networks = [np.reshape(mat, [1,-1]) for mat in all_EC]
    matrix = np.vstack(vec_networks)
    return matrix


# def calculate_performance(x_data, y_data, train_ind, val_ind, test_ind):


def E2Eblock(input, output_dims, num_ROIs):
    hidden1 = tf.layers.conv2d(input, output_dims, [1, num_ROIs])
    hidden2 = tf.layers.conv2d(input, output_dims, [num_ROIs, 1])
    return tf.concat([hidden1] * num_ROIs, 2) + tf.concat([hidden2] * num_ROIs, 1)

def BrainNetCNN(input, num_class=2, num_ROIs=114, keep_prob=0.5, reuse=None):
    with tf.variable_scope("BrainNetCNN") as scope:
        if reuse:
            scope.reuse_variables()
            keep_prob = 1
        # E2E block
        hidden = E2Eblock(input=input, output_dims=10, num_ROIs=num_ROIs)
        hidden = tf.nn.leaky_relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        hidden = E2Eblock(input=hidden, output_dims=10, num_ROIs=num_ROIs)
        hidden = tf.nn.leaky_relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        # E2N block
        hidden = tf.layers.conv2d(hidden, 20, [1, num_ROIs])
        hidden = tf.nn.leaky_relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        # N2G block
        hidden = tf.layers.conv2d(hidden, 40, [num_ROIs, 1])
        hidden = tf.nn.leaky_relu(hidden)
        hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        # Dense layer
        hidden = tf.layers.flatten(hidden)
        decision = tf.layers.dense(hidden, num_class, activation=tf.math.softmax)
    return decision

# class E2EBlock(tf.keras.Model):
#     def __init__(self, output_dims):
#         super(E2EBlock, self).__init__()
#         self.kernel_size = 114  # The number of ROIs
#         self.conv1 = tf.keras.layers.Conv2D(output_dims, (1, self.kernel_size), activation=None)
#         self.conv2 = tf.keras.layers.Conv2D(output_dims, (self.kernel_size, 1), activation=None)
#     def call(self, x):
#         a = self.conv1(x)
#         b = self.conv2(x)
#         return tf.concat([a] * self.kernel_size, 2) + tf.concat([b] * self.kernel_size, 1)
#
# class BrainNetCNN(tf.keras.Model):
#     def __init__(self):
#         super(BrainNetCNN, self).__init__()
#         self.kernel_size = 114
#         self.e2econv1 = E2EBlock(10)
#         self.dropout1 = tf.keras.layers.Dropout(.5)
#         self.e2econv2 = E2EBlock(10)
#         self.dropout2 = tf.keras.layers.Dropout(.5)
#         self.e2n = tf.keras.layers.Conv2D(20, (1, self.kernel_size), activation=None)
#         self.dropout3 = tf.keras.layers.Dropout(.5)
#         self.n2g = tf.keras.layers.Conv2D(40, (self.kernel_size, 1), activation=None)
#         self.dropout4 = tf.keras.layers.Dropout(.5)
#         self.dense1 = tf.keras.layers.Dense(2, activation=None)
#         # self.dropout5 = tf.keras.layers.Dropout(.5)
#         # self.dense2 = tf.keras.layers.Dense(30, activation=None)
#         # self.dropout6 = tf.keras.layers.Dropout(.5)
#         # self.dense3 = tf.keras.layers.Dense(2, activation=None)
#     def call(self, x):
#         x = tf.nn.leaky_relu(self.e2econv1(x))
#         x = self.dropout1(x)
#         x = tf.nn.leaky_relu(self.e2econv2(x))
#         x = self.dropout2(x)
#         x = tf.nn.leaky_relu(self.e2n(x))
#         x = self.dropout3(x)
#         x = tf.nn.leaky_relu(self.n2g(x))
#         x = self.dropout4(x)
#         # x = tf.nn.leaky_relu(self.dense1(x))
#         x = tf.squeeze(tf.keras.activations.softmax(self.dense1(x)))
#         print(x.numpy())
#         # x = self.dropout5(x)
#         # x = tf.nn.leaky_relu(self.dense2(x))
#         # # print(x.shape)
#         # x = self.dropout6(x)
#         # x = tf.squeeze(tf.keras.activations.softmax(self.dense3(x)))
#         # print(x.shape)
#         return x


def MLP_classification(x_data, y_data, train_ind, val_ind, test_ind):
    from sklearn.neural_network import MLPClassifier
    import sklearn

    clf = MLPClassifier(hidden_layer_sizes=(64, 32),  # (64, 32, 16), 64,
                        max_iter=100, alpha=0.0001,
                        activation='relu',
                        solver='adam',  # 'sgd'
                        random_state=manualSeed)

    train_data = x_data[train_ind]
    train_label = np.argmax(y_data[train_ind], axis=1)

    train_t = time.time()
    clf.fit(train_data, train_label)
    print('training time:', time.time() - train_t)

    test_t = time.time()
    pred = clf.predict(x_data[test_ind])
    print('test time:', time.time() - test_t)

    # Compute performance
    lab = np.argmax(y_data[test_ind], axis=1)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(lab, pred).ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))
    return auc, acc, sen, spec, pred, lab


def SVM_classification(x_data, y_data, train_ind, val_ind, test_ind):
    from sklearn.svm import SVC
    import sklearn

    svm = SVC(kernel='linear', C=1.0, random_state=manualSeed)
    train_data = x_data[train_ind]
    train_label = np.argmax(y_data[train_ind], axis=1)

    train_t = time.time()
    svm.fit(train_data, train_label)
    print('training time:', time.time()-train_t)

    test_t = time.time()
    pred = svm.predict(x_data[test_ind])
    print('test time:', time.time() - test_t)

    # Compute performance
    lab = np.argmax(y_data[test_ind], axis=1)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(lab, pred).ravel()
    total = tn + fp + fn + tp
    acc = (tn + tp) / total
    sen = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))
    return auc, acc, sen, spec, pred, lab



def generic_combined_scorer(x1,o1,ii_1,x2,o2,ii_2,y,h):
    s1 = h(x1,y)
    s2 = h(x2,y)
    o1[ii_1] = s1
    o2[ii_2] = s2


def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_ho': '_rois_ho.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


############################## Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = []

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            subject_IDs.append(row['Subject'])

    subject_IDs = np.array(subject_IDs)

    ######################
    enu_subject_IDs = list(enumerate(subject_IDs))
    import random
    random.shuffle(enu_subject_IDs)
    indices, shuffled_subject_IDs = zip(*enu_subject_IDs)
    indices = np.asarray(indices)
    shuffled_subject_IDs = np.asarray(shuffled_subject_IDs)

    return shuffled_subject_IDs, indices
    # return subject_IDs


############################## Get labels for a list of subjects
def get_labels(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['Subject'] in subject_list:
                scores_dict[row['Subject']] = row[score]

    return scores_dict


############################## Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(matrix, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, fnum, step=100, verbose=1)

    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)

    # print("Number of labeled samples %d" % len(train_ind))
    # print("Number of features selected %d" % x_data.shape[1])

    return x_data


def ttest_feature_selection(cur_time, cv, matrix, labels, train_ind):

    trainNormal_idx = np.where(labels[train_ind] == 1)[0]
    trainPatient_idx = np.where(labels[train_ind] == 2)[0]

    matrix2 = matrix[train_ind, :]

    tTestResult = stats.ttest_ind(matrix2[trainNormal_idx, :], matrix2[trainPatient_idx, :])  # two tail t-test
    selectedFeatures = np.where(tTestResult.pvalue < 0.01)[0]

    ###################################
    import scipy.io as sio
    import os

    x_data = matrix[:, selectedFeatures]

    file_path = './Results/10fold_weights/%d/featureIdx_%d.mat' % (cur_time, cv)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sio.savemat(file_path, {'selectedFeatures': list(selectedFeatures)})

    return x_data


# def lasso_feature_selection(matrix, labels, train_ind):
#     clf = linear_model.Lasso(alpha=0.003)
#     clf.fit(matrix[train_ind, :], labels[train_ind])
#     selectedFeaturesIdx = np.where(clf.coef_ != 0)[0]
#
#     x_data = matrix[:, selectedFeaturesIdx]
#     return x_data
#
#
#
def lasso_feature_selection(matrix, labels, train_ind, cv):

    # clf = linear_model.Lasso(alpha=0.0001)
    # clf = linear_model.Lasso(alpha=0.0003)
    # clf = linear_model.Lasso(alpha=0.0006)
    # clf = linear_model.Lasso(alpha=0.001)
    # clf = linear_model.Lasso(alpha=0.002)
    clf = linear_model.Lasso(alpha=0.003)  # 0.741/0.566/0.869/0.7916
    # clf = linear_model.Lasso(alpha=0.004)
    # clf = linear_model.Lasso(alpha=0.005)  # 0.725/0.63/0.8/0.766
    # clf = linear_model.Lasso(alpha=0.006)  # 0.726/0.6/0.82/0.76
    # clf = linear_model.Lasso(alpha=0.01)
    # clf = linear_model.Lasso(alpha=0.02)
    # clf = linear_model.Lasso(alpha=0.03)
    # clf = linear_model.Lasso(alpha=0.035)
    # clf = linear_model.Lasso(alpha=0.04)
    # clf = linear_model.Lasso(alpha=0.05)
    # clf = linear_model.Lasso(alpha=0.06)
    clf.fit(matrix[train_ind, :], labels[train_ind])
    selectedFeaturesIdx = np.where(clf.coef_ != 0)[0]

    LASSO_coef = clf.coef_

    x_data = matrix[:, selectedFeaturesIdx]
    file_path = './feature_selection/featureIdx_%d.mat' % (cv)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    sio.savemat(file_path, {'selectedFeatures': list(selectedFeaturesIdx)})

    # save lasso coefficient values for sensitivity analysis indices
    SA_idx = [114*23+18,
              114*26+107,
              114*55+111,
              114*23+2,
              114*23+78,
              114*38+37,
              114*43+22,
              114*57+58,
              114*18+19,
              114*81+108,
              114*93+28,
              114*23+112,
              114*34+92]

    LASSO_coef_mean = np.mean(LASSO_coef)
    LASSO_coef_SA_idx = LASSO_coef[SA_idx]

    csvfile = open('/DATA/Project/KUMC-GCN_ING/Code/population-gcn/feature_selection/LASSO_coef_SA_idx_%d.csv' % (cv),
                   'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in zip(SA_idx, LASSO_coef_SA_idx):
        csvwriter.writerow(row)
    csvfile.close()

    print(cv, 'mean:', LASSO_coef_mean)

    return x_data


def ElasticNet_feature_selection(matrix, labels, train_ind):
    # regr = ElasticNet(random_state=0, alpha=0.00001)
    # regr = ElasticNet(random_state=0, alpha=0.00003)
    # regr = ElasticNet(random_state=0, alpha=0.00006)
    # regr = ElasticNet(random_state=0, alpha=0.0001)
    # regr = ElasticNet(random_state=0, alpha=0.0003)
    # regr = ElasticNet(random_state=0, alpha=0.0006)
    # regr = ElasticNet(random_state=0, alpha=0.001)
    # regr = ElasticNet(random_state=0, alpha=0.003)
    # regr = ElasticNet(random_state=0, alpha=0.006)
    regr = ElasticNet(random_state=5930, alpha=0.01)  # 0.7/0.533/0.83/0.768
    # regr = ElasticNet(random_state=0, alpha=0.03)
    # regr = ElasticNet(random_state=0, alpha=0.06)

    regr.fit(matrix[train_ind, :], labels[train_ind])
    selectedFeaturesIdx = np.where(regr.coef_ != 0)[0]
    x_data = matrix[:, selectedFeaturesIdx]

    return x_data


def bagging_based_ElasticNet_feature_selection(matrix, labels, train_ind):

    sd = 2300

    spPercent = 0.9  # sampling rate
    numSampling = 10  # number of sampling

    commonFeature_set = []

    for sp in range(numSampling):
        x = np.random.RandomState(seed=manualSeed).choice(train_ind, round(len(train_ind)*spPercent), replace=False)

        # regr = ElasticNet(random_state=0, alpha=0.00001)
        # regr = ElasticNet(random_state=0, alpha=0.00003)
        # regr = ElasticNet(random_state=0, alpha=0.00006)
        regr = ElasticNet(random_state=0, alpha=0.0001)
        # regr = ElasticNet(random_state=0, alpha=0.0003)
        # regr = ElasticNet(random_state=0, alpha=0.0006)
        # regr = ElasticNet(random_state=0, alpha=0.001)
        # regr = ElasticNet(random_state=0, alpha=0.003)
        # regr = ElasticNet(random_state=0, alpha=0.006)
        # regr = ElasticNet(random_state=0, alpha=0.01)
        # regr = ElasticNet(random_state=0, alpha=0.03)
        # regr = ElasticNet(random_state=0, alpha=0.06)

        regr.fit(matrix[x, :], labels[x])
        selectedFeaturesIdx = np.where(regr.coef_ != 0)[0]
        x_data = matrix[:, selectedFeaturesIdx]

        commonFeature_set.append(selectedFeaturesIdx)

    commonFeatures = set.intersection(*map(set, commonFeature_set))
    x_data = matrix[:, list(commonFeatures)]
    return x_data


def bagging_based_lasso_feature_selection(matrix, labels, train_ind):

    sd = 2300

    spPercent = 0.9  # sampling rate
    numSampling = 10  # number of sampling

    commonFeature_set = []

    for sp in range(numSampling):
        x = np.random.RandomState(seed=manualSeed).choice(train_ind, round(len(train_ind)*spPercent), replace=False)

        #clf = linear_model.Lasso(alpha=0.0001)
        #clf = linear_model.Lasso(alpha=0.0003)
        #clf = linear_model.Lasso(alpha=0.0006)
        #clf = linear_model.Lasso(alpha=0.001)
        #clf = linear_model.Lasso(alpha=0.003)
        #clf = linear_model.Lasso(alpha=0.006)
        #clf = linear_model.Lasso(alpha=0.01)
        clf = linear_model.Lasso(alpha=0.03)
        #clf = linear_model.Lasso(alpha=0.06)
        clf.fit(matrix[x, :], labels[x])
        selectedFeaturesIdx = np.where(clf.coef_ != 0)[0]
        commonFeature_set.append(selectedFeaturesIdx)

    commonFeatures = set.intersection(*map(set, commonFeature_set))
    x_data = matrix[:, list(commonFeatures)]
    return x_data


def bagging_based_ttest_feature_selection(cv, matrix, labels, train_ind):

    spPercent = 0.9  # sampling rate
    numSampling = 10  # number of sampling

    trainNormal_idx = np.where(labels[train_ind] == 1)[0]
    trainPatient_idx = np.where(labels[train_ind] == 2)[0]
    #
    commonFeature_set = []
    for sp in range(numSampling):
        x = np.random.RandomState(seed=manualSeed).choice(trainNormal_idx, round(len(trainNormal_idx)*spPercent), replace=False)
        y = np.random.RandomState(seed=manualSeed).choice(trainPatient_idx, round(len(trainPatient_idx) * spPercent), replace=False)
        tTestResult = stats.ttest_ind(matrix[x, :], matrix[y, :])  # two tail t-test
        selectedFeatures = np.where(tTestResult.pvalue < 0.01)[0]
        commonFeature_set.append(selectedFeatures)

    commonFeatures = set.intersection(*map(set, commonFeature_set))
    x_data = matrix[:, list(commonFeatures)]
    # print('num features are', len(commonFeatures))
    #
    # sio.savemat('./Results/10fold_results_group2/selectedfeatures_' + str(cv) + '.mat',
    #             {'commonFeatures': list(commonFeatures)})

    return x_data


# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_labels(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


############################## fMRI connectivity networks
def get_networks(subject_list, variable, isDynamic, isEffective):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    with open('name.data', 'rb') as f:
        name = pickle.load(f)

    with open('alff.data', 'rb') as f:
        alff = pickle.load(f)

    with open('reho.data', 'rb') as f:
        reho = pickle.load(f)


        dynamicset = []
        all_networks = []
        timeseries_set = []

    if isEffective == True:
        fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.1.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.01.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.001.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.2.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.5.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.05.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/Before_Dropout/0.15.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.001.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.01.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.05.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.1.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.15.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.2.mat'))['sparse_f']
        # fc = sio.loadmat(os.path.join('./EffectiveFC/After_Dropout/0.5.mat'))['sparse_f']
        for i in range(len(subject_list)):
            all_networks.append(fc[:, :, i])


        # all_networks_set = np.dstack(all_networks)
        # matrix = np.transpose(all_networks_set, (2,0,1))

        vec_networks = [np.reshape(mat, [1,-1]) for mat in all_networks]
        matrix = np.vstack(vec_networks)

    else:
        for subject in subject_list:
            flname = [i for i in os.listdir(data_folder) if
                      os.path.isfile(os.path.join(data_folder, i)) and subject in i]
            fl = os.path.join(data_folder, flname[0])

            # Estimate connectivity matrix
            timeseries = sio.loadmat(fl)['ROI']

            if variable == 'correlation':
                conn_measure = connectome.ConnectivityMeasure(kind=variable)
                # conn_measure = connectome.ConnectivityMeasure(kind=variable).fit_transform([timeseries])[0]
                # conn_measure_2nd = np.matmul(conn_measure, conn_measure)
                # conn_measure_3rd = np.matmul(conn_measure, conn_measure_2nd)
                # connectivity = conn_measure + conn_measure_2nd + conn_measure_3rd
                ft = conn_measure.fit_transform([timeseries])[0]
            elif variable == 'graph_measure':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                connectivity = conn_measure.fit_transform([timeseries])[0]
                ft = bct.clustering_coef_wu(connectivity)

            timeseries_set.append(timeseries)
            all_networks.append(ft)
            dynamicset.append(np.concatenate((alff[name.index(subject)], reho[name.index(subject)])))

            # all_networks=np.array(all_networks)
            if variable == 'correlation':
                idx = np.triu_indices_from(all_networks[0], 1)
                norm_networks = [np.arctanh(mat) for mat in all_networks]
                vec_networks = [mat[idx] for mat in norm_networks]
                # vec_networks = [mat[idx] for mat in all_networks]
                matrix = np.vstack(vec_networks)
            elif variable == 'graph_measure':
                matrix = np.vstack(all_networks)

            # all_networks_set = np.dstack(all_networks)
            # matrix = np.transpose(all_networks_set, (2,0,1))

    # if isDynamic == True:
    #     dynamicset = np.vstack(dynamicset)
    #     matrix = np.concatenate((matrix, dynamicset), axis=1)
    #
    #
    # with open('./train_data.pkl', 'wb') as filehandle:
    #     pickle.dump(timeseries_set, filehandle)


    return matrix


############################## Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the affinity graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_labels(subject_list, l)

        # quantitative phenotypic scores
        if l in ['Age', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph
