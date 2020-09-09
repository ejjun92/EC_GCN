# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import time
import argparse

# import networkx as nx
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio

# import ABIDEParser as Reader
import utils as Reader
import train_GCN as Train
import random


import tensorflow as tf
# from tensorflow.keras.backend import eval
# tf.enable_eager_execution()


import os
import csv
import numpy as np
import scipy.io as sio
import bct

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import RFE
from nilearn import connectome
from scipy import stats



import os
import shutil

import json

from tempfile import mkdtemp

from tqdm import tqdm


import numpy as np

from joblib import Parallel, delayed

import depmeas

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = ''


manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(cv, train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs, cur_time):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (num_subjects x 2)
        params          : dictionnary of GCNs parameters
        subject_IDs     : list of subject IDs

    #returns:

        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the linear classifier
        lin_auc     : average area under curve over the test samples using the linear classifier
        fold_size   : number of test samples
    """

    # feature selection/dimensionality reduction step
    # x_data = features
    x_data = Reader.lasso_feature_selection(features, y, train_ind, cv)
    # x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])
    # x_data = Reader.feature_selection(features, y, train_ind, params['num_features'])  # no need to consider site info.
    # x_data = Reader.ttest_feature_selection(cur_time, cv, features, y, train_ind)
    # x_data = Reader.bagging_based_ttest_feature_selection(cv, features, y, train_ind)
    # x_data = Reader.ElasticNet_feature_selection(features, y, train_ind)
    # x_data = Reader.bagging_based_ElasticNet_feature_selection(features, y, train_ind)
    # x_data = Reader.bagging_based_lasso_feature_selection(features, y, train_ind)
    print('fold: ' + str(cv) + ', shape: ', np.shape(x_data))

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric='correlation')

    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)

    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph

    # Classification by BrainNetCNN
    # import tensorflow as tf
    # sess = tf.Session()

    # brainnetcnn = Reader.BrainNetCNN(np.reshape(x_data, [x_data.shape[0], 114, -1, 1]))
    # test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab = Reader.calculate_performance(eval(brainnetcnn), y_data, train_ind, val_ind, test_ind)


    # outs_val = sess.run(Reader.BrainNetCNN, feed_dict=np.reshape(x_data, [x_data.shape[0], 114, -1, 1]))


    # Classification by MLP
    # test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab = Reader.MLP_classification(x_data, y_data, train_ind, val_ind, test_ind)

    # Classification with SVM
    # test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab = Reader.SVM_classification(x_data, y_data, train_ind, val_ind, test_ind)

    # Classification by GCNs
    test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab = Train.run_training(cv, final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params, cur_time)


    return test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab


def main():
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ABIDE dataset')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden1', default=32, type=int, help='Number of filters in hidden layers (default: 16)')
    # parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--lrate', default=1e-2, type=float, help='Initial learning rate (default: 0.005)')
    # parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
    #                                                   'see preprocessed-connectomes-project.org/abide/Pipelines.html '
    #                                                   'for more options )')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=2000, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 2000)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the GCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    # parser.add_argument('--seed', default=89, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=11, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=200, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')
    parser.add_argument('--train', default=1, type=int)

    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params['model'] = args.model                    # gcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden1'] = args.hidden1                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix
    params['early_stopping'] = params['epochs']     # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    # params['seed'] = args.seed                      # seed for random initialisation

    # GCN Parameters
    params['num_features'] = args.num_features      # number of features for feature selection step
    params['num_training'] = args.num_training      # percentage of training set used for training
    params['train'] = args.train      # percentage of training set used for training
    # atlas = args.atlas                              # atlas for network construction (node definition)
    # connectivity = args.connectivity                # type of connectivity used for network construction

    # Get class labels
    # subject_IDs = Reader.get_ids()
    ##################################################################
    subject_IDs, shuffled_indices = Reader.get_ids()
    ##################################################################

    labels = Reader.get_labels(subject_IDs, score='DX_Group')  # labels

    # Get acquisition site
    # ####### sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    ########## unique = np.unique(list(sites.values())).tolist()

    num_classes = 2  # MDD or HC
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    ########## site = np.zeros([num_nodes, 1], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])
        ########## site[i] = unique.index(sites[subject_IDs[i]])

    import pickle
    # with open('./label.pkl', 'wb') as filehandle:
    #     pickle.dump(np.argmax(y_data, axis=1), filehandle)


    # Compute feature vectors (vectorised connectivity networks)
    ####### Granger Causality Analysis
    # data_fld = './granger_casuality'
    # features = Reader.load_ec_GCA(subject_IDs, data_fld)
    #######


    features = Reader.get_networks(subject_IDs, variable='correlation', isDynamic=False, isEffective=True)
    ############################################################
    shuffled_features = features[shuffled_indices]
    features = shuffled_features.copy()
    ############################################################
    # features = Reader.get_networks(subject_IDs, variable='graph_measure', isDynamic=True)


    # np.save('./MDD_dataset/features_GCA.npy', features)
    # np.save('./MDD_dataset/labels.npy', np.argmax(y_data, axis=1))


    # Compute population graph using gender and acquisition site
    graph = Reader.create_affinity_graph_from_scores(['Age', 'Sex'], subject_IDs)
    # graph = Reader.create_affinity_graph_from_scores(['Sex'], subject_IDs)


    # Folds for cross validation experiments
    #num_samples = np.shape(features)[0]
    skf = StratifiedKFold(n_splits=10)
    #loo = LeaveOneOut()

    train_ind_set = []
    test_ind_set = []
    for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))):
        train_ind_set.append(train_ind)
        test_ind_set.append(test_ind)
    cur_time = time.time()

    # import pickle
    # with open('./MDD_dataset/train_ind.pkl', 'wb') as filehandle:
    #     pickle.dump(train_ind_set, filehandle)
    # with open('./MDD_dataset/test_ind.pkl', 'wb') as filehandle:
    #     pickle.dump(test_ind_set, filehandle)


    if args.folds == 11:  # run cross validation on all folds
        scores = Parallel(n_jobs=10)(delayed(train_fold)(cv, train_ind, test_ind, test_ind, graph, features, y, y_data,
                                                         params, subject_IDs, cur_time)
                                     for train_ind, test_ind, cv in zip(train_ind_set, test_ind_set, range(10)))

        test_auc = [x[0] for x in scores]
        test_accuracy = [x[1] for x in scores]
        test_sensitivity = [x[2] for x in scores]
        test_specificity = [x[3] for x in scores]
        test_pred = [x[4] for x in scores]
        test_lab = [x[5] for x in scores]

        print('Accuracy : ' + str(np.mean(test_accuracy)) + ' + ' + str(np.std(test_accuracy)))
        print('Sensitivity : ' + str(np.mean(test_sensitivity)) + ' + ' + str(np.std(test_sensitivity)))
        print('Specificity : ' + str(np.mean(test_specificity)) + ' + ' + str(np.std(test_specificity)))
        print('AUC : ' + str(np.mean(test_auc)) + ' + ' + str(np.std(test_auc)))

        # np.savez('./statistical_test/FC_Lasso_MLP_pred.npz', pred=test_pred, allow_pickle=True)
        # np.savez('./statistical_test/FC_Lasso_MLP_lab.npz', lab=test_lab, allow_pickle=True)

    else:  # compute results for only one fold

        cv_splits = list(skf.split(features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]

        val = test

        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold(train, test, val, graph, features, y, y_data, params, subject_IDs, cur_time)

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / fold_size))
        print('overall AUC %f' + str(np.mean(scores_auc)))

        # if args.save == 1:
        #     result_name = 'MDD_classification'
        #     sio.savemat('./results/' + result_name + '.mat',
        #                 {'lin': scores_lin, 'lin_auc': scores_auc_lin,
        #                  'acc': scores_acc, 'auc': scores_auc, 'folds': fold_size})
    # else:
    #     testf

if __name__ == "__main__":
    main()
