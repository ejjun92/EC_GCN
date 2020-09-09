# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import random
from gcn.utils import *
from gcn.models import MLP, Deep_GCN
from gcn.layers import GraphConvolution
import sklearn.metrics

manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)
tf.set_random_seed(manualSeed)

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def run_training(cv, adj, features, labels, idx_train, idx_val, idx_test, params, cur_time):

    # Set random seed
    # random.seed(params['seed'])
    # np.random.seed(params['seed'])
    # tf.set_random_seed(params['seed'])

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    # flags.DEFINE_integer('seed', params['seed'], 'seed.')
    flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden1'], 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')
    flags.DEFINE_float('decay', params['decay'], 'Weight for L2 loss on embedding matrix')

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = Deep_GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = Deep_GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    # model = model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True, name="fold_%d"%cv)

    model = model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True, name="fold_%d" % cv)
    model_predic = tf.nn.softmax(model.outputs)

    # Initialize session
    sess = tf.Session()

    #####################
    summ_trn_writer = tf.summary.FileWriter(logdir="./Summary/%d/%d/trn/"%(cur_time, cv))
    summ_val_writer = tf.summary.FileWriter(logdir="./Summary/%d/%d/val/"%(cur_time, cv))
    summ_op = tf.summary.merge([tf.summary.scalar("accuracy", model.accuracy),
                                tf.summary.scalar("loss", model.loss)])
    saver = tf.train.Saver(max_to_keep=0)
    # print("Summary Folder %s"%("./Summary/%d/%d/trn/"%(cur_time, cv)))
    ####################

    def logger(feed_dict, mode="train", step=0):
        so = sess.run(summ_op, feed_dict=feed_dict)
        if mode == "train":
            summ_trn_writer.add_summary(so, global_step=step)
        elif mode == "valid":
            summ_val_writer.add_summary(so, global_step=step)

    # Define model evaluation function
    def evaluate(feats, graph, label, mask, placeholder):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph, label, mask, placeholder)
        # outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
        outs_val = sess.run([model.loss, model.accuracy, model_predic], feed_dict=feed_dict_val)
        tf.add_to_collection(name='sensitivity analysis', value=model_predic)


        # Compute performance
        pred = outs_val[2]
        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
        lab = label
        lab = lab[np.squeeze(np.argwhere(mask == 1)), :]

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(np.argmax(lab, axis=1), np.argmax(pred, axis=1)).ravel()
        total = tn + fp + fn + tp
        acc = (tn+tp) / total
        sen = tp / (tp+fn)
        spec = tn / (tn+fp)
        auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))

        return outs_val[0], auc, (time.time() - t_test), acc, sen, spec, pred, lab

    # Init variables
    sess.run(tf.global_variables_initializer())
    cost_val = []

    # Train model
    for epoch in range(params['epochs']):

        t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout, placeholders['phase_train']: True})

        # Training step
        # outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict(), model.outputs,
        #                  model.placeholders['labels'], model.placeholders['labels_mask']], feed_dict=feed_dict)

        outs = sess.run([model.opt_op, model.loss, model.accuracy, model_predic, model.outputs,
                         model.placeholders['labels'], model.placeholders['labels_mask']], feed_dict=feed_dict)

        pred = outs[3]
        pred = pred[np.squeeze(np.argwhere(train_mask == 1)), :]
        labs = y_train
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]
        train_auc = sklearn.metrics.roc_auc_score(np.squeeze(labs), np.squeeze(pred))

        training_time = time.time() - t

        t = time.time()

        # Validation
        cost, auc, duration, accuracy, sensitivity, specificity, pred, lab = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)

        #####LOGGER###########
        feed_dict_val = construct_feed_dict(features, support, y_val, val_mask, placeholders)
        feed_dict_val.update({placeholders['phase_train'].name: False})

        logger(feed_dict=feed_dict, mode="train", step=epoch)
        logger(feed_dict=feed_dict_val, mode="valid", step=epoch)
        ######LOGGER##########

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), "train_acc=", "{:.5f}".format(outs[2]),
              "train_auc=", "{:.5f}".format(train_auc), "train_time=", "{:.5f}".format(training_time),
              "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(accuracy), "val_auc=", "{:.5f}".format(auc),
              "validation time=", "{:.5f}".format(duration))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

        saver.save(sess=sess, save_path="./Summary/%d/%d/%d.ckpt"%(cur_time, cv, epoch))

    print("Optimization Finished!")

    # Testing
    sess.run(tf.local_variables_initializer())
    test_cost, test_auc, test_duration, test_accuracy, test_sensitivity, test_specificity, pred, lab = evaluate(features, support, y_test, test_mask, placeholders)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_accuracy),
          "auc=", "{:.5f}".format(test_auc),
          "Test time=", "{:.5f}".format(test_duration))

    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    # print(tvars_vals)

    import scipy.io as sio
    import os


    file_path = './Results/10fold_weights/%d/weight_%d.mat' % (cur_time, cv)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # sio.savemat(file_path, {'W': tvars_vals})
    #
    file_0 = './SA_files/feat_%d.mat.npy' % cv
    np.save(file_0, features)
    file_1 = './SA_files/support_%d.mat.npy' % cv
    np.save(file_1, support)
    file_2 = './SA_files/train_mask_%d.mat.npy' % cv
    np.save(file_2, train_mask)
    file_3 = './SA_files/lab_%d.mat.npy' % cv
    np.save(file_3, y_train)


    # import scipy.io as sio
    #
    # sio.savemat('./Results/10fold_results_group2/testset_results' + str(cv) + '.mat',
    #             {'test_label': test_lab, 'test_prediction': test_pred})

    return test_auc, test_accuracy, test_sensitivity, test_specificity, pred, lab


# def run_test(cv, adj, features, labels, idx_train, idx_val, idx_test, params, cur_time):
#     # Set random seed
#     random.seed(params['seed'])
#     np.random.seed(params['seed'])
#     tf.set_random_seed(params['seed'])
#
#     # Settings
#     flags = tf.app.flags
#     FLAGS = flags.FLAGS
#     flags.DEFINE_string('model', params['model'], 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
#     flags.DEFINE_float('learning_rate', params['lrate'], 'Initial learning rate.')
#     flags.DEFINE_integer('epochs', params['epochs'], 'Number of epochs to train.')
#     flags.DEFINE_integer('hidden1', params['hidden1'], 'Number of units in hidden layer 1.')
#     flags.DEFINE_float('dropout', params['dropout'], 'Dropout rate (1 - keep probability).')
#     flags.DEFINE_float('weight_decay', params['decay'], 'Weight for L2 loss on embedding matrix.')
#     flags.DEFINE_integer('early_stopping', params['early_stopping'], 'Tolerance for early stopping (# of epochs).')
#     flags.DEFINE_integer('max_degree', params['max_degree'], 'Maximum Chebyshev polynomial degree.')
#     flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')
#
#     # Create test, val and train masked variables
#     y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(labels, idx_train, idx_val, idx_test)
#
#     # Some preprocessing
#     features = preprocess_features(features)
#     if FLAGS.model == 'gcn':
#         support = [preprocess_adj(adj)]
#         num_supports = 1
#         model_func = Deep_GCN
#     elif FLAGS.model == 'gcn_cheby':
#         support = chebyshev_polynomials(adj, FLAGS.max_degree)
#         num_supports = 1 + FLAGS.max_degree
#         model_func = Deep_GCN
#     elif FLAGS.model == 'dense':
#         support = [preprocess_adj(adj)]  # Not used
#         num_supports = 1
#         model_func = MLP
#     else:
#         raise ValueError('Invalid argument for GCN model ')
#
#     # Define placeholders
#     placeholders = {
#         'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
#         'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
#         'phase_train': tf.placeholder_with_default(False, shape=()),
#         'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
#         'labels_mask': tf.placeholder(tf.int32),
#         'dropout': tf.placeholder_with_default(0., shape=()),
#         'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
#     }
#
#     # Create model
#     model = model_func(placeholders, input_dim=features[2][1], depth=FLAGS.depth, logging=True)
