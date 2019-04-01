# -*- coding:utf-8 -*-
import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import yaml
import argparse
from PMN_model.input import get_batch, get_random_batch
from PMN_model.model import Model
from sklearn import metrics

parser = argparse.ArgumentParser(description="Run din training.")
parser.add_argument('--config', nargs='?', default='config/config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    model_config = yaml.safe_load(f)["configuration"]

GPU_ID = '0'
gpu_fraction = 0.7
work_space = model_config['work_space']
tf_board = '%s/tfboard' % work_space
model_dir = '%s/nn_models' % work_space
print('set up work space...')
for i in [work_space, model_dir]:
    if not os.path.exists(i):
        os.mkdir(i)

batch_size = 64
genre_embedding_size = model_config['model']['genre_embedding_size']
best_AUC = 0.0


with open('dataset-small.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    genre_info, genre_count, movie_count = pickle.load(f)

def add_summary(summary_writer, global_step, tag, value):
    """
    Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def _eval(sess, model, model_dir, dataset):

    epoch = int((len(test_set) + batch_size - 1) / batch_size)  # 向上取整
    # epoch = int(len(dataset) / batch_size)
    label_list = []
    pred_list = []
    tp, fp, fn, tn = 0, 0, 0, 0  # 混淆矩阵

    for i in range(epoch):
        uij = get_batch(dataset, i, batch_size, genre_info, genre_embedding_size)
        # res = model.debug(sess, uij)
        label_, pred_ = model.eval(sess, uij)
        label_list.extend(list(label_))
        pred_list.extend(list(pred_))
        for idx, _ in enumerate(pred_):
            if pred_[idx] > 0.5 and label_[idx] == 1:
                tp += 1
            if pred_[idx] > 0.5 and label_[idx] == 0:
                fp += 1
            if pred_[idx] <= 0.5 and label_[idx] == 1:
                fn += 1
            if pred_[idx] <= 0.5 and label_[idx] == 0:
                tn += 1

    label_list = np.array(label_list, dtype=np.int32)
    pred_list = np.array(pred_list, dtype=np.float32)
    AUC = metrics.roc_auc_score(label_list, pred_list)
    acc = (tp + tn) / (tp + fp + fn + tn)
    prec = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall != 0 else 0
    # 记录效果最好的模型
    global best_AUC
    if best_AUC < AUC:
        best_AUC = AUC
        model.save(sess, '%s/model.ckpt' % model_dir)
    return AUC, acc, prec, recall, f1


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=GPU_ID)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# CPU
config = tf.ConfigProto(device_count={'GPU': 0})
with tf.Session(config=config) as sess:
    with tf.variable_scope('root'):
        model_train = Model(movie_count, genre_count, model_config, mode='train')
    with tf.variable_scope('root', reuse=True):
        model_eval = Model(movie_count, genre_count, model_config, mode='eval')

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # tensorbord
    train_writer = tf.summary.FileWriter('%s/train/' % tf_board, sess.graph)
    dev_writer = tf.summary.FileWriter('%s/dev/' % tf_board, sess.graph)

    print('Eval_AUC: %.4f\t '
          'Eval_ACC: %.4f\t '
          'Eval_PREC: %.4f\t '
          'Eval_RECALL: %.4f\t '
          'Eval_F1: %.4f' % _eval(sess, model_eval, model_dir, test_set))

    sys.stdout.flush()
    lr = model_config['model']['learning_rate']
    start_time = time.time()
    loss_sum = 0.0

    for i in range(50):
        # batch循环
        print('epoch:%s' % i)
        random.shuffle(train_set)

        epoch = int((len(train_set) + batch_size - 1) / batch_size)  # 向上取整
        for i in range(epoch):
            uij = get_batch(train_set, i, batch_size, genre_info, genre_embedding_size)
            loss = model_train.train(sess, uij, lr)
            loss_sum += loss
            if model_train.global_step.eval() % 100 == 0 and model_train.global_step.eval() != 0:
                test_auc, test_acc, test_prec, test_recall, test_f1 = _eval(sess, model_eval, model_dir, test_set)
                train_auc, train_acc, train_prec, train_recall, train_f1 = _eval(sess, model_eval, model_dir, train_set)
                loss_mean = loss_sum / 100
                print('Global_step %d\t'
                      'Train_loss: %.4f\n'
                      'Train_AUC: %.4f\t'
                      'Train_ACC: %.4f\t'
                      'Train_PREC: %.4f\t'
                      'Train_RECALL: %.4f\t'
                      'Train_F1: %.4f' %
                      (model_train.global_step.eval(),
                       loss_mean, train_auc, train_acc, train_prec, train_recall, train_f1))
                print('Eval_AUC: %.4f\t'
                      'Eval_ACC: %.4f\t'
                      'Eval_PREC: %.4f\t'
                      'Eval_RECALL: %.4f\t'
                      'Eval_F1: %.4f' %
                      (test_auc, test_acc, test_prec, test_recall, test_f1))
                print()
                add_summary(train_writer, model_train.global_step.eval(), 'train loss', loss_mean)
                add_summary(dev_writer, model_train.global_step.eval(), 'test AUC', test_auc)
                sys.stdout.flush()
                loss_sum = 0.0

    print('best test_AUC:', best_AUC)
    sys.stdout.flush()
