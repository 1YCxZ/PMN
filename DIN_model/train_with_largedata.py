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
from DIN_model.input import thread_get_batch
from DIN_model.model import Model
from sklearn import metrics
import csv
import threading
from multiprocessing import Queue
from datetime import datetime
import time

parser = argparse.ArgumentParser(description="Run din training.")
parser.add_argument('--config', nargs='?', default='config/config.yaml')
args = parser.parse_args()
with open(args.config) as f:
    model_config = yaml.safe_load(f)["configuration"]

GPU_ID = '0'
gpu_fraction = 0.6
work_space = model_config['work_space']
tf_board = '%s/tfboard' % work_space
model_dir = '%s/nn_models' % work_space
print('set up work space...')
for i in [work_space, model_dir]:
    if not os.path.exists(i):
        os.mkdir(i)

batch_size = 512
thread_num = 25
train_queue = Queue(maxsize=500)  # 训练数据缓冲队列
test_queue = Queue(maxsize=500)  # 测试数据缓冲队列

genre_embedding_size = model_config['model']['genre_embedding_size']
print_per_epoch = 200


# 训练测试数据地址
MOVIELENS_TRAIN = 15555161
MOVIELENS_TEST = 3891129
test_file_folder = "../testset/"
train_file_folder = "../trainset/"
def f_l(file_folder):
    file_list = []
    for root, folder, files in os.walk(file_folder):
        for file in files:
            print(os.path.join(root, file))
            file_list.append(os.path.join(root, file))
    return file_list

train_file_list = f_l(train_file_folder)
test_file_list = f_l(test_file_folder)


with open('../data/movieinfo-large.pkl', 'rb') as f:
    movie_info = pickle.load(f)
    genre_info, genre_count = pickle.load(f)
movie_count = len(movie_info)


def add_summary(summary_writer, global_step, tag, value):
    """
    Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def produce_batch_data(thread_num, file_list, data_queue):
    print('\tthread %s start' % thread_num)
    # 线程生成数据
    for path in file_list:
        cur_batch = []  # 当前batch
        f = open(path, 'r', newline='')
        csv_reader = csv.reader(f, dialect='excel')
        for _, line in enumerate(csv_reader):
            # line -> [str, str, str, str]
            uid = int(line[0])
            watch_history = eval(line[1])
            target = int(line[2])
            label = int(line[3])
            cur_batch.append((uid, watch_history, target, label))
            if len(cur_batch) == batch_size:
                random.shuffle(cur_batch)
                _batch = thread_get_batch(cur_batch, genre_info, genre_embedding_size)
                data_queue.put(_batch)
                cur_batch = []
        if len(cur_batch) > 0:
            random.shuffle(cur_batch)
            _batch = thread_get_batch(cur_batch, genre_info, genre_embedding_size)
            data_queue.put(_batch)
        f.close()


def _eval(sess, model, test_file_list):

    test_threads = []  # 线程池

    file_num = len(test_file_list)
    file_num_per_thread = (file_num + thread_num - 1) // thread_num
    for i in range(thread_num):
        random.shuffle(test_file_list)
        sub_file_list = test_file_list[i * file_num_per_thread: (i + 1) * file_num_per_thread]
        t = threading.Thread(target=produce_batch_data, args=(i, sub_file_list, test_queue))
        test_threads.append(t)
    # test数据生产者线程启动
    for t in test_threads:
        t.start()
        time.sleep(0.1)
    time.sleep(5)

    label_list = []
    pred_list = []
    tp, fp, fn, tn = 0, 0, 0, 0  # 混淆矩阵

    # 消费者线程开始，模型forward过程
    test_total_batch = (MOVIELENS_TEST + batch_size - 1) // batch_size
    for _ in range(test_total_batch):
        start = time.time()
        uij = test_queue.get()
        label_, pred_ = model.eval(sess, uij)
        end = time.time()
        print('cost: {:.4f}s/batch'.format(end - start))
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

    for t in test_threads:
        t.join()
    return AUC, prec, recall, f1


# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list=GPU_ID)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# CPU
config = tf.ConfigProto(device_count={'GPU': 0})
with tf.Session(config=config) as sess:
    model = Model(movie_count, genre_count, model_config)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # tensorbord
    train_writer = tf.summary.FileWriter('%s/train/' % tf_board, sess.graph)
    dev_writer = tf.summary.FileWriter('%s/dev/' % tf_board, sess.graph)

    print('Eval_AUC: %.4f\t '
          'Eval_PREC: %.4f\t '
          'Eval_RECALL: %.4f\t '
          'Eval_F1: %.4f' % _eval(sess, model, test_file_list))

    sys.stdout.flush()
    lr = model_config['model']['learning_rate']

    loss_sum = 0.0

    for i in range(50):
        # batch循环
        print('epoch:%s' % i)
        train_threads = []  # 线程池
        file_num = len(train_file_list)
        file_num_per_thread = (file_num + thread_num - 1) // thread_num
        for i in range(thread_num):
            # 打乱文件顺序
            random.shuffle(train_file_list)
            sub_file_list = train_file_list[i * file_num_per_thread: (i + 1) * file_num_per_thread]
            t = threading.Thread(target=produce_batch_data, args=(i, sub_file_list, train_queue))
            train_threads.append(t)
        # train数据生产者线程启动
        for t in train_threads:
            t.start()
            time.sleep(0.1)
        time.sleep(5)

        train_total_batch = (MOVIELENS_TRAIN + batch_size - 1) // batch_size
        for _ in range(train_total_batch):
            uij = train_queue.get()
            loss = model.train(sess, uij, lr)
            loss_sum += loss
            if model.global_step.eval() % print_per_epoch == 0 and model.global_step.eval() != 0:
                test_auc, test_prec, test_recall, test_f1 = _eval(sess, model, test_file_list)
                loss_mean = loss_sum / print_per_epoch
                print('Global_step %d\t'
                      'Train_loss: %.4f\n'
                      'Eval_AUC: %.4f\t'
                      'Eval_PREC: %.4f\t'
                      'Eval_RECALL: %.4f\t'
                      'Eval_F1: %.4f\t'
                      '%s'%
                      (model.global_step.eval(),
                       loss_mean, test_auc, test_prec, test_recall, test_f1, datetime.now()))
                print()
                add_summary(train_writer, model.global_step.eval(), 'train loss', loss_mean)
                add_summary(dev_writer, model.global_step.eval(), 'test AUC', test_auc)
                sys.stdout.flush()
                loss_sum = 0.0

        for t in train_threads:
            t.join()

