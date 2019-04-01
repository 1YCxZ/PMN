# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from Dice import dice

class Model(object):

    def __init__(self, user_count, item_count, plot_count, plot_num, model_config):

        self.u = tf.placeholder(tf.int32, [None, ])  # [B] user
        self.i = tf.placeholder(tf.int32, [None, ])  # [B] item
        self.ip = tf.placeholder(tf.int32, [None, plot_num])  # 每部电影选取三个情节embedding
        self.ip_len = tf.placeholder(tf.float32, [None, plot_num, None])  # 每部电影拥有的情节的个数
        self.y = tf.placeholder(tf.float32, [None, ])  # [B] label
        self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T] 用户历史观看记录
        self.hist_ip = tf.placeholder(tf.int32, [None, None, plot_num])  # [B, T, 3] 每部电影三个情节embedding
        self.hist_ip_len = tf.placeholder(tf.float32, [None, None, plot_num, None])  # [B, T, 3, item_embedding]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B] 用户观看历史记录长度
        self.lr = tf.placeholder(tf.float64, [])
        # model_config
        self.hidden_units = model_config['model']['hidden_units']
        self.item_embedding_size = model_config['model']['item_embedding_size']
        self.plot_embedding_size = model_config['model']['plot_embedding_size']
        self.fc1_size = model_config['model']['fc1_size']
        self.fc2_size = model_config['model']['fc2_size']
        self.fc1_attn_size = model_config['model']['fc1_attn_size']
        self.fc2_attn_size = model_config['model']['fc2_attn_size']

        assert self.item_embedding_size + self.plot_embedding_size == self.hidden_units
        initializer = tf.contrib.layers.xavier_initializer()
        # 加入user_emb似乎会造成过拟合
        # self.user_emb_w = tf.get_variable("user_emb_w",
        #                              [user_count, self.item_embedding_size],
        #                              initializer=initializer,
        #                              dtype=tf.float32)
        self.item_emb_w = tf.get_variable("item_emb_w",
                                          [item_count, self.item_embedding_size],
                                          initializer=initializer,
                                          dtype=tf.float32)
        item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
        self.plot_emb_w = tf.get_variable("plot_emb_w",
                                          [plot_count, self.plot_embedding_size],
                                          initializer=initializer,
                                          dtype=tf.float32)

        #  候选物品 embedding
        i_plot_emb = tf.nn.embedding_lookup(self.plot_emb_w, self.ip)  # 这里面会有不到3个情节的电影
        i_emb = tf.concat(values=[
            tf.nn.embedding_lookup(self.item_emb_w, self.i),  # [B, item_emb]
            tf.reduce_sum(i_plot_emb * self.ip_len, axis=1)  # [B, 3, plot_emb] -> [B, plot_emb] 对3个情节取平均
        ], axis=-1)
        i_b = tf.gather(item_b, self.i)

        #  历史观看记录 embedding
        h_plot_emb = tf.nn.embedding_lookup(self.plot_emb_w, self.hist_ip)
        h_emb = tf.concat(values=[
            tf.nn.embedding_lookup(self.item_emb_w, self.hist_i),  # [B, T, item_emb]
            tf.reduce_sum(h_plot_emb * self.hist_ip_len, axis=2)  # [B, T, 3, plot_emb] -> [B, T, plot_emb] 对三个情节取平均
        ], axis=-1)

        hist = self.attention(i_emb, h_emb, self.sl)  # [B, 1, H] queries, keys, keys_length

        hist = tf.layers.batch_normalization(inputs=hist)
        hist = tf.reshape(hist, [-1, self.hidden_units], name='hist_bn')
        hist = tf.layers.dense(hist, self.hidden_units, name='hist_fcn')
        u_emb = hist

        din_i = tf.concat([u_emb, i_emb], axis=-1)

        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, self.fc1_size, activation=None, name='fc1')
        d_layer_1_i = dice(d_layer_1_i, name='dice_1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, self.fc2_size, activation=None, name='fc2')
        d_layer_2_i = dice(d_layer_2_i, name='dice_2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='fc3')
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        self.logits = i_b + d_layer_3_i  # for training loss

        self.prediction = tf.nn.sigmoid(self.logits)  # [batch_size]

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step', dtype=tf.int32)
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.y)
        )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train(self, sess, uij, l):
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.ip: uij[2],
            self.ip_len: uij[3],
            self.y: uij[4],
            self.hist_i: uij[5],
            self.hist_ip: uij[6],
            self.hist_ip_len: uij[7],
            self.sl: uij[8],
            self.lr: l,
        })
        return loss

    def debug(self, sess, uij, l):
        r_list = [self.loss,
                  self.prediction,
                  self.y,
                  self.item_emb_w,
                  self.plot_emb_w,
                  self.train_op
                  ]
        res = sess.run(r_list, feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.ip: uij[2],
            self.ip_len: uij[3],
            self.y: uij[4],
            self.hist_i: uij[5],
            self.hist_ip: uij[6],
            self.hist_ip_len: uij[7],
            self.sl: uij[8],
            self.lr: l,
        })
        return res

    def eval(self, sess, uij):
        label, pred = sess.run([self.y, self.prediction], feed_dict={
            self.u: uij[0],
            self.i: uij[1],
            self.ip: uij[2],
            self.ip_len: uij[3],
            self.y: uij[4],
            self.hist_i: uij[5],
            self.hist_ip: uij[6],
            self.hist_ip_len: uij[7],
            self.sl: uij[8]
        })
        return label, pred

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, logdir):
        """
        Load the latest checkpoint
        """
        saver = tf.train.Saver()
        print("Trying to restore saved checkpoints from {} ...".format(logdir),
              end="")
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            # vars = tf.train.list_variables(ckpt.model_checkpoint_path)
            # print('checkpoint_variables:')
            # pprint(vars)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(" Done.")
        else:
            print(" No checkpoint found.")

    def attention(self, queries, keys, keys_length):
        '''
          queries:     [Batch_size, movie_embedding]
          keys:        [Batch_size, history_length, movie_embedding]
          keys_length: [Batch_size]
        '''
        queries_hidden_units = queries.get_shape().as_list()[-1]
        queries = tf.tile(queries, [1, tf.shape(keys)[1]])  # tf.tile在这个维度内横向复制
        queries = tf.reshape(
            queries, [-1, tf.shape(keys)[1], queries_hidden_units])
        din_all = tf.concat([queries, keys, queries - keys, queries * keys], axis=-1)
        d_layer_1_all = tf.layers.dense(
            din_all, self.fc1_attn_size, activation=tf.nn.sigmoid, name='fc1_att')
        d_layer_2_all = tf.layers.dense(
            d_layer_1_all, self.fc2_attn_size, activation=tf.nn.sigmoid, name='fc2_att')
        d_layer_3_all = tf.layers.dense(
            d_layer_2_all, 1, activation=None, name='fc3_att')
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
        outputs = d_layer_3_all
        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

        # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

        # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]

        return outputs





