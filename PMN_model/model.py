# -*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from PMN_model.modules import multihead_attention, ff, positional_encoding

class Model(object):

    def __init__(self,
                 movie_count,
                 genre_count,
                 model_config,
                 mode='train',
                 pos_emb_maxlen=100):

        self.user = tf.placeholder(tf.int32, [None, ])  # [B] user
        self.candidate = tf.placeholder(tf.int32, [None, ])  # [B] candidate

        self.candidate_genre = tf.placeholder(tf.int32, [None, 3])  # 每部电影选取三个情节embedding
        self.candidate_genre_len = tf.placeholder(tf.float32, [None, 3, None])

        self.history_movie = tf.placeholder(tf.int32, [None, None])  # [B, T] 用户历史观看记录
        self.history_score = tf.placeholder(tf.float32, [None, None, 1])  # [B, T, 1] 用户历史评分

        self.history_genre = tf.placeholder(tf.int32, [None, None, 3])  # [B, T, 3] 每部电影三个情节embedding
        self.history_genre_len = tf.placeholder(tf.float32, [None, None, 3, None])  # [B, T, 3, genre_embedding]


        self.label = tf.placeholder(tf.float32, [None, ])  # [B] label
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B] 用户观看历史记录长度
        self.lr = tf.placeholder(tf.float64, [])

        # 每个batch的数据的信息
        self.max_sl = tf.reduce_max(self.sl, name='max_histroy_len')
        self.batch_size = tf.shape(self.user)[0]

        # model_config
        self.movie_embedding_size = model_config['model']['movie_embedding_size']
        self.genre_embedding_size = model_config['model']['genre_embedding_size']
        self.hidden_units = (self.movie_embedding_size + self.genre_embedding_size) * 2
        self.mlp_1_units = model_config['model']['mlp_1_units']
        self.mlp_2_units = model_config['model']['mlp_2_units']
        self.num_heads = model_config['transformer']['num_heads']
        self.dropout_rate = model_config['transformer']['dropout_rate']
        self.num_blocks = model_config['transformer']['num_blocks']

        initializer = tf.contrib.layers.xavier_initializer()
        self.movie_embedding = tf.get_variable('movie_embedding_w',
                                    [movie_count, self.movie_embedding_size],
                                     initializer=initializer,
                                     dtype=tf.float32)


        self.genre_embedding = tf.get_variable('genre_embedding',
                                               [genre_count, self.genre_embedding_size],
                                               initializer=initializer,
                                               dtype=tf.float32)

        ######特殊token embedding######
        self.CLS = tf.get_variable("CLS_embedding", [1, self.hidden_units], initializer=initializer, dtype=tf.float32)

        ######候选观看节目特征######
        candidate_embed = tf.nn.embedding_lookup(self.movie_embedding, self.candidate)  # [batch_size, embedding_size]
        candidate_genre_emb = tf.nn.embedding_lookup(self.genre_embedding, self.candidate_genre)  # 这里面会有不到3个情节的电影
        candidate_embed = tf.concat(values=[
            candidate_embed,  # [B, item_emb]
            tf.reduce_sum(candidate_genre_emb * self.candidate_genre_len, axis=1)  # [B, 3, genre_emb] -> [B, genre_emb] 对3个情节取平均
        ], axis=-1)


        ######历史观看节目特征######
        self.history_movie_embed = tf.nn.embedding_lookup(self.movie_embedding, self.history_movie)  #[batch_size, hist_max_len, embedding_size]
        history_genre_emb = tf.nn.embedding_lookup(self.genre_embedding, self.history_genre)
        self.history_movie_embed = tf.concat(values=[
            self.history_movie_embed,  # [B, T, item_emb]
            tf.reduce_sum(history_genre_emb * self.history_genre_len, axis=2)  # [B, T, 3, genre_emb] -> [B, T, genre_emb] 对三个情节取平均
        ], axis=-1)

        ######将候选观看和历史观看的每个都做concat，得到matching_vector
        candidate_embed = tf.expand_dims(candidate_embed, axis=1)
        self.candidate_embed = tf.tile(candidate_embed, multiples=[1, self.max_sl, 1])
        matching_vector = tf.concat([self.history_movie_embed, self.candidate_embed], axis=-1)  # [batch_size, max_lengh, m_emb+g_emb]
        # matching_vector = tf.layers.dense(matching_vector, self.hidden_units, activation=tf.nn.sigmoid, name='match_vector_dense')


        ######对于matching_vector进行mask，padding用0
        key_masks = tf.sequence_mask(self.sl, tf.shape(matching_vector)[1])
        key_masks = tf.expand_dims(key_masks, -1)
        key_masks = tf.tile(key_masks, multiples=[1, 1, self.hidden_units])
        paddings = tf.ones_like(matching_vector) * .0
        matching_vector = tf.where(key_masks, matching_vector, paddings)

        ######分数激活######
        matching_score = tf.layers.dense(self.history_score, self.hidden_units, name='macth_score_dense')  # 是否使用bias
        matching_score = tf.nn.sigmoid(matching_score)

        ######分数的重要程度反应在matching vector上
        matching_vector = matching_vector * matching_score  # [batch_size, hist_max_len, embedding_size]

        ########位置信息编码
        matching_vector *= self.hidden_units ** 0.5  # scale
        matching_vector += positional_encoding(matching_vector, pos_emb_maxlen)

        ###### 第一个位置加入用于分类的特殊token
        token = tf.expand_dims(self.CLS, [0])
        token = tf.tile(token, multiples=[self.batch_size, 1, 1])
        matching_vector = tf.concat([token, matching_vector], axis=1)
        ### 加入BN层
        matching_vector = tf.layers.batch_normalization(matching_vector)

        # 送入transformer
        # self_attention提取信息
        ## Blocks
        self.num_blocks = self.num_blocks if self.num_blocks else 1
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                enc_att = multihead_attention(queries=matching_vector,
                                          keys=matching_vector,
                                          values=matching_vector,
                                          num_heads=self.num_heads,
                                          dropout_rate=self.dropout_rate,
                                          training=(mode == 'train'),
                                          causality=False)
                # feed forward
                enc = ff(enc_att, num_units=[self.hidden_units, self.hidden_units])


        #取第一个位置的token代表整个句子
        enc_token = enc[:, :1, :]
        # enc_token = tf.layers.batch_normalization(enc_token)

        # self.mlp_1 = tf.layers.dense(self.enc_token, self.mlp_1_units, activation=tf.nn.sigmoid, name='mlp_1')
        mlp_2 = tf.layers.dense(enc_token, self.mlp_2_units, activation=tf.nn.relu, name='mlp_2')
        mlp_3 = tf.layers.dense(mlp_2, 1, activation=None, name='mlp_3')
        mlp_3 = tf.reshape(mlp_3, [-1])
        self.logits = mlp_3  # for training loss

        self.prediction = tf.nn.sigmoid(self.logits)  # [batch_size]

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step', dtype=tf.int32)
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.label)
        )

        self.total_loss = self.loss

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.total_loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)


    def train(self, sess, uij, l):
        loss, _  = sess.run([self.loss, self.train_op], feed_dict={
            self.user: uij[0],
            self.candidate: uij[1],
            self.candidate_genre: uij[2],
            self.candidate_genre_len: uij[3],
            self.history_movie: uij[4],
            self.history_score: uij[5],
            self.history_genre: uij[6],
            self.history_genre_len: uij[7],
            self.label: uij[8],
            self.sl: uij[9],
            self.lr: l,
        })
        return loss

    def debug(self, sess, uij):
        r_list = [
                  self.history_movie_embed,
                  self.candidate_embed,
                  self.logits,
                  self.prediction,
                  ]
        res = sess.run(r_list, feed_dict={
            self.user: uij[0],
            self.candidate: uij[1],
            self.candidate_genre: uij[2],
            self.candidate_genre_len: uij[3],
            self.history_movie: uij[4],
            self.history_score: uij[5],
            self.history_genre: uij[6],
            self.history_genre_len: uij[7],
            self.label: uij[8],
            self.sl: uij[9]
        })
        return res

    def eval(self, sess, uij):
        label, pred = sess.run([self.label, self.prediction], feed_dict={
            self.user: uij[0],
            self.candidate: uij[1],
            self.candidate_genre: uij[2],
            self.candidate_genre_len: uij[3],
            self.history_movie: uij[4],
            self.history_score: uij[5],
            self.history_genre: uij[6],
            self.history_genre_len: uij[7],
            self.label: uij[8],
            self.sl: uij[9]
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






