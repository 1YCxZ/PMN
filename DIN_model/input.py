# -*- coding:utf-8 -*-
import numpy as np


def get_batch(data, i, batch_size, genre_info, genre_embedding_size):
    start = i * batch_size
    end = (i + 1) * batch_size
    if end > len(data):
        end = len(data)

    ts = data[start: end]

    u, i, y, sl = [], [], [], []
    ip = []
    ip_len = np.zeros([len(ts), 3, genre_embedding_size], dtype=np.int32)
    for idx, t in enumerate(ts):
        u.append(t[0])  # user
        i.append(t[2])  # movie to predict
        plot = genre_info[t[2]]  # [3]
        plot_len = len(plot)
        divide = 1.0 / plot_len if plot_len != 0 else 0.0
        temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
        ip_len[idx] = temp
        if plot_len < 3:
            padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
            plot = plot + padding
        ip.append(plot)
        y.append(t[3])  # label
        sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int32)
    hist_ip = np.zeros([len(ts), max_sl, 3], np.int32)
    hist_ip_len = np.zeros([len(ts), max_sl, 3, genre_embedding_size],
                           np.float32)  # [batch_size, max_length, 3, plot_embedding_size] 与模型一致

    for idx, t in enumerate(ts):
        for l in range(len(t[1])):  # t[1] represents watch history
            mID = t[1][l][0]
            hist_i[idx][l] = mID
            plot = genre_info[mID]  # [3] movie plot
            plot_len = len(plot)  # movie plot len
            if plot_len < 3:
                padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
                plot = plot + padding
            hist_ip[idx][l] = plot
            divide = 1.0 / plot_len if plot_len != 0 else 0.0
            temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
            hist_ip_len[idx][l] = temp  # k:batch, l:user_watch_idx,

    return (u, i, ip, ip_len, y, hist_i, hist_ip, hist_ip_len, sl)



def get_random_batch(data, batch_size, genre_info, genre_embedding_size):

    random_idx = np.random.choice(len(data), size=batch_size, replace=False)
    ts = []
    for i in random_idx:
        ts.append(data[i])

    u, i, y, sl = [], [], [], []
    ip = []
    ip_len = np.zeros([len(ts), 3, genre_embedding_size], dtype=np.int32)
    for idx, t in enumerate(ts):
        u.append(t[0])  # user
        i.append(t[2])  # movie to predict
        plot = genre_info[t[2]]  # [3]
        plot_len = len(plot)
        divide = 1.0 / plot_len if plot_len != 0 else 0.0
        temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
        ip_len[idx] = temp
        if plot_len < 3:
            padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
            plot = plot + padding
        ip.append(plot)
        y.append(t[3])  # label
        sl.append(len(t[1]))
    max_sl = max(sl)

    hist_i = np.zeros([len(ts), max_sl], np.int32)
    hist_ip = np.zeros([len(ts), max_sl, 3], np.int32)
    hist_ip_len = np.zeros([len(ts), max_sl, 3, genre_embedding_size],
                           np.float32)  # [batch_size, max_length, 3, plot_embedding_size] 与模型一致

    for idx, t in enumerate(ts):
        for l in range(len(t[1])):  # t[1] represents watch history
            mID = t[1][l][0]
            hist_i[idx][l] = mID
            plot = genre_info[mID]  # [3] movie plot
            plot_len = len(plot)  # movie plot len
            if plot_len < 3:
                padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
                plot = plot + padding
            hist_ip[idx][l] = plot
            divide = 1.0 / plot_len if plot_len != 0 else 0.0
            temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
            hist_ip_len[idx][l] = temp  # k:batch, l:user_watch_idx,

    return (u, i, ip, ip_len, y, hist_i, hist_ip, hist_ip_len, sl)


