# -*- coding:utf-8 -*-
import numpy as np


def get_batch(data, i, batch_size, genre_info, genre_embedding_size):

    start = i * batch_size
    end = (i + 1) * batch_size
    if end > len(data):
        end = len(data)

    ts = data[start: end]

    user, candidate, label, sl = [], [], [], []
    c_genre = []
    c_genre_len = np.zeros([len(ts), 3, genre_embedding_size], dtype=np.int32)
    for idx, t in enumerate(ts):
        user.append(t[0])  # user
        candidate.append(t[2])  # movie to predict 经过映射的ID
        label.append(t[3])  # label
        sl.append(len(t[1]))

        genre = genre_info[t[2]]
        genre_len = len(genre)
        divide = 1.0 / genre_len if genre_len != 0 else 0.0
        temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
        c_genre_len[idx] = temp
        if genre_len < 3:
            padding = [0] * (3 - genre_len)  # 0 respresents 'unk'
            genre = genre + padding
        c_genre.append(genre)

    max_sl = max(sl)

    history_movie = np.zeros([len(ts), max_sl], np.int32)
    history_score = np.zeros([len(ts), max_sl, 1], np.float32)

    history_genre = np.zeros([len(ts), max_sl, 3], np.int32)
    history_genre_len = np.zeros([len(ts), max_sl, 3, genre_embedding_size], np.float32)  # [batch_size, max_length, 3, plot_embedding_size] 与模型一致

    for idx, t in enumerate(ts):
        for l in range(len(t[1])):  # t[1] represents watch history
            mID = t[1][l][0]  # mID remap后的电影ID
            score = t[1][l][1]
            history_movie[idx][l] = mID
            history_score[idx][l][0] = score

            genre = genre_info[mID]  # [3] movie plot
            plot_len = len(genre)  # movie plot len
            if plot_len < 3:
                padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
                genre = genre + padding
            history_genre[idx][l] = genre
            divide = 1.0 / plot_len if plot_len != 0 else 0.0
            temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
            history_genre_len[idx][l] = temp  # k:batch, l:user_watch_idx,

    return (user, candidate,c_genre, c_genre_len, history_movie, history_score,  history_genre, history_genre_len, label, sl)


def get_random_batch(data, batch_size, genre_info, genre_embedding_size):

    random_idx = np.random.choice(len(data), size=batch_size, replace=False)
    ts = []
    for i in random_idx:
        ts.append(data[i])

    user, candidate, label, sl = [], [], [], []
    c_genre = []
    c_genre_len = np.zeros([len(ts), 3, genre_embedding_size], dtype=np.int32)
    for idx, t in enumerate(ts):
        user.append(t[0])  # user
        candidate.append(t[2])  # movie to predict 经过映射的ID
        label.append(t[3])  # label
        sl.append(len(t[1]))

        genre = genre_info[t[2]]
        genre_len = len(genre)
        divide = 1.0 / genre_len if genre_len != 0 else 0.0
        temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
        c_genre_len[idx] = temp
        if genre_len < 3:
            padding = [0] * (3 - genre_len)  # 0 respresents 'unk'
            genre = genre + padding
        c_genre.append(genre)

    max_sl = max(sl)

    history_movie = np.zeros([len(ts), max_sl], np.int32)
    history_score = np.zeros([len(ts), max_sl, 1], np.float32)

    history_genre = np.zeros([len(ts), max_sl, 3], np.int32)
    history_genre_len = np.zeros([len(ts), max_sl, 3, genre_embedding_size],
                                 np.float32)  # [batch_size, max_length, 3, plot_embedding_size] 与模型一致

    for idx, t in enumerate(ts):
        for l in range(len(t[1])):  # t[1] represents watch history
            mID = t[1][l][0]  # mID remap后的电影ID
            score = t[1][l][1]
            history_movie[idx][l] = mID
            history_score[idx][l][0] = score

            genre = genre_info[mID]  # [3] movie plot
            plot_len = len(genre)  # movie plot len
            if plot_len < 3:
                padding = [0] * (3 - plot_len)  # 0 respresents 'unk'
                genre = genre + padding
            history_genre[idx][l] = genre
            divide = 1.0 / plot_len if plot_len != 0 else 0.0
            temp = divide * np.ones([3, genre_embedding_size], dtype=np.float32)
            history_genre_len[idx][l] = temp  # k:batch, l:user_watch_idx,

    return (
    user, candidate, c_genre, c_genre_len, history_movie, history_score, history_genre, history_genre_len, label, sl)


