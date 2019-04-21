# -*- coding: utf-8 -*
import os
import pickle

MOVIE_INFO = "../data/movieinfo-1m.pkl"
movie_info = {}
genres_dict = {}

for idx, line in enumerate(open('ml-1m/movies.dat', 'r', encoding='latin1')):
    # for new data
    # if idx == 0:
    #     continue
    # line = line.strip().split(',')
    try:
        line = line.strip().split('::')
        asin = int(line[0])
        name = ''.join(line[1:-1])
        genres = line[-1].split('|')

        movie_info[asin] = {
            'ID': idx,
            'name': name,
            'genres': genres
        }

        for i in genres:
            if i in genres_dict:
                genres_dict[i] += 1
            else:
                genres_dict[i] = 1
    except:
        print(line)

genres_list = list(sorted(genres_dict.items(), key=lambda x: x[1], reverse=True))
genres_list = ['unk'] + [item[0] for item in genres_list if item[0] != '(no genres listed)']
genre_table = {}  # 电影情节表
reverse_genre_table = {}
# 给电影情节encoding
for idx, i in enumerate(genres_list):
    genre_table[i] = idx
    reverse_genre_table[idx] = i

# 用map过后的电影ID记录每部电影的三个情节
genre_info = {}

for asin in movie_info:
    g_list = movie_info[asin]['genres']
    movie_id = movie_info[asin]['ID']
    maped_genre_list = [genre_table.get(g, 0) for g in g_list][:3]
    genre_info[movie_id] = maped_genre_list


with open(MOVIE_INFO, 'wb') as f:
    pickle.dump(movie_info, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((genre_info, len(genre_table)), f, pickle.HIGHEST_PROTOCOL)
