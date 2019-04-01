import os
import pickle

MOVIE_INFO = "../data/movieinfo-small.pkl"
movie_info = {}
genres_dict = {}

movieID_map = {}

for idx, line in enumerate(open('ml-latest-small/movies.csv','r')):
    if idx == 0:
        continue
    line = line.strip().split(',')
    id = int(line[0])
    name = ''.join(line[1:-1])
    genres = line[-1].split('|')

    movie_info[idx-1] = {
        'movielensID': id,
        'name': name,
        'genres': genres
    }
    movieID_map[id] = idx-1
    for i in genres:
        if i in genres_dict:
            genres_dict[i] += 1
        else:
            genres_dict[i] = 1

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

for i in movie_info:
    g_list = movie_info[i]['genres']
    maped_genre_list = [genre_table.get(g, 0) for g in g_list][:3]
    genre_info[i] = maped_genre_list


with open(MOVIE_INFO, 'wb') as f:
    pickle.dump(movie_info, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((genre_info, len(genre_table)), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(movieID_map, f, pickle.HIGHEST_PROTOCOL)