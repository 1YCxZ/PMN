import pickle
import os


RATINGS_FILE = "ml-latest-small/ratings.csv"
MOVIE_INFO = "../data/movieinfo-small.pkl"

# load movie info
with open(MOVIE_INFO, 'rb') as f:
    movie_info = pickle.load(f)
    genre_info, genre_count = pickle.load(f)
    movieID_map = pickle.load(f)

hist = {}
preuserid = 1
pre_user_hist = {}

# 文件长度
file_length = 0
for _, line in enumerate(open(RATINGS_FILE, 'r')):
    file_length += 1
print('file_length:%s' % file_length)


print('start processing ratings...')
for idx, line in enumerate(open(RATINGS_FILE, 'r')):
    if idx == 0:
        continue
    line = line.strip().split(",")
    userId = int(line[0])
    raw_movie_id = int(line[1])
    maped_movied_id = movieID_map[raw_movie_id]
    rating = float(line[2])
    timestamp = int(line[3])
    if userId != preuserid or idx == file_length - 1:
        # 添加上一个用户的观看记录
        pre_user_hist = list(sorted(pre_user_hist.items(), key=lambda x: x[1]))
        hist[preuserid] = pre_user_hist
        # 记录当前的新用户
        pre_user_hist = {}
        preuserid = userId
    if maped_movied_id in movie_info:
        pre_user_hist[(maped_movied_id, rating)] = timestamp
    if idx % 100000 == 0:
        print("---processed %s lines---" % idx)


with open('data/hist-small.pkl', 'wb') as f:
    pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)


# file_length:100837
# start processing ratings...
# ---processed 100000 lines---
# user:610, movie:9742