import pickle
import os


RATINGS_FILE = "ml-1m/ratings.dat"
MOVIE_INFO = "../data/movieinfo-1m.pkl"

# load movie info
with open(MOVIE_INFO, 'rb') as f:
    movie_info = pickle.load(f)
    genre_info, genre_count = pickle.load(f)

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
    # if idx == 0:
    #     continue
    # line = line.strip().split(",")
    line = line.strip().split('::')
    userId = int(line[0])
    asin = int(line[1])
    movied_id = movie_info[asin]['ID']
    rating = float(line[2])
    timestamp = int(line[3])
    if userId in hist:
        hist[userId].append((movied_id, rating, timestamp))
    else:
        hist[userId] = [(movied_id, rating, timestamp)]
    if idx % 100000 == 0:
        print("---processed %s lines---" % idx)

for userId in hist:
    hist[userId] = sorted(hist[userId], key=lambda x: x[2])  # 按照时间排序

with open('../data/hist-1m.pkl', 'wb') as f:
    pickle.dump(hist, f, pickle.HIGHEST_PROTOCOL)

print('user:{}, good:{}'.format(len(hist), len(movie_info)))
print('{}%'.format(file_length / ( len(hist) * len(movie_info) ) * 100))

# file_length:1000209
# user:6040, good:3883
# 4.264679797998748%