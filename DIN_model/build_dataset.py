import random
import pickle


random.seed(1234)
# movielens:
HISTORY_WATCH = '../data/hist-small.pkl'
MOVIE_INFO = "../data/movieinfo-small.pkl"


with open(HISTORY_WATCH, 'rb') as f:
    hist = pickle.load(f)

with open(MOVIE_INFO, 'rb') as f:
    movie_info = pickle.load(f)
    genre_info, genre_count = pickle.load(f)

train_set = []
test_set = []
train_pos_count = 0
train_neg_count = 0
test_pos_count = 0
test_neg_count = 0

data_set = []

# movielens:
for userID in hist:
    history_list = []
    u_his = hist[userID]
    for log in u_his:
        mID = log[0][0]
        score = log[0][1]
        timestamp = log[1]
        history_list.append((mID, score))  # 历史观看和评分
        label = 1 if score > 3 else 0
        if len(history_list) < 5:
            continue
        data_set.append((userID, history_list[-51:-1], mID, label))


for i in data_set:
    if random.uniform(0, 1) >= 0.2:
        if i[3] == 1:
            train_pos_count += 1
        else:
            train_neg_count += 1
        train_set.append(i)
    else:
        if i[3] == 1:
            test_pos_count += 1
        else:
            test_neg_count += 1
        test_set.append(i)

print('train set length:%s' % len(train_set))
print('train_pos:%s, train_neg:%s' % (train_pos_count, train_neg_count))
print('test set length:%s' % len(test_set))
print('test_pos:%s, test_neg:%s' % (test_pos_count, test_neg_count))

random.shuffle(train_set)
random.shuffle(test_set)

movie_count = len(movie_info)
with open('dataset-small.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((genre_info, genre_count, movie_count), f, pickle.HIGHEST_PROTOCOL)

# 4
# train set length:78846
# train_pos:63796, train_neg:15050
# test set length:19549
# test_pos:15936, test_neg:3613

# 5
# train set length:78846
# train_pos:48139, train_neg:30707
# test set length:19549
# test_pos:12064, test_neg:7485

# 6
# train set length:71500
# train_pos:43491, train_neg:28009
# test set length:17745
# test_pos:10724, test_neg:7021