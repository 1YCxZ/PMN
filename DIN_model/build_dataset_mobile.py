import random
import pickle


random.seed(1234)

# mobile:
HISTORY_WATCH = '../data/hist-mobile.pkl'
MOVIE_INFO = "../data/video-info.pkl"

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


#mobile:
for userID in hist:
    history_list = []
    u_his = hist[userID]
    for log in u_his:
        mID = log[0]
        score = log[1]
        if score < 0.1:
            continue
        history_list.append((mID, score))  # 历史观看和评分
        label = 1 if score >= 1 else 0
        if len(history_list) < 2:
            continue
        data_set.append((userID, history_list[-11:-1], mID, label))

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
with open('dataset-mobile.pkl', 'wb') as f:
    pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((genre_info, genre_count, movie_count), f, pickle.HIGHEST_PROTOCOL)