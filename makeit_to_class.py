import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import roc_curve, auc, average_precision_score
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import os

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

########Temp New User#######
new_user = np.random.rand(1,5064)
apply_lambda = lambda x: 1 if x>0.99 else 0
new_user = new_user.reshape(5064,)
tt = np.array([apply_lambda(xi) for xi in new_user])
tt = tt.reshape(1,5064)
make_random_time = lambda x: random.randint(10, 90) if x==1 else 0
tt_time = np.array([make_random_time(xi) for xi in tt.reshape(5064,)])
tt_time.reshape(5064,)
############################
class CollaborateFiltering:
    def __init__(self, k = 5):
        self.path = 'steam-200k-cleaned.csv'
        self.rec_games = 0#total_game_recoomand_variable
        self.k = k # 우리가 추출해서 눈으로 볼 데이터 수
        #self.train_matrix = 0
        #read_csv file from csv & names columns
        self.df = pd.read_csv(self.path, header = None,names = ['UserID', 'Game', 'Action', 'Hours', 'Other'])
        #Hours_Played column생성, 생성 후 action이 구매인 내역에는 플레이시간 0
        self.df['Hours_Played'] = self.df['Hours'].astype('float32')#tensorflow는 float32
        self.df.loc[(self.df['Action'] == 'purchase') & (self.df['Hours'] == 1.0), 'Hours_Played'] = 0
        self.df.UserID = self.df.UserID.astype('int')#USERID 는 Int형태로 변경
        #df id, 게임이름, 플레이시간 순서로 정렬
        self.df = self.df.sort_values(['UserID', 'Game', 'Hours_Played'])
        #keep last를 통해서 중복값 중 purchase를 삭제
        self.clean_df = self.df.drop_duplicates(['UserID', 'Game'], keep = 'last').drop(['Action', 'Hours', 'Other'], axis = 1)
        #고유한 유저수와 게임 수 구하기 유저수는 12388, 게임수는 5064
        self.n_users_origin = len(self.clean_df.UserID.unique())
        self.n_games_origin = len(self.clean_df.Game.unique())
        #self.sparsity_origin = (self.clean_df.shape[0]+tt.sum()) / float(self.n_users_origin * self.n_games_origin)

        self.user_counter_origin = Counter()
        for user in self.clean_df.UserID.tolist():
            self.user_counter_origin[user] +=1

        self.game_counter_origin = Counter()
        for game in self.clean_df.Game.tolist():
            self.game_counter_origin[game] += 1

        #각각 UserID:clean_df의 index, clean_df의 index:UserID의 형태
        self.user2idx_origin = {user: i for i, user in enumerate(self.clean_df.UserID.unique())}
        self.idx2user_origin = {i: user for user, i in self.user2idx_origin.items()}
        #각각 Game:clean_df의 index, clean_df의 index:Game의 형태
        self.game2idx_origin = {game: i for i, game in enumerate(self.clean_df.Game.unique())}
        self.idx2game_origin = {i: game for game, i in self.game2idx_origin.items()}

        #clean_df의 UserID 행을 다 index로 바꾸었을때의 값들 5250은 첫번째로 등장하는 유저아이디라서 0이다.
        self.user_idx_origin = self.clean_df['UserID'].apply(lambda x: self.user2idx_origin[x]).values
        self.game_idx_origin = self.clean_df['gameIdx'] = self.clean_df['Game'].apply(lambda x: self.game2idx_origin[x]).values
        self.hours_origin = self.clean_df['Hours_Played'].values

        self.zero_matrix_origin = np.zeros(shape = (self.n_users_origin, self.n_games_origin)) # Create a zero matrix(유저수x게임수)
        #Preference matrix (유저가 게임을 보유했는지, 보유하지 않았는지)
        self.user_game_pref_origin = self.zero_matrix_origin.copy()
        self.user_game_pref_origin[self.user_idx_origin, self.game_idx_origin] = 1 #유저가 게임을 보유한 부분에 1의 값을 넣는다.
        # Confidence matrix #유저가 게임을 얼마나 즐겼는지
        self.user_game_interactions_origin = self.zero_matrix_origin.copy()
        self.user_game_interactions_origin[self.user_idx_origin, self.game_idx_origin] = self.hours_origin + 1

    #tt = tt.reshape(1,5064)
    def add_new_user(self, input_val):
        input_val = input_val.reshape(5064,)
        user_purchase = np.zeros(input_val.shape)
        user_purchase[np.where(input_val>0)[0]] = 1
        user_purchase = user_purchase.reshape(1,5064)

        self.n_users = self.n_users_origin + 1
        self.n_games = self.n_games_origin
        self.sparsity_origin = (self.clean_df.shape[0]+user_purchase.sum()) / float(self.n_users * self.n_games)
        self.sparsity = self.sparsity_origin.copy()

        self.user_counter = self.user_counter_origin.copy()
        self.game_counter = self.game_counter_origin.copy()

        self.user_counter[0] = user_purchase.sum()#새로운 유저의 UserID는 0으로 한다.
        for i, (key, val) in enumerate(self.game_counter.items()):
            self.game_counter[key] += user_purchase[0][i]
        
        self.user2idx = self.user2idx_origin.copy()
        self.idx2user = self.idx2user_origin.copy()
        self.game2idx = self.game2idx_origin.copy()
        self.idx2game = self.idx2game_origin.copy()

        self.user_idx = self.user_idx_origin.copy()
        self.game_idx = self.game_idx_origin.copy()
        self.hours  = self.hours_origin.copy()

        self.user2idx[0] = len(self.user2idx)
        self.idx2user[len(self.idx2user)] = 0
        for i in range(int(user_purchase.sum())):
            self.user_idx = np.append(self.user_idx, self.user2idx[0])

        self.game_idx = np.append(self.game_idx, np.where(user_purchase[0] == 1)[0])
        self.hours = np.append(self.hours, input_val[np.where(user_purchase[0] == 1)[0]])

        self.zero_matrix = np.zeros(shape = (self.n_users, self.n_games)) # Create a zero matrix(유저수x게임수)
        #Preference matrix (유저가 게임을 보유했는지, 보유하지 않았는지)
        self.user_game_pref = self.zero_matrix.copy()
        self.user_game_pref[self.user_idx, self.game_idx] = 1
        # Confidence matrix #유저가 게임을 얼마나 즐겼는지
        self.user_game_interactions = self.zero_matrix.copy()
        self.user_game_interactions[self.user_idx, self.game_idx] = self.hours + 1

    def recommend_sim_user(self):
        temp_cos_sim = cosine_similarity(self.user_feature, self.user_feature)
        cos_enum = list(enumerate(temp_cos_sim[-1]))
        sim_scores = sorted(cos_enum, key=lambda x: x[1], reverse=True)
        sim_user_list = []
        for test, i in sim_scores[1:11]:
            print('##############################################')
            purchase_history = np.where(self.user_idx==int(test))[0]
            user_game_list = []
            for j in self.game_idx[purchase_history]:
                user_game_list.append(self.idx2game[j])
            sim_user_list.append([test, user_game_list])
            print('User'+str(test)+' purchases')
            print(', '.join([self.idx2game[game] for game in self.game_idx[purchase_history]]))
        return sim_user_list

    def eval_result(self):
        print('##############################################')
        self.rec_games = np.argsort(-self.rec)
        print('User #{0} recommendations ...'.format(self.idx2user[self.user2idx[0]]))
        #print(user)
        #purchase_history = np.where(train_matrix[user2idx[0], :] != 0)[0]
        purchase_history = np.where(self.user_idx==self.user2idx[0])[0]
        recommendations = self.rec_games[self.user2idx[0], :]
        new_recommendations = recommendations[~np.in1d(recommendations, purchase_history)][:self.k]    
        print('Recommendations')
        recommend_list = []
        for i in new_recommendations:
            recommend_list.append(self.idx2game[i])
        print(', '.join([self.idx2game[game] for game in new_recommendations]))
        print('Actual purchases')
        print(', '.join([self.idx2game[game] for game in self.game_idx[purchase_history]]))
        print('Precision of {0}'.format(len(set(new_recommendations) & set(np.where(self.test_matrix[self.user2idx[0], :] != 0)[0])) / float(self.k)))
        print('--------------------------------------')
        return recommend_list

    def train_data(self):
    # maybe defined in __init__(self, k = 5) k = 5#The Number of Games That will Recommend to User
    # Count the number of purchases for each user
        purchase_counts = np.apply_along_axis(np.bincount, 1, self.user_game_pref.astype(int))
        buyers_idx = np.where(purchase_counts[:, 1] >= 2 * self.k)[0] #find the users who purchase 2 * k games
        print('{0} users bought {1} or more games'.format(len(buyers_idx), 2 * self.k))
        # trainset:validationset:testset = 8:1:1
        test_frac = 0.2 
        test_users_idx = np.random.choice(buyers_idx,size = int(np.ceil(len(buyers_idx) * test_frac)),replace = False)
        val_users_idx = test_users_idx[:int(len(test_users_idx) // 2)]
        test_users_idx = test_users_idx[int(len(test_users_idx) // 2):]

        def data_process(data, train, test, user_idx, k):
            for user in user_idx:
                purchases = np.where(data[user, :] == 1)[0]
                mask = np.random.choice(purchases, size=k, replace = False)        
                train[user, mask] = 0
                test[user, mask] = data[user, mask]
            return train, test

        self.train_matrix = self.user_game_pref.copy()
        self.test_matrix = self.zero_matrix.copy()
        self.val_matrix = self.zero_matrix.copy()
        # Mask the train matrix and create the validation and test matrices
        self.train_matrix, self.val_matrix = data_process(self.user_game_pref, self.train_matrix, self.val_matrix, val_users_idx, self.k)
        self.train_matrix, self.test_matrix = data_process(self.user_game_pref, self.train_matrix, self.test_matrix,test_users_idx, self.k)

        self.train_matrix[test_users_idx[0], self.test_matrix[test_users_idx[0], :].nonzero()[0]]
        self.test_matrix[test_users_idx[0], self.test_matrix[test_users_idx[0], :].nonzero()[0]]

        tf.reset_default_graph() # Create a new graphs

        pref = tf.placeholder(tf.float32, (self.n_users, self.n_games))  # Here's the preference matrix
        interactions = tf.placeholder(tf.float32, (self.n_users, self.n_games)) # Here's the hours played matrix
        users_idx = tf.placeholder(tf.int32, (None))

        n_features = 30 # 추출되는 피쳐 수
        # The X matrix represents the user latent preferences with a shape of user x latent features
        X = tf.Variable(tf.truncated_normal([self.n_users, n_features], mean = 0, stddev = 0.05))
        # The Y matrix represents the game latent features with a shape of game x latent features
        Y = tf.Variable(tf.truncated_normal([self.n_games, n_features], mean = 0, stddev = 0.05))
        # Here's the initilization of the confidence parameter
        conf_alpha = tf.Variable(tf.random_uniform([1], 0, 1))

        #user bias
        user_bias = tf.Variable(tf.truncated_normal([self.n_users, 1], stddev = 0.2))
        # Concatenate the vector to the user matrix
        X_plus_bias = tf.concat([X, user_bias,tf.ones((self.n_users, 1), dtype = tf.float32)], axis = 1)
        # game bias
        item_bias = tf.Variable(tf.truncated_normal([self.n_games, 1], stddev = 0.2))
        # Cocatenate the vector to the game matrix
        Y_plus_bias = tf.concat([Y,tf.ones((self.n_games, 1), dtype = tf.float32),item_bias],axis = 1)
        pred_pref = tf.matmul(X_plus_bias, Y_plus_bias, transpose_b=True)

        # Construct the confidence matrix with the hours played and alpha paramter
        conf = 1 + conf_alpha * interactions
        cost = tf.reduce_sum(tf.multiply(conf, tf.square(tf.subtract(pref, pred_pref))))
        l2_sqr = tf.nn.l2_loss(X) + tf.nn.l2_loss(Y) + tf.nn.l2_loss(user_bias) + tf.nn.l2_loss(item_bias)
        lambda_c = 0.01
        loss = cost + lambda_c * l2_sqr
        lr = 0.05
        optimize = tf.train.AdagradOptimizer(learning_rate = lr).minimize(loss)

        def top_k_precision(pred, mat, k, user_idx):
            precisions = []
            for user in user_idx:
                rec = np.argsort(-pred[user, :]) # Found the top recommendation from the predictions        
                top_k = rec[:k]
                labels = mat[user, :].nonzero()[0]
                precision = len(set(top_k) & set(labels)) / float(k) # Calculate the precisions from actual labels
                precisions.append(precision)
            return np.mean(precisions) 

        iterations = 100
        #print(self.test_users_idx)
        #print(self.test_matrix.shape)
        stop_signal = 0
        while(stop_signal == 0):
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            for i in range(1,iterations):
                sess.run(optimize, feed_dict = {pref: self.train_matrix, interactions: self.user_game_interactions})
                if((i % 10 == 0 )or(i==1)):
                    mod_loss = sess.run(loss, feed_dict = {pref: self.train_matrix, interactions: self.user_game_interactions})
                    if(mod_loss < 0):
                        break
                    mod_pred = pred_pref.eval(session=sess)
                    train_precision = top_k_precision(mod_pred, self.train_matrix, self.k, val_users_idx)
                    val_precision = top_k_precision(mod_pred, self.val_matrix, self.k, val_users_idx)
                    print('Iterations {0}...'.format(i),
                          'Training Loss {:.2f}...'.format(mod_loss),
                          'Train Precision {:.3f}...'.format(train_precision),
                          'Val Precision {:.3f}'.format(val_precision)
                        )
                if(i==99):
                    print('i is 99')
                    mod_loss = sess.run(loss, feed_dict = {pref: self.train_matrix,interactions: self.user_game_interactions})
                    if(mod_loss > 0):
                        stop_signal = 1

        self.rec = pred_pref.eval(session=sess)
        self.user_feature = X_plus_bias.eval(session=sess)
        test_precision = top_k_precision(self.rec, self.test_matrix, self.k, test_users_idx)
        print('\n')
        print('Test Precision{:.3f}'.format(test_precision))