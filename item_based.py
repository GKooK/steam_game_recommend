import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, accuracy
import os

class ItemSVD:
    def __init__(self):
        self.df = pd.read_csv('steam_games.csv', usecols=['url','types','name', 'all_reviews','developer', 'publisher', 'popular_tags'])
        self.df = self.df.drop([704, 35169])
        self.test =self.df['url'].apply(self.change_url)
        self.df['url'] = self.test.astype('int')
        self.test = self.df
        self.test['num_reviews'] = self.df['all_reviews'].apply(self.number_of_review)
        self.test['all_reviews']=self.df['all_reviews'].apply(self.remove_nan)
        self.test['publisher'] = self.test['publisher'].apply(self.remove_tab)
        need_to_remove = self.test[self.test['all_reviews'] == ''].index
        self.test = self.test.drop(need_to_remove)
        self.test_sort = self.test.sort_values('num_reviews', ascending=False)
        remove_things_list = self.test_sort[self.test_sort['num_reviews'] < 100].index
        self.test_sort = self.test_sort.drop(remove_things_list)
        self.mean_of_review_score = self.test_sort['all_reviews'].mean()
        self.test_sort['imbd_score'] = self.test_sort.apply(self.imdb_score, axis=1)
        self.test_sort = self.test_sort.sort_values('imbd_score', ascending=False)
        self.make_vector = self.test_sort
        self.make_vector['popular_tags'] = self.make_vector['popular_tags'].fillna('')
        self.make_vector['popular_tags'] = self.make_vector['popular_tags'].apply(lambda x:str(x).replace(' ', '').lower())
        self.temp_make_vector = self.make_vector
        self.tfidf_vectorizer = TfidfVectorizer()
        self.data_to_vector = self.tfidf_vectorizer.fit_transform(self.temp_make_vector['popular_tags'])
        self.cosine_sims = linear_kernel(self.data_to_vector, self.data_to_vector)
        self.make_vector.reset_index()
        self.make_vector_lower_name = self.make_vector.copy()
        self.make_vector_lower_name['name'] = self.make_vector_lower_name['name'].apply(lambda x:str(x).replace(' ','').lower())
        self.game_name = self.make_vector['name']
        self.index_game_name = pd.Series(self.make_vector.index, index=self.make_vector['name'])

    def get_tag_by_game_list(self, game_list):#게임 리스트를 들고와서 태그를 딕셔너리에 저장해서 센다.
        return_dict = dict()
        for i in game_list:
            game_tags = self.make_vector_lower_name[self.make_vector_lower_name['name']==(i.replace(' ','').lower())]['popular_tags']
            if(len(game_tags)==0):
                continue
            else:
                game_tags = game_tags.values[0].split(',')
            for j in game_tags:
                if( str(j) in return_dict ):
                    return_dict[str(j)] += 1
                else:
                    return_dict[str(j)] = 1
        return_dict = sorted(return_dict.items(), key=lambda x: x[1], reverse=True)
        return return_dict

    def recommend_by_tag_name(self, name, default_num=30):#imbd 점수로 비슷한 게임들 추천
        name = str(name).lower()
        try:
            return_val = self.make_vector[self.make_vector['popular_tags'].str.contains(name, na=False)]['name'].tolist()[1:1+default_num]
        except:
            return_val = []
        return return_val

    def recommend_by_game_name(self, name, default_num=30):#코사인 유사도로 비슷한 게임들 추천
        try:
            sim_scores = list(enumerate(self.cosine_sims[self.index_game_name[name]]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:1+default_num]
            games = [i[0] for i in sim_scores]
            return_val = self.game_name.iloc[games].tolist()
        except:
            return_val = []
        return return_val

    def imdb_score(self, val):
        v = val['num_reviews']
        r = val['all_reviews']
        return (v/(v+100)*r + 100/(100+v)*self.mean_of_review_score)

    def change_url(self, link):
      if('/bundle/' in link):
        return str(link).split('/bundle/')[1].split('/')[0]
      elif('/app/'in link):
        return str(link).split('/app/')[1].split('/')[0]
      elif('/sub/'in link):
        return str(link).split('/sub/')[1].split('/')[0]
      else:#curruent outlier is 704, 35169
        print('stranger : '+link)
    #해서 디버깅해본결과....types에 Other(240) 종류의 데이터들이 있었다.... 공백데이터를 삭제를 안한 것이다. 그래서 삭제해 주었다.

    def remove_nan(self, input_val):
      if((input_val != np.nan) and (str(input_val)!='nan')):
        if('%' in input_val):
          return_val = int(input_val.split('%')[0].split(' ')[-1])
          if((return_val == 100) or (return_val == 0)):
            return ''
          return return_val
        return ''
      else:
        return ''

    def remove_tab(self, val):
      return str(val).replace('\t', '')

    def number_of_review(self, val):
      if((val != np.nan) and (str(val)!='nan')):
        if('%' in val):
          return_val = int(val.split('%')[0].split(' ')[-1])
          if((return_val == 100) or (return_val == 0)):
            return ''
          return int(str(val).split(')')[0].split('(')[-1].replace(',', ''))
        return ''
      else:
        return ''

