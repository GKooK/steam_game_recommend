# steam_game_recommend
> data set Used in Kaggle<br>
> https://www.kaggle.com/jwyang91/steam-games-2017-cleaned
> https://www.kaggle.com/trolukovich/steam-games-complete-dataset 
## Usage Example
```python
from makeit_to_class import CollaborateFiltering
from item_based import ItemSVD
import numpy as np
import random

##########Make New User##############################
#new_user = np.random.rand(1,5064)
#apply_lambda = lambda x: 1 if x>0.99 else 0
#new_user = new_user.reshape(5064,)
#tt = np.array([apply_lambda(xi) for xi in new_user])
#tt = tt.reshape(1,5064)
#make_random_time = lambda x: random.randint(10, 90) if x==1 else 0
#tt_time = np.array([make_random_time(xi) for xi in tt.reshape(5064,)])
#tt_time.reshape(5064,)
######################################################

#선언부
temp = CollaborateFiltering(k=30)
itemsvd = ItemSVD()

temp.add_new_user(tt_time)
temp.train_data()
result_reco = temp.eval_result()#요거
result_sim_user = temp.recommend_sim_user()#요것도

result_reco_tag_dict = itemsvd.get_tag_by_game_list(result_reco)
tag1 = itemsvd.recommend_by_tag_name(result_reco_tag_dict[0][0])
tag2 = itemsvd.recommend_by_tag_name(result_reco_tag_dict[1][0])

#기존 유저 디비에서 꺼내올때
#test = 디비에서 가져온 유저의 추천 게임 리스트
#tag_dict = itemsvd.get_tag_by_game_list(test)
#tag1 = itemsvd.recommend_by_tag_name(tag_dict[0][0])
#tag2 = itemsvd.recommend_by_tag_name(tag_dict[1][0])
```
