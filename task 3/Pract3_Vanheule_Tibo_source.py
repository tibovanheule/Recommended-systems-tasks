#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__author__ = "Tibo Vanheule"
@author = "Tibo Vanheule"


# In[ ]:


get_ipython().system(' pip install pandas numpy joblib')


# In[ ]:


import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from pathlib import Path
from functools import lru_cache

logging.getLogger().setLevel(logging.INFO)

def read_dat_files():
    logging.info("reading")
    R = pd.read_csv("ratings.csv",names=["userID","movieID","rating","timestamp"],skiprows=1)
    I = pd.read_csv("movies.csv",names=["movieID","title","genre"],skiprows=1)
    U = pd.read_csv("tags.csv",skiprows=1)
    logging.info("Done reading")
    return R, I, U 

R, I, U = read_dat_files()


# In[ ]:


@lru_cache()
def pearson_correlation(a: int,b: int, cut_off=10, significance_weigthing=False) -> float:
    r_a = R[R.userID == a]
    r_b = R[R.userID == b]
    intersection = pd.merge(r_a, r_b, how ='inner', on =['movieID'])
    logging.debug(intersection)
    if len(intersection) <=1:
        return 0
    r_a_mean = np.mean(r_a.rating)
    r_u_mean = np.mean(r_b.rating)
    teller = np.sum([(d.rating_x-r_a_mean)*(d.rating_y-r_u_mean) for i,d in intersection.iterrows()])
    noemer1 = np.sum([(d.rating_x-r_a_mean)**2 for i,d in intersection.iterrows()])
    noemer2 = np.sum([(d.rating_y-r_u_mean)**2 for i,d in intersection.iterrows()])
    noemer = np.sqrt(noemer1)*np.sqrt(noemer2)
    if noemer == 0:
        return 0
    
    retval = teller/noemer
    if significance_weigthing:
        return retval*(np.min([cut_off,len(intersection)])/cut_off)
    return retval


[pearson_correlation(1,4), pearson_correlation(1,4,significance_weigthing=True),pearson_correlation(4,5), pearson_correlation(4,7,significance_weigthing=True)]


# In[ ]:


def get_users_who_rated_movie(movie_id: int, return_df=False) -> list:
    ratings_movie = R[R.movieID == movie_id]
    if return_df:
        return ratings_movie
    return list(set(ratings_movie.userID))

len(get_users_who_rated_movie(10))


# In[ ]:


@lru_cache()
def get_top_k_neigbours_who_rated_movie(user_id: int, movie_id:int, only_positive=True, k=20) -> list:
    users = get_users_who_rated_movie(movie_id)
    retval = [(u,pearson_correlation(user_id,u,significance_weigthing=True)) for u in users]
    retval = sorted(retval, key=lambda x : (-x[-1],x[0]))[:k]
    if only_positive:
        return [i for i in retval if i[1] > 0]
    return retval
len(get_top_k_neigbours_who_rated_movie(1,10,k=130, only_positive=True))


# In[ ]:


get_top_k_neigbours_who_rated_movie(1,10,k=20, only_positive=True)


# In[ ]:


def deviation(users,target):
    teller = []
    noemer = []
    for i in users:
        user_df = R[R.userID==i[0]]
        mean_user = np.mean(user_df.rating)
        noemer.append(i[1])
        ru_i=list(user_df[user_df.movieID==target].rating)[0]
        teller.append((ru_i-mean_user)*i[1])
        
    return np.sum(teller)/np.sum(noemer)


# In[ ]:


@lru_cache()
def recommendation_uucf(a:int, item:int, return_dev=False)-> float:
    r_a = R[R.userID == a]
    users = get_top_k_neigbours_who_rated_movie(a,item,k=20,only_positive=True)
    mean_a = np.mean(r_a.rating)
    if return_dev:
        return deviation(users,item)
    if len(users)>0:
        return mean_a + deviation(users,item)
    return mean_a


# In[ ]:


recommendation_uucf(1,10,return_dev=True)


# In[ ]:


recommendation_uucf(1,10)


# In[ ]:


print(I[I.movieID==10].to_latex())


# In[ ]:


len(get_top_k_neigbours_who_rated_movie(1,260,k=500, only_positive=True))


# In[ ]:


get_top_k_neigbours_who_rated_movie(1,260,k=20, only_positive=True)


# In[ ]:


recommendation_uucf(1,260,return_dev=True)


# In[ ]:


recommendation_uucf(1,260)


# In[ ]:


print(I[I.movieID==260].to_latex())


# In[ ]:


def get_top_n_recommendations(user:int)->list():
    retval= [(i,recommendation_uucf(user,i),) for i in set(I.movieID)]
    return sorted(retval, key=lambda x : -x[1])


# In[ ]:


get_top_n_recommendations(1)


# In[ ]:


top = [(3216, 4.936834319526627),
 (40412, 4.936834319526627),
 (92494, 4.936834319526627),
 (3320, 4.928235294117647),
 (4302, 4.928235294117647),
 (4731, 4.928235294117647),
 (5071, 4.928235294117647),
 (86781, 4.928235294117647),
 (97957, 4.717883488583054),
 (3414, 4.543621197252207),]
df = pd.DataFrame({"movieID":[i for i,v in top],"rating prediction":[i for i,v in top]})
df = pd.merge(df, I, how ='inner', on =['movieID'])
logging.info(df.to_latex())


# In[ ]:


get_top_n_recommendations(522)


# In[ ]:


top = [(565, 6.180285714285715),
 (1450, 6.180285714285715),
 (1563, 6.180285714285715),
 (1819, 6.180285714285715),
 (4076, 6.180285714285715),
 (4591, 6.180285714285715),
 (4796, 6.180285714285715),
 (4930, 6.180285714285715),
 (5427, 6.180285714285715),
 (3216, 5.552834319526627),
 (92494, 5.552834319526627),
 (4302, 5.544235294117647),
 (4731, 5.544235294117647),
 (5071, 5.544235294117647),
 (40412, 5.512176190620671),
 (3879, 5.411901639344262),
 (3437, 5.318977412731006),
 (5765, 5.318977412731006),
 (132333, 5.271763688760807)][:10]
df = pd.DataFrame({"movieID":[i for i,v in top],"rating prediction":[i for i,v in top]})
df = pd.merge(df, I, how ='inner', on =['movieID'])
logging.info(df.to_latex())


# In[ ]:


def cosine_sim(i:int,i2:int) -> float:
    users_i1 = get_users_who_rated_movie(i, return_df=True)
    users_i2 = get_users_who_rated_movie(i2, return_df=True)
    
    intersection = pd.merge(users_i1, users_i2, how ='inner', on =['userID'])
    
    mean_i = np.mean(users_i1.rating)
    mean_i2 = np.mean(users_i2.rating)
    
    noemer_1 = np.sqrt(np.sum([(d.rating - mean_i)**2 for _,d in users_i1.iterrows()]))
    noemer_2 = np.sqrt(np.sum([(d.rating - mean_i2)**2 for _,d in users_i2.iterrows()]))
    noemer = noemer_1*noemer_2
    teller = np.sum([(d.rating_x - mean_i)*(d.rating_y - mean_i2) for _,d in intersection.iterrows()])

    if noemer == 0:
        return 0
    return teller/noemer


# In[ ]:


cosine_sim(25,129659)


# In[ ]:


cosine_sim(596,594)


# In[ ]:


import time
def build_model():
    df = pd.DataFrame({"from":[],"to":[],"cosine":[]})
    for index,i in enumerate(list(I.movieID)):
        print(index/9125)
        for i2 in list(I.movieID):
            if i == i2:
                continue
            df = df.append(pd.DataFrame({"from":[i],"to":[i2],"cosine":[cosine_sim(i,i2)]}))
    df.to_csv(str(time.time())+".csv")
    return df
    


# In[ ]:


build_model()


# In[ ]:


def get_from_model_top_and_rated_by_user(i:int,target_user_id:int,n=10)->pd.DataFrame:
    df = pd.DataFrame({"from":[],"to":[],"cosine":[]})
    intersection = R[R.userID == target_user_id]
    for i2 in set(intersection.movieID):
        if i != i2:
            df = df.append(pd.DataFrame({"from":[int(i)],"to":[int(i2)],"cosine":[cosine_sim(i,i2)]}))
    df = df[df.cosine > 0].sort_values("cosine", ascending=[0])
    df = df.head(n=n)
    df.to_csv(str(time.time())+".csv")
    return df


# In[ ]:


data = get_from_model_top_and_rated_by_user(25,522,n=500)
logging.info(len(data))


# In[ ]:


if "to" in data.columns:
    data["movieID"] = data.to
    del data["from"], data["to"]
logging.info(pd.merge(top, I, how="inner", on="movieID").head(n=10).to_latex())


# In[ ]:


def _prediction_rating(target_item,target_user_id):

    user_ratings = R[R.userID==target_user_id]
    similar_items = get_from_model_top_and_rated_by_user(target_item,target_user_id,n=20)
    
    noemer = np.sum(np.abs([d.cosine for _, d in similar_items.iterrows()]))
    if noemer == 0:
        return (target_item,0)
    teller = np.sum([d.cosine*(list(user_ratings[user_ratings.movieID==d.to].rating)[0]) for _, d in similar_items.iterrows()])

    return (target_item,teller/noemer)


# In[ ]:


from joblib import Parallel, delayed
def prediction_rating(target_user_id:int) -> pd.DataFrame:
    df = pd.DataFrame({"movieID":[],"rating prediction":[]})
    
    movies= set(I.movieID)
    results = Parallel(n_jobs=20)(delayed(_prediction_rating)(i,target_user_id) for i in movies)
    
    for i,v in results:
        df = df.append(pd.DataFrame({"movieID":[i],"rating prediction":[v]}))
    return df


# In[ ]:


df = prediction_rating(522)


# In[ ]:


logging.info(prediction_rating(522).to_latex())


# # basket recommendations

# In[ ]:


def _prediction_rating_basket(target_item,basket):
    retval= []
    for basket_i in basket:
        sim = cosine_sim(target_item,basket_i)
        if sim > 0:
            retval.append(sim)
    retval = np.sum(retval) 
    return (target_item,retval)


# In[ ]:


def basket(basket:list)->pd.DataFrame:
    df = pd.DataFrame({"movieID":[],"rating prediction":[]})
    
    movies= set(I.movieID)
    results = Parallel(n_jobs=10)(delayed(_prediction_rating_basket)(i,basket) for i in movies)
    
    for i,v in results:
        df = df.append(pd.DataFrame({"movieID":[i],"rating prediction":[v]}))
    return pd.merge(df.sort_values("rating prediction", ascending=[0]),I, how="inner",on="movieID")
logging.info(basket([1]).to_latex())
logging.info(I[I.movieID==1].to_latex())


# In[ ]:


logging.info(basket([1,48,239]).to_latex())
logging.info(I[(I.movieID==1) | (I.movieID==48) | (I.movieID==239)].to_latex())


# In[ ]:


def _prediction_rating_basket(target_item,basket):
    retval= []
    for basket_i in basket:
        sim = cosine_sim(target_item,basket_i)
        retval.append(sim)
    retval = np.sum(retval) 
    return (target_item,retval)


# In[ ]:


logging.info(basket([1,48,239]).to_latex())


# # Hybrid 

# In[ ]:


def get_top_n_recommendations(user:int)->list():
    movies = set(I.movieID)
    retval = [(i,(recommendation_uucf(user,i)/2)+(_prediction_rating(i,user)[1]/2),) for i in movies]
    return sorted(retval, key=lambda x : -x[1])


# In[ ]:


result = get_top_n_recommendations(522)


# In[ ]:


df = pd.DataFrame({"score":[i[1] for i in result],"movieID":[i[0] for i in result]})
logging.info(pd.merge(df,I,how="inner").to_latex())
    


# In[ ]:




