import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import random
import copy
from tqdm import tqdm
import os
import datetime
import pytz
from scipy.stats import entropy
from polyagamma import random_polyagamma


def make_now_dir(graph_path):
  now = datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H%M%S')
  gen_path = f'{graph_path}/{now}'
  os.makedirs(gen_path)

  return gen_path


def sigmoid(x, beta): #リスト

    d_matirx = np.stack([np.ones(len(x)), x],axis=1)
    input_var = np.dot(d_matirx, beta.T)

    return 1/(1+np.exp(-input_var))


def genarator_norm(X_mu, X_sigma, beta, sample_size):

    X_list = np.random.normal(loc = X_mu, scale = X_sigma , size = sample_size)

    p_list = sigmoid(X_list, beta)

    Y_list = np.array([])
    for i in range(sample_size):
        Y_list = np.append(Y_list, stats.bernoulli.rvs(p=p_list[i], size=1)[0])
    

    df = pd.DataFrame()
    df["X"] = X_list
    df["Y"] = Y_list

    return df


def sampler_cut(df,center,window):

    lower = center-window
    upper = center+window
    df_sample = df.loc[(df["X"]>lower) & (df["X"]<upper)]
    return df_sample


def sampler_proba(df, beta):

    Z_proba = sigmoid(df["X"], beta)
    Z = stats.bernoulli.rvs(p=Z_proba, size=len(df["X"]))
    df["Z"] = Z
    df_sample = df.loc[df["Z"] == 1]
    return df, df_sample


def KLD(p,q):

    slf_ent = -1 * (q * np.log(q) + (1 - q) * np.log(1 - q))
    crs_ent = -1 * (q * np.log(p) + (1 - q) * np.log(1 - p))

    return crs_ent - slf_ent


def sample_polya_gamma(df, prior_b0, prior_B0, burn, draw): #D^n(データフレーム)，Xの母平均(float),Xの母標準偏差(float)，βの事前平均(リスト)，βの事前分散共分散(行列)，バーンイン(int),希望サンプルサイズ(int)

    # Nの決定
    N = len(df)

    # xiとyiの決定
    X = np.stack([np.ones(len(df)), df["X"]],axis=1)
    Y = df["Y"].values

    # サンプリング格納用
    beta_strage = np.array([])

    # パラメータを初期化
    par_beta = np.zeros(2)
    par_w = np.zeros(2)
    par_eta = np.dot(X, par_beta)

    beta_strage = copy.deepcopy(par_beta)

    # ループを回す
    num = 1
    
    while num < burn+draw:

        # PGからwをサンプリング
        par_w = random_polyagamma(1, z = par_eta, size = (1, N))[0]

        # betaのサンプリング
        Z = (Y-0.5) / par_w
        B_input = np.dot(X.T, np.diag(par_w))
        B_input = np.dot(B_input, X)
        B = np.linalg.inv(B_input+np.linalg.inv(prior_B0))
        b_input_1 = np.dot(X.T, np.diag(par_w))
        b_input_1 = np.dot(b_input_1, Z)
        b_input_2 = np.dot(np.linalg.inv(prior_B0), prior_b0)
        b = np.dot(B, b_input_1+b_input_2)

        # パラメータの更新
        par_beta = np.random.multivariate_normal(b, B, size=1)[0]
        par_eta = np.dot(X, par_beta)

        # パラメータの保存
        beta_strage = np.vstack((beta_strage, par_beta))

        num += 1
  

    return beta_strage[burn:]


# In[ ]:




