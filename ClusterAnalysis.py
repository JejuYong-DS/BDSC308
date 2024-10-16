# -*- coding: utf-8 -*-
"""
Cluster Analysis

Team : 8조
Member : 고주용, 최주영

BDSC308 다차원자료분석PBL
"""
#%% 경로 지정
import os
link = r'C:\Users\rhwnd\OneDrive\바탕 화면\SelfLab\BDSC308\project\\'  # 개인 PC 환경에 맞추어 수정
os.chdir(link)

#%% module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.cluster import KMeans
# from sklearn_extra.robust import RobustWeightedKMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

#%% dpi 세팅 #해상도
#plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

#%% 한글 폰트 오류 해결
from matplotlib import font_manager, rc
font_path = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False #마이너스 부호 오류 해결

#%% 배경색
def back_ground_color(figsize=(8,6), bg_color = "#EFEDE3", box=True):
    fig = plt.figure(figsize=figsize)
    plt.box(box)
    
    fig.patch.set_facecolor(bg_color)
    fig.set_facecolor(bg_color)
#%% 데이터 호출
os.chdir(link)
df_region = pd.read_csv('df_REGION.csv', encoding = 'cp949')
df = pd.read_csv('df_X.csv', encoding='cp949')  #, index_col = 'ID'

df_X = df[['인구','건물', '생업', '학생', '도로']]
col_name = list(df_X.columns)

#%% group by region
# region = list(df_region[(df_region['SIDO_NM'] == '서울')
#                     |(df_region['SIDO_NM'] == '경기')
#                     |(df_region['SIDO_NM'] == '부산')
#                     |(df_region['SIDO_NM'] == '경북')
#                     |(df_region['SIDO_NM'] == '경남')
#                     ]['ID'])

# df_region_select = df[~df['ID'].isin(region)]

# df_X = df_region_select[['인구','건물', '생업', '학생', '도로']]
#%% Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_X = pd.DataFrame(scaler.fit_transform(df_X), columns = df_X.columns)

#%% 엘보우 플롯
os.chdir(link+'Clustering\\')
# WSS(within-cluster sum of square) 값을 저장할 리스트 초기화
WSS = []

# 1부터 25까지의 클러스터 개수에 대해 K-means 클러스터링 수행
for i in range(1, 25):
    random.seed(0)
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(df_X)
    WSS.append(kmeans.inertia_)

#%% 엘보우 그래프 그리기
i = 1 ; j = 25
back_ground_color()
plt.plot(range(i,j), WSS[i-1:j-1], marker='o', c='k')
# plt.vlines(3, 0,2.5e6, ls='--', color='r')
plt.xlabel('Number of clusters K')
plt.ylabel('Within-cluster Sum of Square')
plt.title('Elbow_Method')
# plt.savefig('Elbow_Method.jpg')
plt.show()

#%% KMEANS
K = 3

random.seed(0)
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_X)
df_X['KMEANS'] = kmeans.labels_
print(df_X['KMEANS'].value_counts())

df_vis = pd.concat([df[['ID', '인구', '건물', '생업', '학생', '도로']], df_X['KMEANS']],axis = 1)
cluster_dict = {0:'하위집단',1:'중위집단',2:'상위집단'}
#%%
# ID_list = df[df_X['KMEANS']==0]['ID']
# df = df[df_X['KMEANS']==0].reset_index(drop = True)
# df_X = df[['인구','건물', '생업', '학생', '도로']]

#%% clustering_visualization 함수
def clustering_visualization2D(df, i1, i2):
    back_ground_color()
    
    x = df[col_name[i1]]
    y = df[col_name[i2]]

    plt.scatter(x, y, c = df['KMEANS'], s= 5, cmap=cmap)
    for h, marker in enumerate(markers): #클러스터 데이터의 평균
        plt.scatter(kmeans.cluster_centers_[h, i1], kmeans.cluster_centers_[h, i2], 
                    c='k',
                    marker=marker,
                    s=150,
                    label=cluster_dict[h]
                    )

    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.xlabel(col_name[i1])
    plt.ylabel(col_name[i2])
    
    plt.legend(loc='upper right')
    title = f'{col_name[i1]}_{col_name[i2]}_2D'
    plt.title(title)
    # plt.savefig(f'{title}.jpg')
    plt.show()

def clustering_visualization3D(df, i1, i2,i3):
    fig = plt.figure(figsize=(8,6))
    plt.xticks([])
    plt.yticks([])

    plt.box(False)
    
    fig.patch.set_facecolor('#EFEDE3')
    fig.set_facecolor('#EFEDE3')
    
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_facecolor('#EFEDE3')
    
    x = df[col_name[i1]]
    y = df[col_name[i2]]
    z = df[col_name[i3]]
    
    ax.scatter(x, y, z, c = df['KMEANS'], s= 5, cmap=cmap, alpha = .3)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    
    ax.set_xlabel(col_name[i1])
    ax.set_ylabel(col_name[i2])
    ax.set_zlabel(col_name[i3])
    
    ax.zaxis.set_ticks_position('lower')

    title = f'{col_name[i1]}_{col_name[i2]}_{col_name[i3]}_3D'
    plt.title(title)
    # plt.savefig(f'{title}.jpg')
    plt.show()
    
#%% viz option
colors = ['#59ce9b', '#ffe33f', '#e57d77',]
cmap = mcolors.ListedColormap(colors)
# cmap = 'gist_rainbow'
markers = ['o','s','^']#,'d'] #,'^','d','h']#,'x','8','+']

# clustering_visualization2D(df_X, 0,1) #test

#%% visualization2D
for i1 in range(4):
    for i2 in range(i1+1,5):
        clustering_visualization2D(df_X, i1, i2)
        
#%% visualization3D    
for i1 in range(3):
    for i2 in range(i1+1,4):
        for i3 in range(i2+1,5):
            clustering_visualization3D(df_X,i1,i2,i3)
            
#%% 지역별 빈도분석
def clustering_visualization_Region(df, region = None):
    back_ground_color(box = True)

    count = df_vis[df_region['SIDO_NM']==region]['KMEANS'].value_counts().sort_index()
    
    if len(count) == 3:
        plt.bar(['하위집단','중위집단','상위집단'], count, color = colors)
    else:
        plt.bar(['하위집단','중위집단'], count, color = colors[:2])
    
    title = region
    plt.title(title)
    plt.savefig(f'{title}.jpg')
    plt.show()
    
    print(count)

#%% 지역별 빈도 분석
os.chdir(link+'지역별 군집\\')
for region in df_region['SIDO_NM'].unique():
    print(region,len(df_vis[df_region['SIDO_NM']==region]['KMEANS'].unique()))
    clustering_visualization_Region(df_vis, region)

#%% 지도
back_ground_color(box = True)
def world_map(df, color=None):
    plt.scatter(df['XCRDT'], df['YCRDT'],
                s = 1,
                c = color, 
                # cmap = 'gist_rainbow',
                marker = '.',
                alpha = 1,
                )
for i in range(K):
    world_map(df_region[df_X['KMEANS'] == i]
               , color = colors[i]
              )

    plt.xlim(df_region['XCRDT'].min(), df_region['XCRDT'].max())
    plt.ylim(df_region['YCRDT'].min(), df_region['YCRDT'].max())
# df_X.reset_index()
plt.title('world_map', fontsize = 15)
# plt.savefig('world_map.jpg')
plt.show()

#%% 각 지역별 그룹 value_counts()
# for region in df_region['SIDO_NM'].unique():
#     ID_temp = df_region[df_region['SIDO_NM'] == region]['ID']
#     df_temp = df_vis[df_vis['ID'].isin(ID_temp)]
#     print(region)
#     print(df_temp['KMEANS'].value_counts())

#%% recurrence 
def Elbow(df_temp):
    WSS = []
    
    # 1부터 25까지의 클러스터 개수에 대해 K-means 클러스터링 수행
    for i in range(1, 20):
        random.seed(0)
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(df_temp)
        WSS.append(kmeans.inertia_)
        print(i)
    
    i = 1 ; j = 20
    back_ground_color()
    plt.plot(range(i,j), WSS[i-1:j-1], marker='o', c='k')
    # plt.vlines(3, 0,2.5e6, ls='--', color='r')
    plt.xlabel('Number of clusters K')
    plt.ylabel('Within-cluster Sum of Square')
    plt.title('Elbow_Method')
    # plt.savefig('Elbow_Method.jpg')
    plt.show()

#%% 1회차
ID_1 = df[df_X['KMEANS'] == 0]['ID'].reset_index(drop=True)
df_temp_1 = df[df_X['KMEANS'] == 0][col_name].reset_index(drop=True)
df_region_1 = df_region[df_X['KMEANS'] == 0].reset_index(drop=True)
df_1 = pd.DataFrame(scaler.fit_transform(df_temp_1), columns = col_name)

Elbow(df_1)

K = 5
random.seed(0)
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_1)
df_1['KMEANS'] = kmeans.labels_
print(df_1['KMEANS'].value_counts())


cluster_dict = {k:str(f'{k+1}번째 군집') for k in range(K)}
colors = ['#59ce9b', '#ffe33f', '#e57d77','#60b0fb', '#8979f2']
cmap = mcolors.ListedColormap(colors)
markers = ['o','s','^','d','p']#,'h']#,'x','8','+']

# clustering_visualization2D(df_X, 0,1) #test

for i1 in range(4):
    for i2 in range(i1+1,5):
        clustering_visualization2D(df_1, i1, i2)

clustering_visualization2D(df_1,3,4)
clustering_visualization2D(df_1,2,0)
# clustering_visualization2D(df_1,2,0)
# clustering_visualization3D(df_1,3,2,0)

back_ground_color(box = True)
for i in range(5):
    if i in [0,2]:
        world_map(df_region_1[df_1['KMEANS'] == i]
               , color = colors[i]
              )

plt.xlim(df_region['XCRDT'].min(), df_region['XCRDT'].max())
plt.ylim(df_region['YCRDT'].min(), df_region['YCRDT'].max())
# df_X.reset_index()
plt.title('world_map', fontsize = 15)
plt.legend(['1번째 군집','3번째 군집'],loc='lower right')
# plt.savefig('world_map.jpg')
plt.show()


#%% 2회차
col_name = ['인구', '건물', '생업', '도로']
ID_2 = ID_1[(df_1['KMEANS'] == 0)|(df_1['KMEANS'] == 2)].reset_index(drop=True)
df_temp_2 = df_temp_1[(df_1['KMEANS'] == 0)|(df_1['KMEANS'] == 2)][col_name].reset_index(drop=True)
df_region_2 = df_region_1[(df_1['KMEANS'] == 0)|(df_1['KMEANS'] == 2)].reset_index(drop=True)
df_2 = pd.DataFrame(scaler.fit_transform(df_temp_2), columns = col_name)

Elbow(df_2)

K = 5
random.seed(0)
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_2)
df_2['KMEANS'] = kmeans.labels_
print(df_2['KMEANS'].value_counts())


cluster_dict = {k:str(f'{k+1}번째 군집') for k in range(K)}


colors = ['#59ce9b', '#ffe33f', '#e57d77','#60b0fb', '#8979f2']
cmap = mcolors.ListedColormap(colors)
markers = ['o','s','^','d','p']#,'h']#,'x','8','+']

# clustering_visualization2D(df_X, 0,1) #test

for i1 in range(3):
    for i2 in range(i1+1,4):
        clustering_visualization2D(df_2, i1, i2)

clustering_visualization2D(df_2,2,0)


back_ground_color(box = True)
for i in range(K):
    if i in [1]:
        world_map(df_region_2[df_2['KMEANS'] == i]
               , color = colors[i]
               )

    plt.xlim(df_region['XCRDT'].min(), df_region['XCRDT'].max())
    plt.ylim(df_region['YCRDT'].min(), df_region['YCRDT'].max())
# df_X.reset_index()
plt.title('world_map', fontsize = 15)
plt.legend(['2번째 군집'],loc='lower right')
# plt.savefig('world_map.jpg')
plt.show()

#%% 3회차
ID_3 = ID_2[(df_2['KMEANS'] == 1)].reset_index(drop=True)
df_temp_3 = df_temp_2[(df_2['KMEANS'] == 1)][col_name].reset_index(drop=True)
df_region_3 = df_region_2[(df_2['KMEANS'] == 1)].reset_index(drop=True)
df_3 = pd.DataFrame(scaler.fit_transform(df_temp_3), columns = col_name)

Elbow(df_3)

K = 3
random.seed(0)
kmeans = KMeans(n_clusters=K, random_state=0)
kmeans.fit(df_3)
df_3['KMEANS'] = kmeans.labels_
print(df_3['KMEANS'].value_counts())

cluster_dict = {k:str(f'{k+1}번째 군집') for k in range(K)}

colors = ['#59ce9b', '#ffe33f', '#e57d77']#,'#60b0fb']#, '#8979f2']
cmap = mcolors.ListedColormap(colors)
markers = ['o','s','^']#,'d']#,'p']#,'h']#,'x','8','+']


for i1 in range(3):
    for i2 in range(i1+1,4):
        clustering_visualization2D(df_3, i1, i2)
clustering_visualization3D(df_3, 1,2,3)
clustering_visualization3D(df_3, 1,3,2)
clustering_visualization3D(df_3, 2,1,3)
clustering_visualization3D(df_3, 2,3,1)
clustering_visualization3D(df_3, 3,1,2)
clustering_visualization3D(df_3, 3,2,1)

col_name
clustering_visualization2D(df_3,0,1)

back_ground_color(box = True)
for i in range(K):
    if i in [0,2]:
        world_map(df_region_3[df_3['KMEANS'] == i]
               , color = colors[i]
               )

    plt.xlim(df_region['XCRDT'].min(), df_region['XCRDT'].max())
    plt.ylim(df_region['YCRDT'].min(), df_region['YCRDT'].max())
# df_X.reset_index()
plt.title('world_map', fontsize = 15)
plt.legend(['1번째 군집','3번째 군집'],loc='lower right')
# plt.savefig('world_map.jpg')
plt.show()

#%% ID
os.chdir(link)

ID_4 = ID_3[(df_3['KMEANS'] == 0)|(df_3['KMEANS'] == 2)].reset_index(drop=True)
ID_4.to_csv('ID_4.csv')

#%% result
import VariableNames
region_list = VariableNames.region_list
df_sample = pd.read_csv('df_sampling.csv')
df_sample = df_sample[['ID', 'PNU_CD', 'SIDO_NM', 'SIGNGU_NM','LEGALDONG_NM', 'LEGALLI_NM', 'XCRDT','YCRDT']]

df_result = pd.merge(df_sample, ID_4, left_on='ID', right_on = 'ID')

df_result = df_result.sort_values('SIDO_NM')
df_result.sort_values('SIDO_NM').to_csv('df_result.csv', encoding='cp949', index=False)

for region in region_list:
    print(region, sum(df_result['SIDO_NM'] == region))

#%%
back_ground_color()
def world_map(df, color):
    plt.scatter(df['XCRDT'], df['YCRDT'],
                s = 1,
                c = color, 
                marker = '.',
                alpha = 1,
                )

colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','#FF0000','#00FF00','#0000FF','#00FFFF','#FF00FF','#FFFF00','#000000',]
for i,w in enumerate(region_list):
    world_map(df_result[df_result['SIDO_NM'] == w], color = colors[i])

plt.title('world_map', fontsize = 15)
plt.xlim(df_region['XCRDT'].min(), df_region['XCRDT'].max())
plt.ylim(df_region['YCRDT'].min(), df_region['YCRDT'].max())
# plt.savefig('world_map.jpg')
plt.legend(region_list,loc='lower right')
plt.show()
