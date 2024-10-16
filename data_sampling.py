# -*- coding: utf-8 -*-
"""
Data Sampling

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

import VariableNames #my module

#%% seed 고정
random.seed(0)

#%% dpi 세팅 #해상도
#plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")

#%%
#한글 폰트 오류 해결
from matplotlib import font_manager, rc
font_path = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False #마이너스 부호 오류 해결

#%% 변수명 자료구조 호출
col_name_char = VariableNames.col_name_char
col_name_kor_dict = VariableNames.col_name_kor_dict
region_list = VariableNames.region_list

#%% 배경색
def back_ground_color(figsize=(8,6), bg_color = "#EFEDE3"):
    fig = plt.figure(figsize=figsize)
    plt.box(False)
    
    fig.patch.set_facecolor(bg_color)
    fig.set_facecolor(bg_color)

#%% quote processing
def data_read(s):
    # 문자열에서 " 전처리
    data = []
    while True:
        i = s.find('"')  # " 발견
        j = s[i+1:].find('"')  # 다음 " 발견

        if i == -1:
            data.extend(s.split())
            break

        if s[:i].split():  # " 발견 이전의 데이터 추출
            data.extend(s[:i].split())

        data.append(s[i+1:i+j+1])  # " 사이의 문자만 추출
        s = s[i+j+2:]

    return data

#%% data load
data = []
with open("샘플데이터.txt", encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(i)  # 현재 상황 파악용 코드
        if i != 0:
            data.append(data_read(line))
        else:  # header
            col_name = line.replace('"', '').split()
            col_name.insert(0, 'ID')

#%% data frame
df = pd.DataFrame(data, columns=col_name)

#numerical data가 object로 되어 있는 현상 수정

for col in col_name:
    if not col in col_name_char:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print(col)

#Unique()가 1인 변수 제거
for col in col_name:
    if len(df[col].unique()) == 1:
        print(col)
        df = df.drop(col, axis = 1)

#%% 전처리
df['SIGNGU_NM'] = df['SIGNGU_NM'].replace('', '세종시') #결측치 처리
#나머지 문자형 변수는 안쓸거라 패스

#결측치 확인
print(df.isnull().sum())

#결측치 대체
df.fillna(df.max(), inplace = True)

#%% world map 전체
df_world_map = df[['ID', 'SIDO_NM', "XCRDT","YCRDT"]]
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
    world_map(df_world_map[df_world_map['SIDO_NM'] == w], color = colors[i])

plt.title('world_map', fontsize = 15)
# plt.savefig('world_map.jpg')
plt.show()

#%% world map 개별
for i,w in enumerate(region_list):
    back_ground_color()
    world_map(df_world_map[df_world_map['SIDO_NM'] == w], color = colors[i])
    plt.title(w)
    # plt.savefig(w+'.jpg')
    plt.show()

#%%outlier by world map
outlier_list = []

df_temp = df[df['SIDO_NM']=='대구'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 420000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='인천'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] < 400000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='광주'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 300000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='경기'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 650000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='강원'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 700000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='충북'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 530000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='충남'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 525000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='전북'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['YCRDT'] > 425000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='경북'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['XCRDT'] < 550000)&(df_temp['YCRDT'] > 510000)]
outlier_list.extend(list(outlier['ID']))

df_temp = df[df['SIDO_NM']=='경남'][['ID',"XCRDT","YCRDT"]]
outlier = df_temp[(df_temp['XCRDT'] <= 420000)&(df_temp['YCRDT'] > 390000)]
outlier_list.extend(list(outlier['ID']))
outlier = df_temp[(df_temp['XCRDT'] > 420000)&(df_temp['YCRDT'] > 350000)]
outlier_list.extend(list(outlier['ID']))
outlier_list.sort()

#문자로 변환
outlier_list = list(map(str, outlier_list))

#아웃라이어 데이터프레임
df_outlier = df[df['ID'].isin(outlier_list)][['ID', 'SIDO_NM', 'SIGNGU_NM', 'LEGALDONG_NM','LEGALLI_NM']]
df_outlier.sort_values('SIDO_NM', inplace=True)
df_outlier.to_csv('df_outlier.csv')

#아웃라이어 제거
df = df[~df['ID'].isin(outlier_list)]

#원본 데이터프레임 저장
df.to_csv('df.csv', encoding = 'cp949')

#%% Samplings
#다단계층화계통추출
#계통추출법
def systematic_sampling(data, n):
    start = random.randint(0, n-1)
    sampled = [data[i] for i in range(start, len(data), n)]
    
    return sampled

#다단계 층화추출
df_sampling = df[['ID','SIDO_NM','SIGNGU_NM']]
sampling_list = []

for w in region_list:
    df_temp = df_sampling[df_sampling['SIDO_NM'] == w]
    for signgu in list(df_temp['SIGNGU_NM'].unique()):
        data = list(df_temp[df_temp['SIGNGU_NM'] == signgu]['ID'])
        sampling_list.extend(systematic_sampling(data, 5))
        # print(w,signgu)

df = df[df['ID'].isin(sampling_list)]

df.to_csv('df_sampling.csv', encoding = 'cp949')
