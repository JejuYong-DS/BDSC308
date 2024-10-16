# -*- coding: utf-8 -*-
"""
Data Preprocessing

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

#%% 데이터 호출
df = pd.read_csv('df_sampling.csv')
df = df[df.columns[1:]] #csv 변환 과정에서 발생한 문제. 해결 코드

# %% Scoring
col_SHRTDST = VariableNames.col_SHRTDST

from sklearn.preprocessing import MinMaxScaler
def Scoring(data):
    scaler = MinMaxScaler()
    return 1-scaler.fit_transform(data)
   
for col in col_SHRTDST:
    df[col] = Scoring(df[col])

#%% 데이터프레임 분할 by 카테고리
df_REGION = df[['ID',
                'SIDO_NM', 'SIGNGU_NM', 'LEGALDONG_NM',
                "XCRDT","YCRDT"]]
# df_REGION.to_csv('df_REGION.csv', encoding = 'cp949')

df_POP = df[['ID',
             'TOTPUL_CNT_FVHD_METER', 'TOTPUL_CNT_ONE_KM', 'TOTPUL_CNT_THREE_KM', 'TOTPUL_CNT_FIVE_KM', 
             'SNCTPUL_CNT_FVHD_METER', 'SNCTPUL_CNT_ONE_KM', 'SNCTPUL_CNT_THREE_KM', 'SNCTPUL_CNT_FIVE_KM', 
             'YUPUL_CNT_FVHD_METER', 'YUPUL_CNT_ONE_KM', 'YUPUL_CNT_THREE_KM', 'YUPUL_CNT_FIVE_KM', 
             'WKAGEPUL_CNT_FVHD_METER', 'WKAGEPUL_CNT_ONE_KM','WKAGEPUL_CNT_THREE_KM', 'WKAGEPUL_CNT_FIVE_KM',
             'TOTHOHLD_CNT_FVHD_METER', 'TOTHOHLD_CNT_ONE_KM', 'TOTHOHLD_CNT_THREE_KM', 'TOTHOHLD_CNT_FIVE_KM'
             ]]
# df_POP.to_csv('df_POP.csv', encoding = 'cp949')

df_WORK = df[["ID",
              'TOTBS_CNT_FVHD_METER', 'TOTBS_CNT_ONE_KM', 'TOTBS_CNT_THREE_KM', 'TOTBS_CNT_FIVE_KM',
              'TOTENFSN_CNT_FVHD_METER', 'TOTENFSN_CNT_ONE_KM', 'TOTENFSN_CNT_THREE_KM', 'TOTENFSN_CNT_FIVE_KM',
              'BSNS_CNT_WHRTBS_THREE_KM', 'BSNS_CNT_WHRTBS_FIVE_KM',
              'BSNS_CNT_RSTLDBS_THREE_KM', 'BSNS_CNT_RSTLDBS_FIVE_KM',
              'ENFSN_CNT_WHRTBS_THREE_KM', 'ENFSN_CNT_WHRTBS_FIVE_KM',
              'ENFSN_CNT_RSTLDBS_THREE_KM', 'ENFSN_CNT_RSTLDBS_FIVE_KM'
              ]]
# df_WORK.to_csv('df_WORK.csv', encoding = 'cp949')

df_BULD = df[["ID",
              'BULD_CNT_FVHD_METER', 'BULD_CNT_ONE_KM', 'BULD_CNT_THREE_KM', 'BULD_CNT_FIVE_KM',
              'HOUSE_CNT_FVHD_METER', 'HOUSE_CNT_ONE_KM', 'HOUSE_CNT_THREE_KM', 'HOUSE_CNT_FIVE_KM',
              'OLDBLD_CNT_FVHD_METER', 'OLDBLD_CNT_ONE_KM', 'OLDBLD_CNT_THREE_KM', 'OLDBLD_CNT_FIVE_KM',
              'OLDHS_CNT_FVHD_METER', 'OLDHS_CNT_ONE_KM', 'OLDHS_CNT_THREE_KM', 'OLDHS_CNT_FIVE_KM'
              ]]
# df_BULD.to_csv('df_BULD.csv', encoding = 'cp949')

df_LDUSE = df[["ID",
               'LDUSE_BULD_CMPDGR_FVHD_METER', 'LDUSE_BULD_CMPDGR_ONE_KM', 'LDUSE_BULD_CMPDGR_THREE_KM', 'LDUSE_BULD_CMPDGR_FIVE_KM',
               'LDUSE_BULD_CMPLXDGR_FVHD_METER', 'LDUSE_BULD_CMPLXDGR_ONE_KM', 'LDUSE_BULD_CMPLXDGR_THREE_KM', 'LDUSE_BULD_CMPLXDGR_FIVE_KM'
               ]]
# df_LDUSE.to_csv('df_LDUSE.csv', encoding = 'cp949')

df_TFCVN = df[["ID",
               "TFCVN_MARD_SHRTDST","TFCVN_NRBMARD_ECNT_FVHD_METER",
               "TFCVN_SMRD_SHRTDST","TFCVN_NRBSMRD_ECNT_FVHD_METER",
               "TFCVN_STRT_SHRTDST","TFCVN_NRBSRT_ECNT_FVHD_METER",
               "TFCVN_BSTP_SHRTDST","TFCVN_BSTP_CNT_FVHD_METER"
               ]]
# df_TFCVN.to_csv('df_TFCVN.csv', encoding = 'cp949')

df_EDU = df[["ID",
               "LVCNS_ELESCH_SHRTDST","LVCNS_ELESCH_CNT_FVHD_METER","LVCNS_ELESCH_CNT_ONE_KM",'LVCNS_ELESCH_CNT_THREE_KM', 'LVCNS_ELESCH_CNT_FIVE_KM',
               'LVCNS_MSKUL_SHRTDST', 'LVCNS_MSKUL_CNT_FVHD_METER', 'LVCNS_MSKUL_CNT_ONE_KM', 'LVCNS_MSKUL_CNT_THREE_KM', 'LVCNS_MSKUL_CNT_FIVE_KM',
               'LVCNS_HGSCHL_SHRTDST', 'LVCNS_HGSCHL_CNT_FVHD_METER', 'LVCNS_HGSCHL_CNT_ONE_KM','LVCNS_HGSCHL_CNT_THREE_KM', 'LVCNS_HGSCHL_CNT_FIVE_KM'
               ]]
# df_EDU.to_csv('df_EDU.csv', encoding = 'cp949')

df_LVCNS = df[["ID",
               "LVCNS_INSTUT_SHRTDST","LVCNS_INSTUT_CNT_FVHD_METER",'LVCNS_INSTUT_CNT_ONE_KM', 'LVCNS_INSTUT_CNT_THREE_KM', 'LVCNS_INSTUT_CNT_FIVE_KM',
               "LVCNS_CLNCHSPTL_SHRTDST","LVCNS_CLNCHSPTL_CNT_FVHD_METER","LVCNS_CLNCHSPTL_CNT_ONE_KM",'LVCNS_CLNCHSPTL_CNT_THREE_KM', 'LVCNS_CLNCHSPTL_CNT_FIVE_KM',
               "LVCNS_BANK_SHRTDST","LVCNS_BANK_CNT_FVHD_METER","LVCNS_BANK_CNT_ONE_KM",'LVCNS_BANK_CNT_THREE_KM', 'LVCNS_BANK_CNT_FIVE_KM',
               "LVCNS_MARTDMST_SHRTDST","LVCNS_MARTDMST_CNT_FVHD_METER","LVCNS_MARTDMST_CNT_ONE_KM",'LVCNS_MARTDMST_CNT_THREE_KM', 'LVCNS_MARTDMST_CNT_FIVE_KM',
               "LVCNS_PBPACFCT_SHRTDST","LVCNS_PBPACFCT_CNT_FVHD_METER","LVCNS_PBPACFCT_CNT_ONE_KM",'LVCNS_PBPACFCT_CNT_THREE_KM', 'LVCNS_PBPACFCT_CNT_FIVE_KM',
               "LVCNS_CVST_SHRTDST","LVCNS_CVST_CNT_FVHD_METER","LVCNS_CVST_CNT_ONE_KM",'LVCNS_CVST_CNT_THREE_KM', 'LVCNS_CVST_CNT_FIVE_KM']]
# df_LVCNS.to_csv('df_LVCNS.csv', encoding = 'cp949')

df_HMFNFCT = df[['ID', 
                 'HMFNFCT_PWPLNT_SHRTDST','HMFNFCT_CMTYFCT_SHRTDST','HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST']]
# df_HMFNFCT.to_csv('df_HMFNFCT.csv', encoding = 'cp949')


#%% 빈도분석 함수
def count(df, col):
    return df[col].value_counts()

def frequency_analysis(sr, title = 'frequency_analysis', grid_interval = 500):
    plt.bar(sr.index, sr.values,
            color = '#644829')
    plt.yticks()
    
    plt.title(title, fontsize = 20)
    plt.xticks(sr.index,rotation=-90, size=10) # 개체 이름 90도 회전
    plt.yticks(range(0,max(sr.values)+grid_interval, grid_interval)) # default:200 단위로 격자 생성
    plt.grid(True, axis='y', color = 'k', linestyle = '-') 
    plt.savefig(title+'.jpg')
    plt.show()

#%% 빈도분석
#시도
os.chdir(link+'빈도분석//')

back_ground_color(figsize=(10,6))
frequency_analysis(count(df_REGION, 'SIDO_NM'), grid_interval = 5000)

#시군구
for i in list(df_REGION['SIDO_NM'].value_counts().index):
    back_ground_color(figsize=(10,6))
    sr = count(df_REGION[df_REGION['SIDO_NM'] == i], 'SIGNGU_NM')
    frequency_analysis(sr, title = i)

#제주&세종
back_ground_color(figsize=(10,6))
sr = pd.concat((count(df_REGION[df_REGION['SIDO_NM'] == '제주'], 'SIGNGU_NM'),count(df_REGION[df_REGION['SIDO_NM'] == '세종'], 'SIGNGU_NM')))
frequency_analysis(sr, title = '제주&세종')

cnt_total_sido = count(df, 'SIDO_NM').to_frame()
cnt_total_sido.to_excel('cnt_total_sido.xlsx')

#%% 분포 시각화
#2 by 1 subplot
def hist_box_2(dist_df,dist_col,title):
    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 배경색
    fig.patch.set_facecolor('#EFEDE3')
    for ax in (ax1,ax2):
        ax.set_facecolor('#EFEDE3')
    
    #그래프 색상
    colors = ['#8B7355', '#800080']
    
    #그래프 생성
    #1
    sns.histplot(dist_df[dist_col[0]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax1) #Histogram
    ax2_1 = ax1.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[0]], color = colors[1], width=0.3, 
                # fliersize=0, 
                linewidth=1.5, ax=ax2_1, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax1.set_ylabel('Histogram', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2_1.set_ylabel('Boxplot', color=colors[1])
    ax2_1.tick_params(axis='y', labelcolor=colors[1])
    ax1.set_xlabel(col_name_kor_dict[dist_col[0]])
    
    #2
    sns.histplot(dist_df[dist_col[1]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax2) #Histogram
    ax2_2 = ax2.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[1]], color = colors[1], width=0.3,
                # fliersize=0,
                linewidth=1.5, ax=ax2_2, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax2.set_ylabel('Histogram', color=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2_2.set_ylabel('Boxplot', color=colors[1])
    ax2_2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_xlabel(col_name_kor_dict[dist_col[1]])
    
    # 플롯 제목 설정
    plt.suptitle(title, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()
    
    
# 3 by 1 Subplot
def hist_box_3(dist_df,dist_col,title):
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 배경색
    fig.patch.set_facecolor('#EFEDE3')
    for ax in (ax1,ax2,ax3):
        ax.set_facecolor('#EFEDE3')
        
    #그래프 색상
    colors = ['#8B7355', '#800080']
    
    #그래프 생성
    #1
    sns.histplot(dist_df[dist_col[0]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax1) #Histogram
    ax2_1 = ax1.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[0]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_1, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax1.set_ylabel('Histogram', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2_1.set_ylabel('Boxplot', color=colors[1])
    ax2_1.tick_params(axis='y', labelcolor=colors[1])
    ax1.set_xlabel(col_name_kor_dict[dist_col[0]])
    
    #2
    sns.histplot(dist_df[dist_col[1]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax2) #Histogram
    ax2_2 = ax2.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[1]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_2, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax2.set_ylabel('Histogram', color=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2_2.set_ylabel('Boxplot', color=colors[1])
    ax2_2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_xlabel(col_name_kor_dict[dist_col[1]])
    
    #3
    sns.histplot(dist_df[dist_col[2]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax3) #Histogram
    ax2_3 = ax3.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[2]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_3, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax3.set_ylabel('Histogram', color=colors[0])
    ax3.tick_params(axis='y', labelcolor=colors[0])
    ax2_3.set_ylabel('Boxplot', color=colors[1])
    ax2_3.tick_params(axis='y', labelcolor=colors[1])
    ax3.set_xlabel(col_name_kor_dict[dist_col[2]])
    
    # 플롯 제목 설정
    plt.suptitle(title, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()


#2 by 2 Subplot
def hist_box_4(dist_df,dist_col,title):
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 배경색
    fig.patch.set_facecolor('#EFEDE3')
    for ax in (ax1,ax2,ax3,ax4):
        ax.set_facecolor('#EFEDE3')
    
    #그래프 색상
    colors = ['#8B7355', '#800080']
    #그래프 생성
    #1
    sns.histplot(dist_df[dist_col[0]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax1) #Histogram
    ax2_1 = ax1.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[0]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_1, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax1.set_ylabel('Histogram', color=colors[0])
    ax1.tick_params(axis='y', labelcolor=colors[0])
    ax2_1.set_ylabel('Boxplot', color=colors[1])
    ax2_1.tick_params(axis='y', labelcolor=colors[1])
    
    ax1.set_xlabel(col_name_kor_dict[dist_col[0]])

    #2
    sns.histplot(dist_df[dist_col[1]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax2) #Histogram
    ax2_2 = ax2.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[1]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_2, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax2.set_ylabel('Histogram', color=colors[0])
    ax2.tick_params(axis='y', labelcolor=colors[0])
    ax2_2.set_ylabel('Boxplot', color=colors[1])
    ax2_2.tick_params(axis='y', labelcolor=colors[1])
    
    ax2.set_xlabel(col_name_kor_dict[dist_col[1]])

    #3
    sns.histplot(dist_df[dist_col[2]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax3) #Histogram
    ax2_3 = ax3.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[2]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_3, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax3.set_ylabel('Histogram', color=colors[0])
    ax3.tick_params(axis='y', labelcolor=colors[0])
    ax2_3.set_ylabel('Boxplot', color=colors[1])
    ax2_3.tick_params(axis='y', labelcolor=colors[1])
    
    ax3.set_xlabel(col_name_kor_dict[dist_col[2]])

    #4
    sns.histplot(dist_df[dist_col[3]], kde=True, color = colors[0], alpha=0.6, element='step', ax = ax4) #Histogram
    ax2_4 = ax4.twinx() #두번째 y축 생성
    sns.boxplot(x=dist_df[dist_col[3]], color = colors[1], width=0.3, fliersize=0, linewidth=1.5, ax=ax2_4, boxprops=dict(alpha=0.6)) #Boxplot
    
    #옵션 지정
    ax4.set_ylabel('Histogram', color=colors[0])
    ax4.tick_params(axis='y', labelcolor=colors[0])
    ax2_4.set_ylabel('Boxplot', color=colors[1])
    ax2_4.tick_params(axis='y', labelcolor=colors[1])
    ax4.set_xlabel(col_name_kor_dict[dist_col[3]])

    # 플롯 제목 설정
    plt.suptitle(title, fontsize=20)
    
    plt.tight_layout()
    plt.savefig(title+'.jpg')
    plt.show()
#%% 원본 데이터
os.chdir(link+'분포시각화_원본//')

hist_box_2(df_POP, ['TOTPUL_CNT_FVHD_METER', 'TOTPUL_CNT_FIVE_KM'],'총인구수')
hist_box_2(df_POP, ['SNCTPUL_CNT_FVHD_METER', 'SNCTPUL_CNT_FIVE_KM'],'노인인구수')
hist_box_2(df_POP, ['YUPUL_CNT_FVHD_METER', 'YUPUL_CNT_FIVE_KM'],'유소년인구수')
hist_box_2(df_POP, ['WKAGEPUL_CNT_FVHD_METER', 'WKAGEPUL_CNT_FIVE_KM'],'생산가능인구수')
hist_box_2(df_POP, ["TOTHOHLD_CNT_FVHD_METER","TOTHOHLD_CNT_FIVE_KM"], '총가구수')

hist_box_4(df_WORK, ["TOTBS_CNT_FVHD_METER","TOTBS_CNT_FIVE_KM","BSNS_CNT_WHRTBS_FIVE_KM","BSNS_CNT_RSTLDBS_FIVE_KM"], '사업체')
hist_box_4(df_WORK, ["TOTENFSN_CNT_FVHD_METER","TOTENFSN_CNT_FIVE_KM","ENFSN_CNT_WHRTBS_FIVE_KM","ENFSN_CNT_RSTLDBS_FIVE_KM"], '종사자')

hist_box_2(df_BULD, ["BULD_CNT_FVHD_METER","BULD_CNT_FIVE_KM"], '건물수')
hist_box_2(df_BULD, ["HOUSE_CNT_FVHD_METER","HOUSE_CNT_FIVE_KM"], '주택수')    
hist_box_2(df_BULD, ["OLDBLD_CNT_FVHD_METER","OLDBLD_CNT_FIVE_KM"], '노후건물수')
hist_box_2(df_BULD, ["OLDHS_CNT_FVHD_METER","OLDHS_CNT_FIVE_KM"], '노후주택수')

hist_box_2(df_LDUSE, ["LDUSE_BULD_CMPDGR_FVHD_METER","LDUSE_BULD_CMPDGR_FIVE_KM"], '토지이용_압축도')
hist_box_2(df_LDUSE, ["LDUSE_BULD_CMPLXDGR_FVHD_METER","LDUSE_BULD_CMPLXDGR_FIVE_KM"], '토지이용_복합도')

hist_box_2(df_TFCVN, ["TFCVN_MARD_SHRTDST","TFCVN_NRBMARD_ECNT_FVHD_METER"], '대로')
hist_box_2(df_TFCVN, ["TFCVN_SMRD_SHRTDST","TFCVN_NRBSMRD_ECNT_FVHD_METER"], '로')
hist_box_2(df_TFCVN, ["TFCVN_STRT_SHRTDST","TFCVN_NRBSRT_ECNT_FVHD_METER"], '길')
hist_box_2(df_TFCVN, ["TFCVN_BSTP_SHRTDST","TFCVN_BSTP_CNT_FVHD_METER"], '버스정류장')

hist_box_3(df_EDU, ["LVCNS_ELESCH_SHRTDST","LVCNS_ELESCH_CNT_FVHD_METER","LVCNS_ELESCH_CNT_FIVE_KM"], '초등학교')
hist_box_3(df_EDU, ["LVCNS_MSKUL_SHRTDST","LVCNS_MSKUL_CNT_FVHD_METER","LVCNS_MSKUL_CNT_FIVE_KM"], '중학교')
hist_box_3(df_EDU, ["LVCNS_HGSCHL_SHRTDST","LVCNS_HGSCHL_CNT_FVHD_METER","LVCNS_HGSCHL_CNT_FIVE_KM"], '고등학교')

hist_box_3(df_LVCNS, ["LVCNS_INSTUT_SHRTDST","LVCNS_INSTUT_CNT_FVHD_METER","LVCNS_INSTUT_CNT_FIVE_KM"], '학원')
hist_box_3(df_LVCNS, ["LVCNS_CLNCHSPTL_SHRTDST","LVCNS_CLNCHSPTL_CNT_FVHD_METER","LVCNS_CLNCHSPTL_CNT_FIVE_KM"], '병의원')
hist_box_3(df_LVCNS, ["LVCNS_BANK_SHRTDST","LVCNS_BANK_CNT_FVHD_METER","LVCNS_BANK_CNT_FIVE_KM"], '은행')
hist_box_3(df_LVCNS, ["LVCNS_MARTDMST_SHRTDST","LVCNS_MARTDMST_CNT_FVHD_METER","LVCNS_MARTDMST_CNT_FIVE_KM"], '마트백화점')
hist_box_3(df_LVCNS, ["LVCNS_PBPACFCT_SHRTDST","LVCNS_PBPACFCT_CNT_FVHD_METER","LVCNS_PBPACFCT_CNT_FIVE_KM"], '치안시설')
hist_box_3(df_LVCNS, ["LVCNS_CVST_SHRTDST","LVCNS_CVST_CNT_FVHD_METER","LVCNS_CVST_CNT_FIVE_KM"], '편의점')

hist_box_4(df_HMFNFCT,['HMFNFCT_PWPLNT_SHRTDST','HMFNFCT_CMTYFCT_SHRTDST','HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST'],'유해시설_최단거리')

#%% 유의미한 EDA : log 변환 및 (값 == 0)관측치 제외 by 500m
os.chdir(link+'분포시각화_수정//')

df_POP_eda = np.log1p(df_POP.drop(['ID'], axis = 1))
hist_box_2(df_POP_eda, ['TOTPUL_CNT_FVHD_METER', 'TOTPUL_CNT_FIVE_KM'],'총인구수_L') #L
hist_box_2(df_POP_eda, ['SNCTPUL_CNT_FVHD_METER', 'SNCTPUL_CNT_FIVE_KM'],'노인인구수_L') #L
hist_box_2(df_POP_eda, ['YUPUL_CNT_FVHD_METER', 'YUPUL_CNT_FIVE_KM'],'유소년인구수_L') #L
hist_box_2(df_POP_eda, ['WKAGEPUL_CNT_FVHD_METER', 'WKAGEPUL_CNT_FIVE_KM'],'생산가능인구수_L') #L
hist_box_2(df_POP_eda, ["TOTHOHLD_CNT_FVHD_METER","TOTHOHLD_CNT_FIVE_KM"], '총가구수_L') #L

df_WORK_eda = np.log1p(df_WORK.drop(['ID'], axis = 1))
hist_box_4(df_WORK_eda, ["TOTBS_CNT_FVHD_METER","TOTBS_CNT_FIVE_KM","BSNS_CNT_WHRTBS_FIVE_KM","BSNS_CNT_RSTLDBS_FIVE_KM"], '사업체_L') #L
hist_box_4(df_WORK_eda, ["TOTENFSN_CNT_FVHD_METER","TOTENFSN_CNT_FIVE_KM","ENFSN_CNT_WHRTBS_FIVE_KM","ENFSN_CNT_RSTLDBS_FIVE_KM"], '종사자_L') #L

df_BULD_eda = np.log1p(df_BULD.drop(['ID'], axis = 1))
hist_box_2(df_BULD_eda, ["BULD_CNT_FVHD_METER","BULD_CNT_FIVE_KM"], '건물수_L') #L
hist_box_2(df_BULD_eda, ["HOUSE_CNT_FVHD_METER","HOUSE_CNT_FIVE_KM"], '주택수_L') #L
hist_box_2(df_BULD_eda, ["OLDBLD_CNT_FVHD_METER","OLDBLD_CNT_FIVE_KM"], '노후건물수_L') #L
hist_box_2(df_BULD_eda, ["OLDHS_CNT_FVHD_METER","OLDHS_CNT_FIVE_KM"], '노후주택수_L') #L

df_LDUSE_eda = pd.concat([np.log1p(df_LDUSE[["LDUSE_BULD_CMPDGR_FVHD_METER","LDUSE_BULD_CMPDGR_FIVE_KM"]]),
                          df_LDUSE[["LDUSE_BULD_CMPLXDGR_FVHD_METER","LDUSE_BULD_CMPLXDGR_FIVE_KM"]]],
                         axis = 1)
hist_box_2(df_LDUSE_eda, ["LDUSE_BULD_CMPDGR_FVHD_METER","LDUSE_BULD_CMPDGR_FIVE_KM"], '토지이용_압축도_L') #L
hist_box_2(df_LDUSE_eda, ["LDUSE_BULD_CMPLXDGR_FVHD_METER","LDUSE_BULD_CMPLXDGR_FIVE_KM"], '토지이용_복합도')

df_TFCVN_eda = df_TFCVN.drop(df_TFCVN[(df_TFCVN['TFCVN_NRBMARD_ECNT_FVHD_METER'] == 0)|(df_TFCVN['TFCVN_MARD_SHRTDST'] == df_TFCVN['TFCVN_MARD_SHRTDST'].min())].index)
hist_box_2(df_TFCVN_eda, ["TFCVN_MARD_SHRTDST","TFCVN_NRBMARD_ECNT_FVHD_METER"], '대로_V') #V
df_TFCVN_eda = df_TFCVN.drop(df_TFCVN[(df_TFCVN['TFCVN_NRBSMRD_ECNT_FVHD_METER'] == 0)|(df_TFCVN['TFCVN_SMRD_SHRTDST'] == df_TFCVN['TFCVN_SMRD_SHRTDST'].min())].index)
hist_box_2(df_TFCVN_eda, ["TFCVN_SMRD_SHRTDST","TFCVN_NRBSMRD_ECNT_FVHD_METER"], '로_V') #V
df_TFCVN_eda = df_TFCVN.drop(df_TFCVN[(df_TFCVN['TFCVN_NRBSRT_ECNT_FVHD_METER'] == 0)|(df_TFCVN['TFCVN_STRT_SHRTDST'] == df_TFCVN['TFCVN_STRT_SHRTDST'].min())].index)
hist_box_2(df_TFCVN_eda, ["TFCVN_STRT_SHRTDST","TFCVN_NRBSRT_ECNT_FVHD_METER"], '길_V') #V
df_TFCVN_eda = df_TFCVN.drop(df_TFCVN[(df_TFCVN['TFCVN_BSTP_CNT_FVHD_METER'] == 0)|(df_TFCVN['TFCVN_BSTP_SHRTDST'] == df_TFCVN['TFCVN_BSTP_SHRTDST'].min())].index)
hist_box_2(df_TFCVN_eda, ["TFCVN_BSTP_SHRTDST","TFCVN_BSTP_CNT_FVHD_METER"], '버스정류장_V') #V

df_EDU_eda = df_EDU.drop(df_EDU[(df_EDU['LVCNS_ELESCH_CNT_FVHD_METER'] == 0)|(df_EDU['LVCNS_ELESCH_SHRTDST'] == df_EDU['LVCNS_ELESCH_SHRTDST'].min())].index)
hist_box_3(df_EDU_eda, ["LVCNS_ELESCH_SHRTDST","LVCNS_ELESCH_CNT_FVHD_METER","LVCNS_ELESCH_CNT_FIVE_KM"], '초등학교_V') #V
df_EDU_eda = df_EDU.drop(df_EDU[(df_EDU['LVCNS_MSKUL_CNT_FVHD_METER'] == 0)|(df_EDU['LVCNS_MSKUL_SHRTDST'] == df_EDU['LVCNS_MSKUL_SHRTDST'].min())].index)
hist_box_3(df_EDU_eda, ["LVCNS_MSKUL_SHRTDST","LVCNS_MSKUL_CNT_FVHD_METER","LVCNS_MSKUL_CNT_FIVE_KM"], '중학교_V') #V
df_EDU_eda = df_EDU.drop(df_EDU[(df_EDU['LVCNS_HGSCHL_CNT_FVHD_METER'] == 0)|(df_EDU['LVCNS_HGSCHL_SHRTDST'] == df_EDU['LVCNS_HGSCHL_SHRTDST'].min())].index)
hist_box_3(df_EDU_eda, ["LVCNS_HGSCHL_SHRTDST","LVCNS_HGSCHL_CNT_FVHD_METER","LVCNS_HGSCHL_CNT_FIVE_KM"], '고등학교_V') #V

df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_INSTUT_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_INSTUT_SHRTDST'] == df_LVCNS['LVCNS_INSTUT_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_INSTUT_SHRTDST","LVCNS_INSTUT_CNT_FVHD_METER","LVCNS_INSTUT_CNT_FIVE_KM"], '학원_V') #V
df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_CLNCHSPTL_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_CLNCHSPTL_SHRTDST'] == df_LVCNS['LVCNS_CLNCHSPTL_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_CLNCHSPTL_SHRTDST","LVCNS_CLNCHSPTL_CNT_FVHD_METER","LVCNS_CLNCHSPTL_CNT_FIVE_KM"], '병의원_V') #V
df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_BANK_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_BANK_SHRTDST'] == df_LVCNS['LVCNS_BANK_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_BANK_SHRTDST","LVCNS_BANK_CNT_FVHD_METER","LVCNS_BANK_CNT_FIVE_KM"], '은행_V') #V
df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_MARTDMST_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_MARTDMST_SHRTDST'] == df_LVCNS['LVCNS_MARTDMST_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_MARTDMST_SHRTDST","LVCNS_MARTDMST_CNT_FVHD_METER","LVCNS_MARTDMST_CNT_FIVE_KM"], '마트백화점_V') #V
df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_PBPACFCT_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_PBPACFCT_SHRTDST'] == df_LVCNS['LVCNS_PBPACFCT_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_PBPACFCT_SHRTDST","LVCNS_PBPACFCT_CNT_FVHD_METER","LVCNS_PBPACFCT_CNT_FIVE_KM"], '치안시설_V') #V
df_LVCNS_eda = df_LVCNS.drop(df_LVCNS[(df_LVCNS['LVCNS_CVST_CNT_FVHD_METER'] == 0)|(df_LVCNS['LVCNS_CVST_SHRTDST'] == df_LVCNS['LVCNS_CVST_SHRTDST'].min())].index)
hist_box_3(df_LVCNS_eda, ["LVCNS_CVST_SHRTDST","LVCNS_CVST_CNT_FVHD_METER","LVCNS_CVST_CNT_FIVE_KM"], '편의점_V') #V

df_HMFNFCT_eda = df_HMFNFCT.drop(df_HMFNFCT[(df_HMFNFCT['HMFNFCT_PWPLNT_SHRTDST'] == df_HMFNFCT['HMFNFCT_PWPLNT_SHRTDST'].max())|
                                            (df_HMFNFCT['HMFNFCT_CMTYFCT_SHRTDST'] == df_HMFNFCT['HMFNFCT_CMTYFCT_SHRTDST'].max())|
                                            (df_HMFNFCT['HMFNFCT_ESRNFCT_SHRTDST'] == df_HMFNFCT['HMFNFCT_ESRNFCT_SHRTDST'].max())|
                                            (df_HMFNFCT['HMFNFCT_WSTFCT_SHRTDST'] == df_HMFNFCT['HMFNFCT_WSTFCT_SHRTDST'].max())].index)
hist_box_4(df_HMFNFCT_eda,['HMFNFCT_PWPLNT_SHRTDST','HMFNFCT_CMTYFCT_SHRTDST','HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST'],'유해시설_최단거리_V') #V

#%% 변수 선택
df = df[['TOTPUL_CNT_FVHD_METER', 
'SNCTPUL_CNT_FVHD_METER', 
'YUPUL_CNT_FVHD_METER', 
'WKAGEPUL_CNT_FVHD_METER', 
"TOTHOHLD_CNT_FVHD_METER",

"TOTBS_CNT_FVHD_METER",
"TOTENFSN_CNT_FVHD_METER",
"BULD_CNT_FVHD_METER",
"HOUSE_CNT_FVHD_METER",
"OLDBLD_CNT_FVHD_METER",
"OLDHS_CNT_FVHD_METER",
"LDUSE_BULD_CMPDGR_FVHD_METER",
"LDUSE_BULD_CMPLXDGR_FVHD_METER",

"TFCVN_NRBMARD_ECNT_FVHD_METER",
"TFCVN_NRBSMRD_ECNT_FVHD_METER",
"TFCVN_NRBSRT_ECNT_FVHD_METER",
"TFCVN_BSTP_CNT_FVHD_METER",
"LVCNS_ELESCH_CNT_FVHD_METER",
"LVCNS_MSKUL_CNT_FVHD_METER",
"LVCNS_HGSCHL_CNT_FVHD_METER",
"LVCNS_INSTUT_CNT_FVHD_METER",
"LVCNS_CLNCHSPTL_CNT_FVHD_METER",
"LVCNS_BANK_CNT_FVHD_METER",
"LVCNS_MARTDMST_CNT_FVHD_METER",
"LVCNS_PBPACFCT_CNT_FVHD_METER",
"LVCNS_CVST_CNT_FVHD_METER",

'TFCVN_MARD_SHRTDST',
'TFCVN_SMRD_SHRTDST',
'TFCVN_STRT_SHRTDST',
'TFCVN_BSTP_SHRTDST',
'LVCNS_ELESCH_SHRTDST',
'LVCNS_MSKUL_SHRTDST',
'LVCNS_HGSCHL_SHRTDST',
'LVCNS_INSTUT_SHRTDST',
'LVCNS_CLNCHSPTL_SHRTDST',
'LVCNS_BANK_SHRTDST',
'LVCNS_MARTDMST_SHRTDST',
'LVCNS_PBPACFCT_SHRTDST',
'LVCNS_CVST_SHRTDST',
'HMFNFCT_PWPLNT_SHRTDST',
'HMFNFCT_CMTYFCT_SHRTDST',
'HMFNFCT_ESRNFCT_SHRTDST',
'HMFNFCT_WSTFCT_SHRTDST']]


sep_heatmap_list = [
['TOTPUL_CNT_FVHD_METER', 
'SNCTPUL_CNT_FVHD_METER', 
'YUPUL_CNT_FVHD_METER', 
'WKAGEPUL_CNT_FVHD_METER', 
"TOTHOHLD_CNT_FVHD_METER",

"TOTBS_CNT_FVHD_METER",
"TOTENFSN_CNT_FVHD_METER",
"BULD_CNT_FVHD_METER",
"HOUSE_CNT_FVHD_METER",
"OLDBLD_CNT_FVHD_METER",
"OLDHS_CNT_FVHD_METER",
"LDUSE_BULD_CMPDGR_FVHD_METER",
"LDUSE_BULD_CMPLXDGR_FVHD_METER",], 

['TOTPUL_CNT_FVHD_METER', 
'SNCTPUL_CNT_FVHD_METER', 
'YUPUL_CNT_FVHD_METER', 
'WKAGEPUL_CNT_FVHD_METER', 
"TOTHOHLD_CNT_FVHD_METER",

"TFCVN_NRBMARD_ECNT_FVHD_METER",
"TFCVN_NRBSMRD_ECNT_FVHD_METER",
"TFCVN_NRBSRT_ECNT_FVHD_METER",
"TFCVN_BSTP_CNT_FVHD_METER",
"LVCNS_ELESCH_CNT_FVHD_METER",
"LVCNS_MSKUL_CNT_FVHD_METER",
"LVCNS_HGSCHL_CNT_FVHD_METER",
"LVCNS_INSTUT_CNT_FVHD_METER",
"LVCNS_CLNCHSPTL_CNT_FVHD_METER",
"LVCNS_BANK_CNT_FVHD_METER",
"LVCNS_MARTDMST_CNT_FVHD_METER",
"LVCNS_PBPACFCT_CNT_FVHD_METER",
"LVCNS_CVST_CNT_FVHD_METER",
],

['TOTPUL_CNT_FVHD_METER', 
'SNCTPUL_CNT_FVHD_METER', 
'YUPUL_CNT_FVHD_METER', 
'WKAGEPUL_CNT_FVHD_METER', 
"TOTHOHLD_CNT_FVHD_METER",

'TFCVN_MARD_SHRTDST',
'TFCVN_SMRD_SHRTDST',
'TFCVN_STRT_SHRTDST',
'TFCVN_BSTP_SHRTDST',
'LVCNS_ELESCH_SHRTDST',
'LVCNS_MSKUL_SHRTDST',
'LVCNS_HGSCHL_SHRTDST',
'LVCNS_INSTUT_SHRTDST',
'LVCNS_CLNCHSPTL_SHRTDST',
'LVCNS_BANK_SHRTDST',
'LVCNS_MARTDMST_SHRTDST',
'LVCNS_PBPACFCT_SHRTDST',
'LVCNS_CVST_SHRTDST',
'HMFNFCT_PWPLNT_SHRTDST',
'HMFNFCT_CMTYFCT_SHRTDST',
'HMFNFCT_ESRNFCT_SHRTDST',
'HMFNFCT_WSTFCT_SHRTDST'],
]
#%% heatmap
def heatmap_cov(data, title='Covariance_Matrix'):
    back_ground_color(figsize=(len(data.columns),len(data.columns)*0.8))
    
    data_cov = data.cov()
    data_cov.index = [col_name_kor_dict[w] for w in list(data.columns)]

    sns.heatmap(data_cov,
                cmap="Greys",
                )
    
    plt.xticks([])
    plt.savefig(title+'.jpg')
    plt.show()
#%%
def heatmap_corr(data, title='Correlation_Matrix'):
    back_ground_color(figsize=(len(data.columns),len(data.columns)*0.8))
    
    data_cor = data.corr()
    data_cor.index = [col_name_kor_dict[w] for w in list(data.columns)]
    
    sns.heatmap(data_cor,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1)
    plt.xticks([])
    # plt.title(title)
    # plt.savefig(title+'.jpg')
    plt.show()

#%% 전체 히트맵
os.chdir(link+'Heatmap//')

# corr matrix를 위한 df
df_cor = df[VariableNames.col_sorting]

heatmap_corr(df_cor)
# for c in sep_heatmap_list:
#     heatmap_corr(df[c], title = c[-1])
# heatmap_cov(df_cor)

# cov_mat = df_cor.cov()
#%% corr matrix 저장
# os.chdir(link)
# cor_mat = df_cor.corr()
# cor_mat.to_csv('cor_mat.csv', encoding='cp949')

#%% 통합 데이터프레임
os.chdir(link)
df_main = df_REGION[['ID']]
df_main = pd.concat([df_main, df], axis = 1)
df_main.to_csv('df_main.csv')
# df_main.to_csv('df_main_no_minmax.csv')

#%%
#기술통계량 출력
df_describe_number = df_main.describe(include=['number'])
df_describe_number.to_html('df_describe_number.html')

#%% 부분 히트맵
# dfs = [df_REGION, df_POP, df_WORK, df_BULD, df_LDUSE, df_TFCVN, df_EDU, df_LVCNS, df_HMFNFCT]
# dfs_name = ['df_REGION', 'df_POP', 'df_WORK', 'df_BULD', 'df_LDUSE', 'df_TFCVN', 'df_EDU', 'df_LVCNS', 'df_HMFNFCT']
# for data, name in zip(dfs[1:],dfs_name[1:]):
#     heatmap_corr(data[data.columns[1:]], name)


