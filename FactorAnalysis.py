# -*- coding: utf-8 -*-
"""
Factor Analysis

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
#%% 변수 이름
col_name_kor_dict = VariableNames.col_name_kor_dict
col_name_pca = VariableNames.col_name_pca

#%% dpi 세팅 #해상도
#plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
#sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_context('notebook')
sns.set_style("ticks")
#%% 출력 세팅 #... 없앰
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# pd.options.display.max_rows = 100
# pd.options.display.max_columns = 70

#%% 한글 폰트 오류 해결
from matplotlib import font_manager, rc
font_path = 'malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False #마이너스 부호 오류 해결

#%% 배경색
def back_ground_color(figsize=(8,6), bg_color = "#EFEDE3"):
    fig = plt.figure(figsize=figsize)
    plt.box(False)
    
    fig.patch.set_facecolor(bg_color)
    fig.set_facecolor(bg_color)

#%% 데이터 호출
os.chdir(link)

df_region = pd.read_csv('df_REGION.csv', encoding = 'cp949')

df = pd.read_csv('df_main.csv')
df = df[df.columns[1:]] #csv 변환 과정에서 발생한 문제. 해결 코드

df = df.drop(['ID'], axis=1) #ID 제거
df = df[VariableNames.col_sorting]

#%% 지역별
# a = list(df_region[(df_region['SIDO_NM'] == '서울')
#                     |(df_region['SIDO_NM'] == '경기')
#                     |(df_region['SIDO_NM'] == '경기')
#                     |(df_region['SIDO_NM'] == '경기')
#                     |(df_region['SIDO_NM'] == '경기')
#                    ]['ID'])

# df = df[df['ID'].isin(a)]

#%% 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_StandardScaling = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaler = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)


#%% PCA 함수
from sklearn.decomposition import PCA

def PCA_function(df_temp):
    
    pca = PCA(n_components=1) #PCA #max(1,len(df_temp.columns)-1)
    data = pca.fit_transform(df_temp) #PCA fitting
    
    if pca.explained_variance_ratio_ > 0.9 and len(df_temp.columns) != 1: #설명분산 0.9 이상인 변수집단은 500m만 채택
        data = df_temp[df_temp.columns[0]]
    
    # print('설명분산 :', pca.explained_variance_ratio_)
    return data, pca.explained_variance_ratio_

#%% 유해시설 매장
a = pd.DataFrame(scaler.fit_transform(pd.read_csv('df_main.csv')[['HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST']]), columns = ['HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST'])
b = pd.DataFrame(scaler.fit_transform(pd.read_csv('df_main_no_minmax.csv')[['HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST']]), columns = ['HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST'])
pca_a,_ = PCA_function(a)
pca_b,_ = PCA_function(b)
dfa = pd.concat([a,pd.DataFrame(pca_a, columns = ['PCA'])], axis = 1)
dfb = pd.concat([b,pd.DataFrame(pca_b, columns = ['PCA'])], axis = 1)

#%% 표준화 vs 로부스트
df_diff = pd.DataFrame(columns = ['변수', 'StandardScaling','RobustScaling'])
for w in col_name_pca:
    if len(w) == 1:
        continue
    _,standard_scaling_ev = PCA_function(df_StandardScaling[w])
    _,robust_scaling_ev = PCA_function(df_scaler[w])
    df_temp = pd.DataFrame({'변수':str([col_name_kor_dict[i] for i in w]), 'StandardScaling':[round(sum(standard_scaling_ev),3)],'RobustScaling':[round(sum(robust_scaling_ev),3)]})
    df_diff = pd.concat([df_diff, df_temp])

df_diff.index = df_diff['변수']
df_diff = df_diff.drop(['변수'], axis=1)
df_diff.to_html('df_diff.html')

#%% PCA 및 변수 이름 변경
df_pca = pd.DataFrame()

for w in col_name_pca :

    col_name = [col_name_kor_dict[i] for i in w] #한글 변수
    print('\n-변수- :',col_name)
    
    data, ev = PCA_function(df_scaler[w]) #변수 단일화
    print('설명분산 :', ev[0])
    
    sep_col_name = w[0][0:w[0].find('_',w[0].find('_')+1)] #이름 축약
    kor_col = col_name_kor_dict[sep_col_name]
    
    if kor_col in df_pca.columns: #S
        kor_col = '건물복합도'

    if type(data) == np.ndarray:
        df_temp = pd.DataFrame(data, columns = [kor_col])
    else:
        df_temp = data.to_frame()
        df_temp.columns = [kor_col]
    print('새로운 변수 이름 : ',kor_col)
    
    if ev < 0.6 :
        df_temp = df_scaler[w]
        df_temp.columns = [col_name_kor_dict[i] for i in w]
    if kor_col == '유해시설_매장':
        df_temp = -df_temp
        
    df_pca = pd.concat([df_pca, df_temp], axis=1)


#%%
def heatmap_corr(data, title='Correlation_Matrix'):
    back_ground_color(figsize=(len(data.columns),len(data.columns)*0.8))
    
    data_cor = data.corr()
    
    sns.heatmap(data_cor,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1)
    plt.xticks([])
    # plt.title(title)
    # plt.savefig(title+'.jpg')
    plt.show()
heatmap_corr(df_pca)

#%% Factor Analysis
#요인분석 데이터프레임
df_fa = df_pca

#%% 요인분석 적합성 검정
#바틀렛 검정
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df_fa)
print(chi_square_value, p_value) #p<.001

#KMO 검정
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df_fa)
print(round(kmo_model, 2)) #.95

#%% 요인분석 함수
#요인분석
def factor_analysis(n, df_FA, method=None):
    # n : 요인 수, df_FA : dataframe, method : rotation method
    fa = FactorAnalyzer(n_factors=n, rotation=method)
    fa.fit(df_FA)
    result = pd.DataFrame(fa.loadings_, index=df_FA.columns)

    df_communality = pd.DataFrame(fa.get_communalities(),index = df_fa.columns, columns = ['공통성'])
    df_uniqueness = pd.DataFrame(fa.get_uniquenesses(),index = df_fa.columns, columns = ['유일성'])
    
    result = pd.concat([result, df_communality], axis = 1)
    result = pd.concat([result, df_uniqueness], axis = 1)

    return result, fa

#요인분석 히트맵
def FA_heatmap(df_FA_loadings, title='Heatmap_Loadings'):
    back_ground_color(figsize=(12,17))
    
    # 절댓값 0.4 미만 제거
    # df_FA_loadings[(df_FA_loadings > -0.4) & (df_FA_loadings < 0.4)] = 0
    
    #이름 한글화
    name_kor_list = []
    for w in list(df_fa.columns):
        try:
            HMFNFCT_dict = {'HMFNFCT_PWPLNT_SHRTDST':'유해시설_발전소','HMFNFCT_CMTYFCT_SHRTDST':'유해시설_화장시설','HMFNFCT_ESRNFCT_SHRTDST':'유해시설_봉안시설','HMFNFCT_WSTFCT_SHRTDST':'유해시설_폐기물시설'}
            name_kor_list.append(HMFNFCT_dict[w])
        except:
            name_kor_list.append(w)
            
    df_FA_loadings.index = name_kor_list

    sns.heatmap(df_FA_loadings,
                cmap="RdBu",
                annot=True,
                fmt='.2f',
                vmax=1, vmin=-1,
                center=0)

    plt.title(title, fontsize=13)
    plt.savefig(title+'.png', bbox_inches='tight')
    plt.show()

#%%
# os.chdir(link+'FA//')

#요인분석 진행
from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer(n_factors=len(df_fa.columns), rotation = None)
fa.fit(df_fa)

#Kaiser's Rule
ev,v = fa.get_eigenvalues()  #eigen value
# print(ev)
ev_1 = [i for i in ev if i >= 1] # (ev >= 1) # 9

# # Rotation None
# df_FA_loadings, fa = factor_analysis(len(ev_1), df_fa, method=None)
# FA_heatmap(df_FA_loadings, title='None')
# print(round(fa.get_factor_variance()[2][-1],2))

# Orthogonal Rotations
df_FA_loadings, fa = factor_analysis(len(ev_1), df_fa, method='Varimax')
FA_heatmap(df_FA_loadings, title='Varimax')
print(round(fa.get_factor_variance()[2][-1],2))

# df_FA_loadings, fa = factor_analysis(len(ev_1), df_fa, method='Quartimax')
# FA_heatmap(df_FA_loadings, title='Quartimax')
# print(round(fa.get_factor_variance()[2][-1],2))

# # Oblique Rotations
# df_FA_loadings, fa = factor_analysis(len(ev_1), df_fa, method='Promax')
# FA_heatmap(df_FA_loadings, title='Promax')
# print(round(fa.get_factor_variance()[2][-1],2))

# df_FA_loadings, fa = factor_analysis(len(ev_1), df_fa, method='Oblimin')
# FA_heatmap(df_FA_loadings, title='Oblimin')
# print(round(fa.get_factor_variance()[2][-1],2))

#%%
#분석 결과
fa.get_factor_variance()
df_FA_variance = pd.DataFrame(np.round(fa.get_factor_variance(),2), index=['SS Loadings', 'Proportion Var', 'Cumulative Var'])

df_FA_variance.to_html('df_FA_variance.html')

#%% 데이터 변환
f1 = 0.81*df_fa['총인구수'] + 0.69*df_fa['노인인구수'] + 0.82*df_fa['유소년인구수'] + 0.82*df_fa['생산가능인구수'] + 0.76*df_fa['총가구수'] + 0.57*df_fa['학원'] + 0.52*df_fa['편의점']
f2 = 0.51*df_fa['노인인구수'] + 0.75*df_fa['건물수'] + 0.80*df_fa['주택수'] + 0.81*df_fa['노후건물수'] + 0.85*df_fa['노후주택수'] 
f3 = 0.82*df_fa['총사업체수'] + 0.80*df_fa['총종사자수'] + 0.40*df_fa['학원'] + 0.67*df_fa['병의원'] + 0.49*df_fa['은행'] + 0.62*df_fa['편의점']
f4 = 0.64*df_fa['건물복합도'] + 0.38*df_fa['정류장'] + 0.48*df_fa['초등학교'] + 0.47*df_fa['중학교'] + 0.39*df_fa['고등학교'] + 0.63*df_fa['마트백화점'] + 0.43*df_fa['치안시설'] + 0.48*df_fa['유해시설_매장']
f5 = 0.65*df_fa['로'] + 0.80*df_fa['길']


u1 = 0.90*df_fa['건물압축도']
u2 = 0.90*df_fa['대로']
u3 = 0.95*df_fa['유해시설_발전소_최단거리'] #발전소
u4 = 0.92*df_fa['유해시설_화장시설_최단거리'] #화장시설

df_X = pd.concat([f1, f2, f3, f4, f5, u1, u2, u3, u4], axis = 1)
df_X.index = df_region['ID']
df_X.columns = ['인구', '건물', '생업', '학생', '도로', 'u_건물압축도', 'u_대로', 'u_발전소', 'u_화장시설']
df_X.to_csv('df_X.csv', encoding = 'cp949')




