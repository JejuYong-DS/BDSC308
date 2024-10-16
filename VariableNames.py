# -*- coding: utf-8 -*-
"""
Variable Names

Team : 8조
Member : 고주용, 최주영

BDSC308 다차원자료분석PBL
"""

#%% charactor data columns names
col_name_char = """PNU_CD
LEGALDONG_CD
SIDO_NM
SIGNGU_NM
LEGALDONG_NM
LEGALLI_NM
LDGRFL_CD
HMNO
VCNO
LNCT_CD
LDAR
SUSAR_CD_ONE
SUSAR_CD_TWO
LAND_USE_STTN_CD
TPGRP_HGHT_CD
TPGRP_STTN_CD
ROAD_COPL_CD
STD_YN
OSILP_STRD_PNTM
TFCVN_NRBSTATN
TFCVN_NRBHWIC""".split('\n')
col_name_char.insert(0, 'ID')

#%% 변수명 한글 Dict.
col_name_kor = """PNU코드	PNU_CD
법정동코드	LEGALDONG_CD
광역시도명	SIDO_NM
시군구명	SIGNGU_NM
법정동명	LEGALDONG_NM
법정리명	LEGALLI_NM
대장구분코드	LDGRFL_CD
본번지	HMNO
부번지	VCNO
좌표X	XCRDT
좌표Y	YCRDT
지목코드	LNCT_CD
토지면적	LDAR
용도지역코드1	SUSAR_CD_ONE
용도지역코드2	SUSAR_CD_TWO
토지이용상황코드	LAND_USE_STTN_CD
지형높이코드	TPGRP_HGHT_CD
지형상황코드	TPGRP_STTN_CD
도로접면코드	ROAD_COPL_CD
표준지여부	STD_YN
개별공시지가	OSILP
개별공시지가기준시점	OSILP_STRD_PNTM
총인구수_500m	TOTPUL_CNT_FVHD_METER
총인구수_1km	TOTPUL_CNT_ONE_KM
총인구수_3km	TOTPUL_CNT_THREE_KM
총인구수_5km	TOTPUL_CNT_FIVE_KM
노인인구수_500m	SNCTPUL_CNT_FVHD_METER
노인인구수_1km	SNCTPUL_CNT_ONE_KM
노인인구수_3km	SNCTPUL_CNT_THREE_KM
노인인구수_5km	SNCTPUL_CNT_FIVE_KM
유소년인구수_500m	YUPUL_CNT_FVHD_METER
유소년인구수_1km	YUPUL_CNT_ONE_KM
유소년인구수_3km	YUPUL_CNT_THREE_KM
유소년인구수_5km	YUPUL_CNT_FIVE_KM
노령화지수_500m	AGDX_FVHD_METER
노령화지수_1km	AGDX_ONE_KM
노령화지수_3km	AGDX_THREE_KM
노령화지수_5km	AGDX_FIVE_KM
생산가능인구수_500m	WKAGEPUL_CNT_FVHD_METER
생산가능인구수_1km	WKAGEPUL_CNT_ONE_KM
생산가능인구수_3km	WKAGEPUL_CNT_THREE_KM
생산가능인구수_5km	WKAGEPUL_CNT_FIVE_KM
총가구수_500m	TOTHOHLD_CNT_FVHD_METER
총가구수_1km	TOTHOHLD_CNT_ONE_KM
총가구수_3km	TOTHOHLD_CNT_THREE_KM
총가구수_5km	TOTHOHLD_CNT_FIVE_KM
총사업체수_500m	TOTBS_CNT_FVHD_METER
총사업체수_1km	TOTBS_CNT_ONE_KM
총사업체수_3km	TOTBS_CNT_THREE_KM
총사업체수_5km	TOTBS_CNT_FIVE_KM
총종사자수_500m	TOTENFSN_CNT_FVHD_METER
총종사자수_1km	TOTENFSN_CNT_ONE_KM
총종사자수_3km	TOTENFSN_CNT_THREE_KM
총종사자수_5km	TOTENFSN_CNT_FIVE_KM
사업체수_도소매업_3km	BSNS_CNT_WHRTBS_THREE_KM
사업체수_도소매업_5km	BSNS_CNT_WHRTBS_FIVE_KM
사업체수_음식숙박업_3km	BSNS_CNT_RSTLDBS_THREE_KM
사업체수_음식숙박업_5km	BSNS_CNT_RSTLDBS_FIVE_KM
종사자수_도소매업_3km	ENFSN_CNT_WHRTBS_THREE_KM
종사자수_도소매업_5km	ENFSN_CNT_WHRTBS_FIVE_KM
종사자수_음식숙박업_3km	ENFSN_CNT_RSTLDBS_THREE_KM
종사자수_음식숙박업_5km	ENFSN_CNT_RSTLDBS_FIVE_KM
건물수_500m	BULD_CNT_FVHD_METER
건물수_1km	BULD_CNT_ONE_KM
건물수_3km	BULD_CNT_THREE_KM
건물수_5km	BULD_CNT_FIVE_KM
주택수_500m	HOUSE_CNT_FVHD_METER
주택수_1km	HOUSE_CNT_ONE_KM
주택수_3km	HOUSE_CNT_THREE_KM
주택수_5km	HOUSE_CNT_FIVE_KM
노후건물수_500m	OLDBLD_CNT_FVHD_METER
노후건물수_1km	OLDBLD_CNT_ONE_KM
노후건물수_3km	OLDBLD_CNT_THREE_KM
노후건물수_5km	OLDBLD_CNT_FIVE_KM
노후주택수_500m	OLDHS_CNT_FVHD_METER
노후주택수_1km	OLDHS_CNT_ONE_KM
노후주택수_3km	OLDHS_CNT_THREE_KM
노후주택수_5km	OLDHS_CNT_FIVE_KM
건물연면적_500m	BULD_GRFA_FVHD_METER
건물연면적_1km	BULD_GRFA_ONE_KM
건물연면적_3km	BULD_GRFA_THREE_KM
건물연면적_5km	BULD_GRFA_FIVE_KM
토지이용_건물_압축도_500m	LDUSE_BULD_CMPDGR_FVHD_METER
토지이용_건물_압축도_1km	LDUSE_BULD_CMPDGR_ONE_KM
토지이용_건물_압축도_3km	LDUSE_BULD_CMPDGR_THREE_KM
토지이용_건물_압축도_5km	LDUSE_BULD_CMPDGR_FIVE_KM
토지이용_건물_복합도_500m	LDUSE_BULD_CMPLXDGR_FVHD_METER
토지이용_건물_복합도_1km	LDUSE_BULD_CMPLXDGR_ONE_KM
토지이용_건물_복합도_3km	LDUSE_BULD_CMPLXDGR_THREE_KM
토지이용_건물_복합도_5km	LDUSE_BULD_CMPLXDGR_FIVE_KM
교통편의성_인접고속도로개수_500m	TFCVN_NRBHW_ECNT_FVHD_METER
교통편의성_고속도로_최단거리	TFCVN_HIWY_SHRTDST
교통편의성_인접대로개수_500m	TFCVN_NRBMARD_ECNT_FVHD_METER
교통편의성_대로_최단거리	TFCVN_MARD_SHRTDST
교통편의성_인접로개수_500m	TFCVN_NRBSMRD_ECNT_FVHD_METER
교통편의성_로_최단거리	TFCVN_SMRD_SHRTDST
교통편의성_인접길개수_500m	TFCVN_NRBSRT_ECNT_FVHD_METER
교통편의성_길_최단거리	TFCVN_STRT_SHRTDST
교통편의성_정류장_최단거리	TFCVN_BSTP_SHRTDST
교통편의성_정류장수_500m	TFCVN_BSTP_CNT_FVHD_METER
교통편의성_정류장수_1km	TFCVN_BSTP_CNT_ONE_KM
교통편의성_정류장수_3km	TFCVN_BSTP_CNT_THREE_KM
교통편의성_정류장수_5km	TFCVN_BSTP_CNT_FIVE_KM
교통편의성_인접역	TFCVN_NRBSTATN
교통편의성_역_최단거리	TFCVN_STATN_SHRTDST
교통편의성_인접고속도로IC	TFCVN_NRBHWIC
교통편의성_고속도로IC_최단거리	TFCVN_HWIC_SHRTDST
생활편의시설_초등학교_최단거리	LVCNS_ELESCH_SHRTDST
생활편의시설_초등학교_수_500m	LVCNS_ELESCH_CNT_FVHD_METER
생활편의시설_초등학교_수_1km	LVCNS_ELESCH_CNT_ONE_KM
생활편의시설_초등학교_수_3km	LVCNS_ELESCH_CNT_THREE_KM
생활편의시설_초등학교_수_5km	LVCNS_ELESCH_CNT_FIVE_KM
생활편의시설_중학교_최단거리	LVCNS_MSKUL_SHRTDST
생활편의시설_중학교_수_500m	LVCNS_MSKUL_CNT_FVHD_METER
생활편의시설_중학교_수_1km	LVCNS_MSKUL_CNT_ONE_KM
생활편의시설_중학교_수_3km	LVCNS_MSKUL_CNT_THREE_KM
생활편의시설_중학교_수_5km	LVCNS_MSKUL_CNT_FIVE_KM
생활편의시설_고등학교_최단거리	LVCNS_HGSCHL_SHRTDST
생활편의시설_고등학교_수_500m	LVCNS_HGSCHL_CNT_FVHD_METER
생활편의시설_고등학교_수_1km	LVCNS_HGSCHL_CNT_ONE_KM
생활편의시설_고등학교_수_3km	LVCNS_HGSCHL_CNT_THREE_KM
생활편의시설_고등학교_수_5km	LVCNS_HGSCHL_CNT_FIVE_KM
생활편의시설_학원_최단거리	LVCNS_INSTUT_SHRTDST
생활편의시설_학원_수_500m	LVCNS_INSTUT_CNT_FVHD_METER
생활편의시설_학원_수_1km	LVCNS_INSTUT_CNT_ONE_KM
생활편의시설_학원_수_3km	LVCNS_INSTUT_CNT_THREE_KM
생활편의시설_학원_수_5km	LVCNS_INSTUT_CNT_FIVE_KM
생활편의시설_병의원_최단거리	LVCNS_CLNCHSPTL_SHRTDST
생활편의시설_병의원_수_500m	LVCNS_CLNCHSPTL_CNT_FVHD_METER
생활편의시설_병의원_수_1km	LVCNS_CLNCHSPTL_CNT_ONE_KM
생활편의시설_병의원_수_3km	LVCNS_CLNCHSPTL_CNT_THREE_KM
생활편의시설_병의원_수_5km	LVCNS_CLNCHSPTL_CNT_FIVE_KM
생활편의시설_은행_최단거리	LVCNS_BANK_SHRTDST
생활편의시설_은행_수_500m	LVCNS_BANK_CNT_FVHD_METER
생활편의시설_은행_수_1km	LVCNS_BANK_CNT_ONE_KM
생활편의시설_은행_수_3km	LVCNS_BANK_CNT_THREE_KM
생활편의시설_은행_수_5km	LVCNS_BANK_CNT_FIVE_KM
생활편의시설_마트백화점_최단거리	LVCNS_MARTDMST_SHRTDST
생활편의시설_마트백화점_수_500m	LVCNS_MARTDMST_CNT_FVHD_METER
생활편의시설_마트백화점_수_1km	LVCNS_MARTDMST_CNT_ONE_KM
생활편의시설_마트백화점_수_3km	LVCNS_MARTDMST_CNT_THREE_KM
생활편의시설_마트백화점_수_5km	LVCNS_MARTDMST_CNT_FIVE_KM
생활편의시설_치안시설_최단거리	LVCNS_PBPACFCT_SHRTDST
생활편의시설_치안시설_수_500m	LVCNS_PBPACFCT_CNT_FVHD_METER
생활편의시설_치안시설_수_1km	LVCNS_PBPACFCT_CNT_ONE_KM
생활편의시설_치안시설_수_3km	LVCNS_PBPACFCT_CNT_THREE_KM
생활편의시설_치안시설_수_5km	LVCNS_PBPACFCT_CNT_FIVE_KM
생활편의시설_편의점_최단거리	LVCNS_CVST_SHRTDST
생활편의시설_편의점_수_500m	LVCNS_CVST_CNT_FVHD_METER
생활편의시설_편의점_수_1km	LVCNS_CVST_CNT_ONE_KM
생활편의시설_편의점_수_3km	LVCNS_CVST_CNT_THREE_KM
생활편의시설_편의점_수_5km	LVCNS_CVST_CNT_FIVE_KM
유해시설_발전소_최단거리	HMFNFCT_PWPLNT_SHRTDST
유해시설_화장시설_최단거리	HMFNFCT_CMTYFCT_SHRTDST
유해시설_봉안시설_최단거리	HMFNFCT_ESRNFCT_SHRTDST
유해시설_폐기물시설_최단거리	HMFNFCT_WSTFCT_SHRTDST""".replace('\t',',').replace('\n',',').split(',')
#%%
col_name_kor_dict = {}
for i in range(len(col_name_kor)//2):
    col_name_kor_dict[col_name_kor[i*2+1]] = col_name_kor[i*2]

sep_col = '''TOTPUL_CNT
SNCTPUL_CNT
YUPUL_CNT
WKAGEPUL_CNT
TOTHOHLD_CNT
TOTBS_CNT
TOTENFSN_CNT
BSNS_CNT
BULD_CNT
HOUSE_CNT
OLDBLD_CNT
OLDHS_CNT
LDUSE_BULD
TFCVN_NRBMARD
TFCVN_NRBSMRD
TFCVN_NRBSRT
TFCVN_BSTP
LVCNS_ELESCH
LVCNS_MSKUL
LVCNS_HGSCHL
LVCNS_INSTUT
LVCNS_CLNCHSPTL
LVCNS_BANK
LVCNS_MARTDMST
LVCNS_PBPACFCT
LVCNS_CVST
HMFNFCT_PWPLNT
HMFNFCT_ESRNFCT'''.split('\n')

k_sep_col = ['총인구수', '노인인구수','유소년인구수','생산가능인구수','총가구수','총사업체수','총종사자수','생계종류',
             '건물수','주택수','노후건물수','노후주택수',
             '건물압축도',
             '대로','로','길','정류장',
             '초등학교','중학교','고등학교',
             '학원','병의원','은행','마트백화점','치안시설','편의점',
             '유해시설_소각','유해시설_매장'
             ]

for w,k in zip(sep_col, k_sep_col):
    col_name_kor_dict[w] = k



#%% 지역명
region_list = ['서울','부산','대구','인천','광주','대전','울산','세종','경기','강원','충북','충남','전북','전남','경북','경남','제주']

#%% 변수

"LVCNS_CVST_SHRTDST"
col_name_pca = [
['TOTPUL_CNT_FVHD_METER'],
['SNCTPUL_CNT_FVHD_METER'],
['YUPUL_CNT_FVHD_METER'], 
['WKAGEPUL_CNT_FVHD_METER'],
["TOTHOHLD_CNT_FVHD_METER"],
["TOTBS_CNT_FVHD_METER"],
["TOTENFSN_CNT_FVHD_METER"],
["BULD_CNT_FVHD_METER"],
["HOUSE_CNT_FVHD_METER"],
["OLDBLD_CNT_FVHD_METER"],
["OLDHS_CNT_FVHD_METER"],
["LDUSE_BULD_CMPDGR_FVHD_METER"],
["LDUSE_BULD_CMPLXDGR_FVHD_METER"],
["TFCVN_NRBMARD_ECNT_FVHD_METER","TFCVN_MARD_SHRTDST"],
["TFCVN_NRBSMRD_ECNT_FVHD_METER","TFCVN_SMRD_SHRTDST"],
["TFCVN_NRBSRT_ECNT_FVHD_METER","TFCVN_STRT_SHRTDST"],
["TFCVN_BSTP_CNT_FVHD_METER","TFCVN_BSTP_SHRTDST"],
["LVCNS_ELESCH_CNT_FVHD_METER","LVCNS_ELESCH_SHRTDST"],
["LVCNS_MSKUL_CNT_FVHD_METER","LVCNS_MSKUL_SHRTDST"],
["LVCNS_HGSCHL_CNT_FVHD_METER","LVCNS_HGSCHL_SHRTDST"],
["LVCNS_INSTUT_CNT_FVHD_METER","LVCNS_INSTUT_SHRTDST"],
["LVCNS_CLNCHSPTL_CNT_FVHD_METER","LVCNS_CLNCHSPTL_SHRTDST"],
["LVCNS_BANK_CNT_FVHD_METER","LVCNS_BANK_SHRTDST"],
["LVCNS_MARTDMST_CNT_FVHD_METER","LVCNS_MARTDMST_SHRTDST"],
["LVCNS_PBPACFCT_CNT_FVHD_METER","LVCNS_PBPACFCT_SHRTDST"],
["LVCNS_CVST_CNT_FVHD_METER","LVCNS_CVST_SHRTDST"],
['HMFNFCT_PWPLNT_SHRTDST','HMFNFCT_CMTYFCT_SHRTDST'],['HMFNFCT_ESRNFCT_SHRTDST','HMFNFCT_WSTFCT_SHRTDST']]

#%% col_SHRTDST
col_SHRTDST = [['TFCVN_MARD_SHRTDST',
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
'HMFNFCT_WSTFCT_SHRTDST'
]]
#%%
col_sorting = ['TOTPUL_CNT_FVHD_METER', 
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
'HMFNFCT_WSTFCT_SHRTDST']

#%%

    
"""    
    
'TOTPUL_CNT_FVHD_METER', 'SNCTPUL_CNT_FVHD_METER', 'YUPUL_CNT_FVHD_METER','WKAGEPUL_CNT_FVHD_METER',"TOTHOHLD_CNT_FVHD_METER", #인구 500m
"BULD_CNT_FVHD_METER","HOUSE_CNT_FVHD_METER","OLDBLD_CNT_FVHD_METER","OLDHS_CNT_FVHD_METER", #건물 500m
"LDUSE_BULD_CMPDGR_FVHD_METER","LDUSE_BULD_CMPLXDGR_FVHD_METER", #건물 정보 500m #압축도,복합도
"TOTBS_CNT_FVHD_METER","TOTENFSN_CNT_FVHD_METER", #생계 500m

# 'TOTPUL_CNT_FIVE_KM', 'SNCTPUL_CNT_FIVE_KM', 'YUPUL_CNT_FIVE_KM', 'WKAGEPUL_CNT_FIVE_KM',"TOTHOHLD_CNT_FIVE_KM", #인구 5km
# "BULD_CNT_FIVE_KM","HOUSE_CNT_FIVE_KM","OLDBLD_CNT_FIVE_KM","OLDHS_CNT_FIVE_KM", #건물 5km
# "LDUSE_BULD_CMPDGR_FIVE_KM","LDUSE_BULD_CMPLXDGR_FIVE_KM", #건물 정보 5km
# "TOTBS_CNT_FIVE_KM","TOTENFSN_CNT_FIVE_KM", #생계 5km

"BSNS_CNT_WHRTBS_FIVE_KM","BSNS_CNT_RSTLDBS_FIVE_KM","ENFSN_CNT_WHRTBS_FIVE_KM","ENFSN_CNT_RSTLDBS_FIVE_KM", #생계 종류

"TFCVN_NRBMARD_ECNT_FVHD_METER","TFCVN_NRBSMRD_ECNT_FVHD_METER","TFCVN_NRBSRT_ECNT_FVHD_METER","TFCVN_BSTP_CNT_FVHD_METER", #도로 500m

"LVCNS_ELESCH_CNT_FVHD_METER","LVCNS_MSKUL_CNT_FVHD_METER","LVCNS_HGSCHL_CNT_FVHD_METER", #교육기관 500m
"LVCNS_INSTUT_CNT_FVHD_METER","LVCNS_CLNCHSPTL_CNT_FVHD_METER","LVCNS_BANK_CNT_FVHD_METER","LVCNS_MARTDMST_CNT_FVHD_METER","LVCNS_PBPACFCT_CNT_FVHD_METER","LVCNS_CVST_CNT_FVHD_METER", #편의시설 500m

# "LVCNS_ELESCH_CNT_FIVE_KM","LVCNS_MSKUL_CNT_FIVE_KM","LVCNS_HGSCHL_CNT_FIVE_KM", #교육기관 5km
# "LVCNS_INSTUT_CNT_FIVE_KM","LVCNS_CLNCHSPTL_CNT_FIVE_KM","LVCNS_BANK_CNT_FIVE_KM","LVCNS_MARTDMST_CNT_FIVE_KM","LVCNS_PBPACFCT_CNT_FIVE_KM","LVCNS_CVST_CNT_FIVE_KM", #편의시설 5km

#도로 최단거리
'TFCVN_MARD_SHRTDST',
'TFCVN_SMRD_SHRTDST',
'TFCVN_STRT_SHRTDST',
'TFCVN_BSTP_SHRTDST',

#교육기관 최단거리
'LVCNS_ELESCH_SHRTDST',
'LVCNS_MSKUL_SHRTDST',
'LVCNS_HGSCHL_SHRTDST',

#편의시설 최단거리
'LVCNS_INSTUT_SHRTDST',
'LVCNS_CLNCHSPTL_SHRTDST',
'LVCNS_BANK_SHRTDST',
'LVCNS_MARTDMST_SHRTDST',
'LVCNS_PBPACFCT_SHRTDST',
'LVCNS_CVST_SHRTDST',

#유해시설 최단거리
'HMFNFCT_PWPLNT_SHRTDST',
'HMFNFCT_CMTYFCT_SHRTDST',
'HMFNFCT_ESRNFCT_SHRTDST',
'HMFNFCT_WSTFCT_SHRTDST'


"""