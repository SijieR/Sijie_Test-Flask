# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import xlrd
import joblib
import math
import random
import hashlib


app = FastAPI()

class Item(BaseModel):
    age: int = 6
    gender: int = 1
    height: int = 112
    weight: int = 30
    DAI: int = 1
    FH: int = 170
    FW: int = 60
    MH: int = 160
    MW: int = 50
    vision: float = 1.20
    RA: float = 1.00
    AXL: float = 23.23

def vision_predict_for5years(x, mean_list, std_list, age):
    svr_delta1year = joblib.load(r'.\2021-03-22-RF_vision_delta_1year_Standard.pkl')
    svr_delta2year = joblib.load(r'.\2021-03-22-RF_vision_delta_2year_Standard.pkl')
    svr_delta3year = joblib.load(r'.\2021-03-22-RF_vision_delta_3year_Standard.pkl')
    svr_delta4year = joblib.load(r'.\2021-03-22-RF_vision_delta_4year_Standard.pkl')
    svr_delta5year = joblib.load(r'.\2021-03-22-RF_vision_delta_5year_Standard.pkl')
    model_list = [svr_delta1year, svr_delta2year, svr_delta3year, svr_delta4year, svr_delta5year]
    y = []
    y.append(x[0][-3])
    x_ = np.zeros((5, 13))
    vision_mean = [0.936447492, 0.824880185, 0.769592319, 0.736325625, 0.637456867, 0.549336381]
    ###需要改变，同时对对应分级的函数还需要重新复杂设计
    for ii in range(5):
        for jj in range(13):
            x_[ii][jj] = (x[ii][jj] - mean_list[jj + 1]) / std_list[jj + 1]

    for ii in range(5):
        predict_value = model_list[ii].predict(np.array(x_[ii]).reshape(1, -1))
        predict_value = predict_value * std_list[14] + mean_list[14]
        # predict_value = take_min(y[ii], predict_value*std_list[14]+mean_list[14])
        y.append(predict_value)

    '''
    '''
    vision_level = 1
    vision_score = 0
    predict_level = 0
    y_referrence = [1.25, 1.2, 1.2, 1.0, 1.0, 1.0]  ###视力-6,7,8,9,10,11,12...
    ### vision_level/predict_level 0->verygood, 1->good, 2->fair, 3->bad
    ids = math.floor(x[0][1]) - 7
    if y[0] >= 1.25:
        vision_level = 0
    elif y[0] >= 1.0:
        vision_level = 1
    elif y[0] >= 0.6:
        vision_level = 2
    elif y[0] >= 0.3:
        vision_level = 3
    else:
        vision_level = 4

    if y[0] - y[1] >= 0.25:
        predict_level = 3
    elif y[0] - y[1] <= 0.2:
        predict_level = 1
    else:
        predict_level = 2

    return y, y_referrence, vision_level, predict_level


def RA_predict_for5years(x, mean_list, std_list, age):
    svr_delta1year = joblib.load(r'.\2021-03-22-RF_RA_delta_1year_Standard.pkl')
    svr_delta2year = joblib.load(r'.\2021-03-22-RF_RA_delta_2year_Standard.pkl')
    svr_delta3year = joblib.load(r'.\2021-03-22-RF_RA_delta_3year_Standard.pkl')
    svr_delta4year = joblib.load(r'.\2021-03-22-RF_RA_delta_4year_Standard.pkl')
    svr_delta5year = joblib.load(r'.\2021-03-22-RF_RA_delta_5year_Standard.pkl')
    model_list = [svr_delta1year, svr_delta2year, svr_delta3year, svr_delta4year, svr_delta5year]
    y = []
    y.append(x[0][-2])
    x_ = np.zeros((5, 13))
    vision_mean = [0.936447492, 0.824880185, 0.769592319, 0.736325625, 0.637456867, 0.549336381]  ###xuyao xiugai
    ###需要改变，同时对对应分级的函数还需要重新复杂设计
    for ii in range(5):
        for jj in range(13):
            x_[ii][jj] = (x[ii][jj] - mean_list[jj + 1]) / std_list[jj + 1]

    for ii in range(5):
        predict_value = model_list[ii].predict(np.array(x_[ii]).reshape(1, -1))
        predict_value = predict_value * std_list[15] + mean_list[15]
        # predict_value = take_min(y[ii], predict_value*std_list[14]+mean_list[14])
        y.append(predict_value)

    '''
    '''
    vision_level = 2
    predict_level = 0
    y_referrence = []
    y_referrence_all = [1.25, 0.75, 0.25, 0.1, 0, -0.15, -0.3, -0.4, -0.55, -0.65, -0.75, -0.9, -1]
    ###6,7,8,9,10,11,12,13,14,15,16,17,18...

    ### vision_level/predict_level 1->good, 2->fair, 3->bad
    ids = math.floor(x[0][1]) - 6  ###ceritirio needed to be modified!!!!!!
    if y[0] - y_referrence_all[ids] >= 0.15:
        vision_level = 1
    elif y[0] - y_referrence_all[ids] <= 0:
        vision_level = 3
    else:
        vision_level = 2

    if y[0] - y[1] >= 0.15:
        predict_level = 3
    elif y[0] - y[1] <= 0.1:
        predict_level = 1
    else:
        predict_level = 2

    for ii in range(ids, ids + 6):
        y_referrence.append(y_referrence_all[ii])

    return y, y_referrence, vision_level, predict_level


def AXL_predict_for5years(x, mean_list, std_list, age):
    svr_delta1year = joblib.load(r'.\2021-03-22-RF_AXL_delta_1year_Standard.pkl')
    svr_delta2year = joblib.load(r'.\2021-03-22-RF_AXL_delta_2year_Standard.pkl')
    svr_delta3year = joblib.load(r'.\2021-03-22-RF_AXL_delta_3year_Standard.pkl')
    svr_delta4year = joblib.load(r'.\2021-03-22-RF_AXL_delta_4year_Standard.pkl')
    svr_delta5year = joblib.load(r'.\2021-03-22-RF_AXL_delta_5year_Standard.pkl')
    model_list = [svr_delta1year, svr_delta2year, svr_delta3year, svr_delta4year, svr_delta5year]
    y = []
    y.append(x[0][-1])
    x_ = np.zeros((5, 13))
    vision_mean = [0.936447492, 0.824880185, 0.769592319, 0.736325625, 0.637456867, 0.549336381]  ###xuyao xiugai
    ###需要改变，同时对对应分级的函数还需要重新复杂设计
    for ii in range(5):
        for jj in range(13):
            x_[ii][jj] = (x[ii][jj] - mean_list[jj + 1]) / std_list[jj + 1]

    for ii in range(5):
        predict_value = model_list[ii].predict(np.array(x_[ii]).reshape(1, -1))
        predict_value = predict_value * std_list[16] + mean_list[16]
        # predict_value = take_min(y[ii], predict_value*std_list[14]+mean_list[14])
        y.append(predict_value)

    '''
    '''
    vision_level = 2
    predict_level = 0
    ### vision_level/predict_level 1->good, 2->fair, 3->bad
    ids = math.floor(x[0][1]) - 7  ###ceritirio needed to be modified!!!!!!
    if y[0] - vision_mean[ids] >= 0.15:
        vision_level = 1
    elif y[0] - vision_mean[ids] <= -0.03:
        vision_level = 3
    else:
        vision_level = 2

    if y[0] - y[1] >= 0.4:
        predict_level = 3
    elif y[0] - y[1] <= 0.25:
        predict_level = 1
    else:
        predict_level = 2

    y_referrence = []
    for ii in range(6):
        ran = random.uniform(-1, 1)
        # print(ran)
        y_referrence.append(y[0] + 0.4 * ii + 0.1 * ran)

    return y, y_referrence, predict_level


'/timestamp={timestamp}/token={token}'
@app.get('/AIreport-test/age={age}/gender={gender}/height={height}/weight={weight}/DAI={DAI}/FH={FH}/FW={FW}/MH={MH}/MW={MW}/vision={vision}/RA={RA}/AXL={AXL}')
def evaluate(age:int=1,
              gender:int=1,
              height: int = 112,
                weight: int = 30,
                DAI: int = 1,
                FH: int = 170,
                FW: int = 60,
                MH: int = 160,
                MW: int = 50,
                vision: float = 1.20,
                RA: float = 1.00,
                AXL: float = 23.23,
                timestamp: Optional[str] = '',
                token: Optional[str] = ''):
    #exit = exit
    #c = age
    result = {
        "code": 0,
        "msg": "empty",
        "data": {}
    }
    try:
        md5_key = 'MrPupq2FkL@B$l6*7L8g24F&yZOwZKT^M7GoRS&ydKq0HiwbiY2L%MRh@05IfD'
        token_local = hashlib.md5((timestamp + md5_key).encode(encoding='utf-8')).hexdigest()
        if timestamp == '':
            result['code'] = 66
            return result
        elif not token_local == token:
            result['code'] = 88
            return result
        else:
            input_info = {'deltaT':1,
                    "age":age,
                   'gender':gender,
                   'height':height,
                   'weight':weight,
                   'DAI':DAI,
                   'FH':FH,
                   'FW':FW,
                   'MH':MH,
                   'MW':MW,
                   'vision':vision,
                   'RA':RA,
                   'AXL':AXL,
                   }
            input_info_list = list(input_info.values())
            '''
            模型输入数据x结构改变
            '''
            x = [[],[],[],[],[]]
            for ii in range(5):
                for jj in range(len(input_info_list)):
                    if jj ==0:
                        x[ii].append(input_info_list[jj]*(ii+1))
                    else:
                        x[ii].append(input_info_list[jj])

            """归一化数据"""
            MEAN_MAP = {
                'tag': 'mean',
                'deltaT': 2.5,
                'age1': 9.65527629852295,
                'GENDER': 1.4143646955490112,
                'HEIGHT_1': 137.7775421142578,
                'WEIGHT_1': 35.11663818359375,
                'DAI': 0.40032464265823364,
                'FH': 172.61509704589844,
                'FW': 74.3406982421875,
                'MH': 160.7814178466797,
                'MW': 59.31727600097656,
                'R_LUO': 0.7537265419960022,
                'RA_1': -0.05085483565926552,
                'AL_1': 23.37474822998047,
                'L_LUO': 0.7537265419960022,
                'RA_2': -0.05085483565926552,
                'AL_2': 23.37474822998047,
                'AD_1': 3.026252031326294,
                'CCT_1': 545.697265625,
                'K1_1': 42.814842224121094,
                'K2_1': 43.846710205078125,
                'White_to_White_1': 12.04364013671875,
                'grade': 2.5,
                'deltaT2': 3.5,
            }
            mean_list = list(MEAN_MAP.values())

            STD_MAP = {
                'tag': 'std',
                'deltaT': 1.7078251838684082,
                'age1': 1.7567731142044067,
                'GENDER': 0.49261200428009033,
                'HEIGHT_1': 11.735713005065918,
                'WEIGHT_1': 11.313467025756836,
                'DAI': 0.6161134243011475,
                'FH': 4.556289196014404,
                'FW': 13.932219505310059,
                'MH': 4.130949020385742,
                'MW': 11.593335151672363,
                'R_LUO': 0.29731976985931396,
                'RA_1': 1.621026635169983,
                'AL_1': 0.9859156608581543,
                'L_LUO': 0.29731976985931396,
                'RA_2': 1.621026635169983,
                'AL_2': 0.9859156608581543,
                'AD_1': 0.26024743914604187,
                'CCT_1': 29.92780113220215,
                'K1_1': 1.4588570594787598,
                'K2_1': 1.4786999225616455,
                'White_to_White_1': 0.4059635102748871,
                'grade': 1.7078251838684082,
                'deltaT2': 1.7078251838684082,
            }
            std_list = list(STD_MAP.values())

            input_info_sumdict = {
                'x':x,
                'mean_list':mean_list,
                'std_list':std_list,
                'age':age,
            }
            x = input_info_sumdict['x']
            mean_list = input_info_sumdict['mean_list']
            std_list = input_info_sumdict['std_list']
            age = input_info_sumdict['age']
            res = {}

            vision_predict, vision_referrence, vision_level, vision_predict_level = vision_predict_for5years(x, mean_list, std_list, age)
            vision_list = [0.1, 0.08, 0.06, 0.04, 0.03]  ###make sure larger than this value
            # print(max(vision_predict[0] - 1*0.01, vision_list[0]))
            if vision_predict[0] > 0.5:

                pass
            else:
                for ii in range(len(vision_predict) - 1):
                    if vision_predict[ii + 1] > vision_predict[ii]:
                        a = random.randint(1, 5)
                        vision_predict[ii + 1] = max(vision_predict[ii] - a * 0.01, np.array([vision_list[ii]]))

            print('-------------->vision')
            print(vision_referrence)
            res['vision'] = {
                'predict': {
                    'current_vision': vision_predict[0],
                    'predict_1_year': vision_predict[1][0],
                    'predict_2_year': vision_predict[2][0],
                    'predict_3_year': vision_predict[3][0],
                    'predict_4_year': vision_predict[4][0],
                    'predict_5_year': vision_predict[5][0],
                },
                'referrence': {
                    'current_referrence': vision_referrence[0],
                    'referrence_1_year': vision_referrence[1],
                    'referrence_2_year': vision_referrence[2],
                    'referrence_3_year': vision_referrence[3],
                    'referrence_4_year': vision_referrence[4],
                    'referrence_5_year': vision_referrence[5],
                },
                'vision_level': vision_level,
                'vision_predict_level': vision_predict_level,
            }

            RA_predict, RA_referrence, RA_level, RA_predict_level = RA_predict_for5years(x, mean_list, std_list, age)
            RA_list = [-6.00, -6.50, -6.75, -7.25, -8.125]  ###make sure larger than this value
            for ii in range(1, len(RA_predict)):
                RA_predict[ii] = max(RA_predict[ii], np.array([-8.125]))
                RA_referrence[ii] = max(RA_referrence[ii], -8.125)
                pass
            for ii in range(len(RA_predict) - 1):
                if RA_predict[ii + 1] > RA_predict[ii]:
                    a = random.randint(1, 5)
                    RA_predict[ii + 1] = max(RA_predict[ii] - a * 0.01, np.array([RA_list[ii]]))

            print('-------------->ra')
            print(RA_referrence)
            # current_app.logger.error('-------------->ra')
            # current_app.logger.error(RA_referrence)
            res['ra'] = {
                'predict': {
                    'current_ra': RA_predict[0],
                    'predict_1_year': RA_predict[1][0],
                    'predict_2_year': RA_predict[2][0],
                    'predict_3_year': RA_predict[3][0],
                    'predict_4_year': RA_predict[4][0],
                    'predict_5_year': RA_predict[5][0],
                },
                'referrence': {
                    'current_referrence': RA_referrence[0],
                    'referrence_1_year': RA_referrence[1],
                    'referrence_2_year': RA_referrence[2],
                    'referrence_3_year': RA_referrence[3],
                    'referrence_4_year': RA_referrence[4],
                    'referrence_5_year': RA_referrence[5],
                },
                'ra_level': RA_level,
                'ra_predict_level': RA_predict_level,
            }
            AXL_predict, AXL_referrence, AXL_predict_level = AXL_predict_for5years(x, mean_list, std_list, age)
            for ii in range(1, len(AXL_predict)):
                AXL_predict[ii] = min(AXL_predict[ii], np.array([26.82]))
                AXL_referrence[ii] = min(AXL_referrence[ii], 26.82)
                pass
            print('-------------->axl')
            print(AXL_referrence)
            res['axl'] = {
                'predict': {
                    'current_axl': AXL_predict[0],
                    'predict_1_year': AXL_predict[1][0],
                    'predict_2_year': AXL_predict[2][0],
                    'predict_3_year': AXL_predict[3][0],
                    'predict_4_year': AXL_predict[4][0],
                    'predict_5_year': AXL_predict[5][0],
                },
                'referrence': {
                    'current_referrence': AXL_referrence[0],
                    'referrence_1_year': AXL_referrence[1],
                    'referrence_2_year': AXL_referrence[2],
                    'referrence_3_year': AXL_referrence[3],
                    'referrence_4_year': AXL_referrence[4],
                    'referrence_5_year': AXL_referrence[5],
                },
                'axl_predict_level': RA_predict_level,
            }

            print(vision_predict)
            print('array1', vision_predict[1][0])
            print('array', AXL_predict[0])

            print(vision_predict)
            print(vision_referrence)

            print('--- ---')
            print(RA_predict)
            print(RA_referrence)

            print('--- ---')
            print(AXL_predict)
            print(AXL_referrence)

            print('--- ---')
            current_score = 0
            predict_score = 0

            print(vision_level)
            print(RA_level)
            current_score = current_score + 10 + 20 * vision_level + 10 * RA_level
            if current_score >= 100:
                current_score = 100

            print(vision_predict_level)
            print(RA_predict_level)
            print(AXL_predict_level)
            predict_score = predict_score + 10 + 10 * vision_predict_level + 10 * RA_predict_level + 10 * AXL_predict_level
            if predict_score >= 100:
                predict_score = 100

            y = []
            y.append(x[0][-3])
            if y[0] >= 1.25:
                vision_level = 0
                current_score = 10
                predict_score = 30
            elif y[0] >= 1.0:
                vision_level = 1
                current_score = 25
                predict_score = 40
            elif y[0] >= 0.6:
                vision_level = 2
                current_score = 60
                predict_score = 80
            elif y[0] >= 0.3:
                vision_level = 3
                current_score = 80
                predict_score = 95
            else:
                vision_level = 4
                current_score = 90
                predict_score = 95
            res['current_score'] = current_score
            res['predict_score'] = predict_score

            result['code'] = 100
            result['msg'] = 'success'
            result["data"] = res
            return result
    except:
        return result


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,  host="localhost",   port=8002,       workers=1)




