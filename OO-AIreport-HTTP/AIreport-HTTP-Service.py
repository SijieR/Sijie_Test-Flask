# -*- coding: utf-8 -*-

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
app = FastAPI()

class list_(BaseModel):
    list_ :List[float] = None

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



@app.get('/test11/age={age}/gender={gender}/height={height}/weight={weight}/DAI={DAI}/FH={FH}/FW={FW}/MH={MH}/MW={MW}/vision={vision}/RA={RA}/AXL={AXL}')
def calculate(age:int=1,
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
                AXL: float = 23.23):
    #exit = exit
    #c = age
    input_info = {'deltaT':1,
            "age":age,
           'gender':gender,
           'height':height,
           'weight':weight,
           'DAI':DAI,
           'FW':FH,
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
    return input_info_sumdict



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,  host="localhost",   port=8001,       workers=1)




