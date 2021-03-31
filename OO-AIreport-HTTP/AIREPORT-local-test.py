# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:52:13 2020

@author: landehuxi
"""


import numpy as np  # numpy库
#import sklearn
import xlrd
#import xlwt
#import os
#from sklearn import preprocessing
#from sklearn.svm import SVR
import joblib
#from sklearn.externals import joblib
#from sklearn.model_selection import GridSearchCV
import math
#import matplotlib.pyplot as plt
import random

###[[deltaT, 年龄， ],
###[]]
x1 = [[1, 7.279452055, 1, 117.4, 20.4, 1, 170, 70, 159, 51, 1.202264435, 0.875, 21.89],
      [2, 7.279452055, 1, 117.4, 20.4, 1, 170, 70, 159, 51, 1.202264435, 0.875, 21.89],
      [3, 7.279452055, 1, 117.4, 20.4, 1, 170, 70, 159, 51, 1.202264435, 0.875, 21.89],
      [4, 7.279452055, 1, 117.4, 20.4, 1, 170, 70, 159, 51, 1.202264435, 0.875, 21.89],
      [5, 7.279452055, 1, 117.4, 20.4, 1, 170, 70, 159, 51, 1.202264435, 0.875, 21.89],
      ]
x2 = [[1, 7.11, 0, 119.4, 21.8, 1, 170, 70, 158, 60, 0.25, 0.375, 21.56],
      [2, 7.11, 0, 119.4, 21.8, 1, 170, 70, 158, 60, 0.25, 0.375, 21.56],
      [3, 7.11, 0, 119.4, 21.8, 1, 170, 70, 158, 60, 0.25, 0.375, 21.56],
      [4, 7.11, 0, 119.4, 21.8, 1, 170, 70, 158, 60, 0.25, 0.375, 21.56],
      [5, 7.11, 0, 119.4, 21.8, 1, 170, 70, 158, 60, 0.25, 0.375, 21.56],
      ]
x3 = [[1, 8.11, 2, 129.4, 21.8, 1, 170, 70, 158, 60, 0.5, 0.175, 21.23],
      [2, 8.11, 2, 129.4, 21.8, 1, 170, 70, 158, 60, 0.5, 0.175, 21.23],
      [3, 8.11, 2, 129.4, 21.8, 1, 170, 70, 158, 60, 0.5, 0.175, 21.23],
      [4, 8.11, 2, 129.4, 21.8, 1, 170, 70, 158, 60, 0.5, 0.175, 21.23],
      [5, 8.11, 2, 129.4, 21.8, 1, 170, 70, 158, 60, 0.5, 0.175, 21.23],
      ]

x4 = [[1, 9.11, 2, 136.4, 21.8, 1, 170, 70, 158, 60, 0.6, 0, 21.43],
      [2, 9.11, 2, 136.4, 21.8, 1, 170, 70, 158, 60, 0.6, 0, 21.43],
      [3, 9.11, 2, 136.4, 21.8, 1, 170, 70, 158, 60, 0.6, 0, 21.43],
      [4, 9.11, 2, 136.4, 21.8, 1, 170, 70, 158, 60, 0.6, 0, 21.43],
      [5, 9.11, 2, 136.4, 21.8, 1, 170, 70, 158, 60, 0.6, 0, 21.43],
      ]
x5 = [[1, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.6, -3.7, 25.43],
      [2, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.6, -3.7, 25.43],
      [3, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.6, -3.7, 25.43],
      [4, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.6, -3.7, 25.43],
      [5, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.6, -3.7, 25.43],
      ]

x6 = [[1, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.1, -3.7, 25.43],
      [2, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.1, -3.7, 25.43],
      [3, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.1, -3.7, 25.43],
      [4, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.1, -3.7, 25.43],
      [5, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.1, -3.7, 25.43],
      ]
x7 = [[1, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.2, -3.7, 25.43],
      [2, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.2, -3.7, 25.43],
      [3, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.2, -3.7, 25.43],
      [4, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.2, -3.7, 25.43],
      [5, 10.11, 2, 146.4, 41.8, 1, 170, 70, 158, 60, 0.2, -3.7, 25.43],
      ]


def read_excel(deltaT=1, fold='train'):
    # open label excel
    workbook1 = xlrd.open_workbook(r".\AIdata2020-07-20-存储mean-std值-精简版.xls")
    # get all the sheets
    # print('process ', fold)
    # print(workbook1.sheet_names())
    sheet1_name = workbook1.sheet_names()[0]
    # get the sheet content by using sheet index
    sheet1 = workbook1.sheet_by_index(0)  # sheet index begins from [0]
    # sheet1 = workbook1.sheet_by_name('Sheet1')
    # print('info of sheet1 :', sheet1.name, sheet1.nrows, sheet1.ncols)

    feature_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    feature_mask_1 = np.zeros(20, ).astype('bool')
    for ids in feature_list:
        feature_mask_1[ids] = True

    deltaT = deltaT
    fold = fold

    mean_list = []
    std_list = []

    mean_list = sheet1.row_values(2)
    std_list = sheet1.row_values(3)
    # print(mean_list)
    # print(std_list)

    return mean_list, std_list


def vision_predict_for5years(x, mean_list, std_list, age):
    svr_delta1year = joblib.load(r'.\svr_vision_delta_1year_addRA-yanzhou.pkl')
    svr_delta2year = joblib.load(r'.\svr_vision_delta_2year_addRA-yanzhou.pkl')
    svr_delta3year = joblib.load(r'.\svr_vision_delta_3year_addRA-yanzhou.pkl')
    svr_delta4year = joblib.load(r'.\svr_vision_delta_4year_addRA-yanzhou.pkl')
    svr_delta5year = joblib.load(r'.\svr_vision_delta_5year_addRA-yanzhou.pkl')
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
    视力以对比1.2健康视力为准，以一年后下降0.2/0.4为预警信号阈值
    屈光度七岁：+0.5~+1.0， 八岁：0~+0.5D。
    屈光度以每年下降0.1~0.15为正常值参考
    眼轴以每年增加0.3~0.5mm为正常值参考
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
    svr_delta1year = joblib.load(r'.\svr_RA_delta_1year.pkl')
    svr_delta2year = joblib.load(r'.\svr_RA_delta_2year.pkl')
    svr_delta3year = joblib.load(r'.\svr_RA_delta_3year.pkl')
    svr_delta4year = joblib.load(r'.\svr_RA_delta_4year.pkl')
    svr_delta5year = joblib.load(r'.\svr_RA_delta_5year.pkl')
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
    视力以对比1.2健康视力为准，以一年后下降0.2/0.4为预警信号阈值
    屈光度七岁：+0.5~+1.0， 八岁：0~+0.5D。
    屈光度以每年下降0.1~0.15为正常值参考
    眼轴以每年增加0.3~0.5mm为正常值参考
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
    svr_delta1year = joblib.load(r'.\svr_yanzhou_delta_1year.pkl')
    svr_delta2year = joblib.load(r'.\svr_yanzhou_delta_2year.pkl')
    svr_delta3year = joblib.load(r'.\svr_yanzhou_delta_3year.pkl')
    svr_delta4year = joblib.load(r'.\svr_yanzhou_delta_4year.pkl')
    svr_delta5year = joblib.load(r'.\svr_yanzhou_delta_5year.pkl')
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
    视力以对比1.2健康视力为准，以一年后下降0.2/0.4为预警信号阈值
    屈光度七岁：+0.5~+1.0， 八岁：0~+0.5D。
    屈光度以每年下降0.1~0.15为正常值参考
    眼轴以每年增加0.3~0.5mm为正常值参考
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


def evaluate(x,a,b,age):
    mean_list = ['mean',
                 2.5,
                 9.65527629852295,
                 1.4143646955490112,
                 137.7775421142578,
                 35.11663818359375,
                 0.40032464265823364,
                 172.61509704589844,
                 74.3406982421875,
                 160.7814178466797,
                 59.31727600097656,
                 0.7537265419960022,
                 -0.05085483565926552,
                 23.37474822998047,
                 0.7537265419960022,
                 -0.05085483565926552,
                 23.37474822998047,
                 3.026252031326294,
                 545.697265625,
                 42.814842224121094,
                 43.846710205078125,
                 12.04364013671875, 2.5, 3.5]
    std_list = ['std',
                1.7078251838684082,
                1.7567731142044067,
                0.49261200428009033,
                11.735713005065918,
                11.313467025756836,
                0.6161134243011475,
                4.556289196014404,
                13.932219505310059,
                4.130949020385742,
                11.593335151672363,
                0.29731976985931396,
                1.621026635169983,
                0.9859156608581543,
                0.29731976985931396,
                1.621026635169983,
                0.9859156608581543,
                0.26024743914604187,
                29.92780113220215,
                1.4588570594787598,
                1.4786999225616455,
                0.4059635102748871,
                1.7078251838684082,
                1.7078251838684082]
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
    return res


a, b = read_excel()
#print(a)
#print(b)
res = evaluate(x7, a, b, x7[0][1])

print(res)

# x_ = np.zeros((5, 11))
# print('x_ is :', x_)
# print(x_[0][5])
input_info = {'deltaT':1,
            "age":1,
           'gender':1,
           'height':1 ,
           'weight':1,
           'DAI':1,
           'FW':1,
           'FW':1,
           'MH':1,
           'MW':1,
           'vision':1,
           'RA':1,
           'AXL':1,
           }
input_info_list = list(input_info.values())
#print(input_info_list)
x = [[],[],[],[],[]]
for ii in range(5):
    for jj in range(len(input_info_list)):
        print(jj)
        if jj ==0:
            x[ii].append(input_info_list[jj]*(ii+1))
        else:
            x[ii].append(input_info_list[jj])
#print(x)
'''
xneed = np.linspace(0, 100, 100)[:, None]
y_pre = svr.predict(xneed)# 对结果进行可视化：
plt.scatter(x, y, c='k', label='data', zorder=1)
# plt.hold(True)
plt.plot(xneed, y_pre, c='r', label='SVR_fit')
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus Kernel Ridge')
plt.legend()
plt.show()
print(svr.best_params_)
'''