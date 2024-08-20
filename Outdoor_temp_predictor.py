import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
import time
import os
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from attention import Attention
from keras import Input
from keras.models import load_model, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))
def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
def smape(act,forc):
    return 100/len(act) * np.sum(2 * np.abs(forc - act) / (np.abs(act) + np.abs(forc)))
def tic (act,forc):
    leng=len(act)
    useless=np.zeros(leng)
    upper= mae(act,forc)
    lower=mae(act,useless)+mae(forc,useless)
    return upper/lower


def ONE_STEP_AHEAD_predictor (for_test, trainX, trainY):

    # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    # trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))
    alpha = 0.8
    trainX_2D = np.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
    trainY_2D  = trainY #np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
    # trainY_2D = (trainY_2D[:, 0]).reshape(-1, 1)
    model_ML = GradientBoostingRegressor(loss='squared_error',  n_estimators=5)   #, alpha=alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    # make a one-step prediction
    for_test_1D = np.reshape(for_test, (1, 2))
    y_ = model_ML.predict(for_test_1D)
    #################################
    # model_ML.set_params(alpha=1.0 - alpha)
    # model_ML.fit(trainX_2D, trainY_2D)
    # y_lower = model_ML.predict(for_test_1D)
    # if(y_upper>y_lower):
    #     o_upper=y_upper
    #     o_lower=y_lower
    # else:
    #     o_upper = y_lower
    #     o_lower = y_upper
    return (y_)

def LSTM_ATTENTION_PREDICTOR (for_test, trainX, trainY):
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    for_test=np.reshape(for_test,(1,2))
    look_forward = 1

    n_samples = trainX.shape[0]
    time_step = trainX.shape[1]
    n_layers = n_samples

    loss_ = 'mean_squared_error'
    neurals = time_step
    model_input = Input(batch_input_shape=(n_samples, time_step, 1))
    x = LSTM(neurals, return_sequences=True)(model_input)
    x = Attention(units=time_step)(x)
    x = Dense(50 * look_forward)(x)
    x = Dense(5 * look_forward)(x)
    x = Dense(look_forward)(x)
    model = Model(model_input, x)
    model.compile(loss=loss_, optimizer='adam')
    # print(model.summary())
    model.fit(trainX, trainY, epochs=100, batch_size=n_samples, verbose=0)

    # re-define model
    model_input_new = Input(batch_input_shape=(1, time_step, 1))
    x_new = LSTM(neurals, return_sequences=True)(model_input_new)
    x_new = Attention(units=time_step)(x_new)
    x_new = Dense(50 * look_forward)(x_new)
    x_new = Dense(5 * look_forward)(x_new)
    x_new = Dense(look_forward)(x_new)
    model_new = Model(model_input_new, x_new)

    # copy weights
    old_weights = model.get_weights()
    model_new.set_weights(old_weights)
    model_new.compile(loss=loss_, optimizer='adam')
    current_24_h = np.reshape(for_test, (for_test.shape[0], for_test.shape[1], 1))
    pred1 = model_new.predict(current_24_h)[0][0]
    return (pred1)

def rf_predictor (for_test, trainX, trainY):
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    for_test=np.reshape(for_test,(1,2))

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(trainX, trainY)
    pred1= rf.predict(for_test)
    return (pred1)

target = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\step_no_control_50_typical_heat_day_.csv"
data=pd.read_csv(target)

Training_Y=np.array(data['outside'])
Training_X=np.array(data[['outside','wend_speed']])

start_point=200
training_length=200

address = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\predictions_all_pro.csv"
f_1 = open(address, "w+")
record='real,XGBoost, adjusted_XGBoost, LSTM+att, adjusted_LSTM+att,rf,rf_adjust\n'
f_1.write(record)

prediction_results_XGB=[]
adjusted_prediction_results_XGB=[]

prediction_results_LSTM=[]
adjusted_prediction_results_LSTM=[]

prediction_results_RF=[]
adjusted_prediction_results_RF=[]

real_test_temp=[]
for x in range(start_point,len(data)-1):
    head=x-training_length
    tail=x
    onling_training_x = Training_X[head:tail]
    onling_training_y = Training_Y[(head+1):(tail+1)]
    for_test=Training_X[(tail)]
    real_load=Training_Y[(tail+1)]
    real_last=Training_Y[(tail)]

    result_1= ONE_STEP_AHEAD_predictor(for_test,onling_training_x, onling_training_y )[0]
    result_3 = rf_predictor(for_test, onling_training_x, onling_training_y)[0]
    result_2=LSTM_ATTENTION_PREDICTOR(for_test,onling_training_x, onling_training_y )

    if(x>(start_point+1)):
        error_1 =   real_last - prediction_results_XGB[-1]
        error_2 =   real_last - prediction_results_LSTM[-1]
        error_3 =   real_last - prediction_results_RF[-1]
        # print(1)
    else:
        error_1 = 0
        error_2 = 0
        error_3 = 0

    prediction_results_XGB.append(result_1)
    prediction_results_LSTM.append(result_2)
    prediction_results_RF.append(result_3)

    adjusted_result_1=result_1+error_1
    adjusted_prediction_results_XGB.append(adjusted_result_1)

    adjusted_result_2 = result_2 + error_2
    adjusted_prediction_results_LSTM.append(adjusted_result_2)

    adjusted_result_3 = result_3 + error_3
    adjusted_prediction_results_RF.append(adjusted_result_3)

    real_test_temp.append(real_load)

    result_sting= str(real_load) + ','  + str(result_1) + ','  + str(adjusted_result_1) + ','  + str(result_2) + ','  + str(adjusted_result_2) + ','+str(result_3) + ','  + str(adjusted_result_3) + '\n'
    result_sting.replace('[','').replace(']','')
    f_1 = open(address, "a+")
    f_1.write(result_sting)
    f_1.close()
print(1)

prediction_results=np.array(prediction_results_XGB)
adjusted_prediction_results=np.array(adjusted_prediction_results_XGB)
real_test_temp=np.array(real_test_temp)
prediction_results_LSTM=np.array(prediction_results_LSTM)
adjusted_prediction_results_LSTM=np.array(adjusted_prediction_results_LSTM)
prediction_results_RF=np.array(prediction_results_RF)
adjusted_prediction_results_RF=np.array(adjusted_prediction_results_RF)

rmse_1 = rmse(real_test_temp, prediction_results)
r_2_1 = r2_score(real_test_temp, prediction_results)
mape_1 = mape(real_test_temp, prediction_results)
mae_1 = mae(real_test_temp, prediction_results)
smape_1 = smape(real_test_temp, prediction_results)
tic_1 = tic(real_test_temp, prediction_results)

rmse_2 = rmse(real_test_temp, adjusted_prediction_results)
r_2_2 = r2_score(real_test_temp, adjusted_prediction_results)
mape_2 = mape(real_test_temp, adjusted_prediction_results)
mae_2 = mae(real_test_temp, adjusted_prediction_results)
smape_2 = smape(real_test_temp, adjusted_prediction_results)
tic_2 = tic(real_test_temp, adjusted_prediction_results)

rmse_3 = rmse(real_test_temp, prediction_results_LSTM)
r_2_3 = r2_score(real_test_temp, prediction_results_LSTM)
mape_3 = mape(real_test_temp, prediction_results_LSTM)
mae_3 = mae(real_test_temp, prediction_results_LSTM)
smape_3 = smape(real_test_temp, prediction_results_LSTM)
tic_3= tic(real_test_temp, prediction_results_LSTM)

rmse_4 = rmse(real_test_temp, adjusted_prediction_results_LSTM)
r_2_4 = r2_score(real_test_temp, adjusted_prediction_results_LSTM)
mape_4 = mape(real_test_temp, adjusted_prediction_results_LSTM)
mae_4 = mae(real_test_temp, adjusted_prediction_results_LSTM)
smape_4 = smape(real_test_temp, adjusted_prediction_results_LSTM)
tic_4 = tic(real_test_temp, adjusted_prediction_results_LSTM)

rmse_5 = rmse(real_test_temp, prediction_results_RF)
r_2_5 = r2_score(real_test_temp, prediction_results_RF)
mape_5 = mape(real_test_temp, prediction_results_RF)
mae_5 = mae(real_test_temp, prediction_results_RF)
smape_5 = smape(real_test_temp, prediction_results_RF)
tic_5= tic(real_test_temp, prediction_results_RF)

rmse_6 = rmse(real_test_temp, adjusted_prediction_results_RF)
r_2_6 = r2_score(real_test_temp, adjusted_prediction_results_RF)
mape_6 = mape(real_test_temp, adjusted_prediction_results_RF)
mae_6 = mae(real_test_temp, adjusted_prediction_results_RF)
smape_6 = smape(real_test_temp, adjusted_prediction_results_RF)
tic_6 = tic(real_test_temp, adjusted_prediction_results_RF)


s = str(rmse_1) + ',' + str(r_2_1) + ',' + str(mape_1)+ ',' + str(mae_1) + ',' + str(smape_1)+ ',' + str(tic_1)+ '\n'
print (s)

s = str(rmse_2) + ',' + str(r_2_2) + ',' + str(mape_2)+ ',' + str(mae_2) + ',' + str(smape_2)+ ',' + str(tic_2)+ '\n'
print (s)

s = str(rmse_3) + ',' + str(r_2_3) + ',' + str(mape_3)+ ',' + str(mae_3) + ',' + str(smape_3)+ ',' + str(tic_3)+ '\n'
print (s)

s = str(rmse_4) + ',' + str(r_2_4) + ',' + str(mape_4)+ ',' + str(mae_4) + ',' + str(smape_4)+ ',' + str(tic_4)+ '\n'
print (s)

s = str(rmse_5) + ',' + str(r_2_5) + ',' + str(mape_5)+ ',' + str(mae_5) + ',' + str(smape_5)+ ',' + str(tic_5)+ '\n'
print (s)

s = str(rmse_6) + ',' + str(r_2_6) + ',' + str(mape_6)+ ',' + str(mae_6) + ',' + str(smape_6)+ ',' + str(tic_6)+ '\n'
print (s)