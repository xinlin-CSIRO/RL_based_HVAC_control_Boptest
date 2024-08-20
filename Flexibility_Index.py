import os
import numpy as np
import pandas as pd


source_file="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\results.xlsx"
wo_u=pd.read_excel(source_file)['all_consumption_wo_Price']
w_u=pd.read_excel(source_file)['all_consumption_w_Price']
price=pd.read_excel(source_file)['price']
c_1,c_0=0,0
outputs="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\Flexibility_.csv"

f_1 = open(outputs, "w+")
record='FI(t), Price(t),C_1(t), C_0(t)\n'
f_1.write(record)
f_1.close()

for x in range(len(wo_u)):
    c_1 += price[x] * w_u[x]
    c_0 += price[x] * wo_u[x]
    FI_x=1-c_1/c_0

    result_sting = str(FI_x) + ',' + str(price[x]) + ',' + str(c_1) + ',' + str(c_0)  + '\n'
    result_sting.replace('[', '').replace(']', '')
    f_1 = open(outputs, "a+")
    f_1.write(result_sting)
    f_1.close()


