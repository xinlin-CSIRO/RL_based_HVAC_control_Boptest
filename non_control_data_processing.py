import os
import pandas as pd
import numpy as np
import random
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
url = 'https://api.boptest.net'
# Instantite environment
# Seed for random starting times of episodes
seed = 123456
random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(seed)

source = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\step_no_control_365_.csv"
data=pd.read_csv(source)
# new_data=pd.DataFrame(columns=['indoor','outdoor','heat','cool'])
new_data_heat=[['indoor','outdoor','heat','cool','solar']]
new_data_cool=[['indoor','outdoor','heat','cool','solar']]
for x in range (len(data)):
    indoor=data['return'][x]
    out = data['outside'][x]
    heat_power = data['heat'][x]
    cool_power = data['cool'][x]
    solar=data['solar'][x]
    if(indoor<20 ) and (heat_power<10 and cool_power<10) and (solar>0):
        new_row = [indoor, out, heat_power, cool_power,solar]
        # new_data.append(pd.Series(new_row))
        new_data_cool.append(new_row)
    if ( indoor > 27) and (heat_power < 10 and cool_power < 10) and (solar>0):
        new_row = [indoor, out, heat_power, cool_power,solar]
        # new_data.append(pd.Series(new_row))
        new_data_heat.append(new_row)
new_data_heat=pd.DataFrame(new_data_heat)
new_data_cool=pd.DataFrame(new_data_cool)
df_reverse_rows_heat = new_data_heat.iloc[:, ::-1]
df_reverse_rows_cool = new_data_cool.iloc[:, ::-1]
output_heat = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\processed_raw_no_control_heat365.csv"
output_cool = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\processed_raw_no_control_cool365.csv"
df_reverse_rows_heat.to_csv(output_heat, index=False)
df_reverse_rows_cool.to_csv(output_cool, index=False)
print(1)