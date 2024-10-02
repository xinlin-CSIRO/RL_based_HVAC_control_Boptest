import os
import requests
import numpy as np
import random
import math
import pandas as pd
from gym import spaces
from Univ_parameters import training_days, energy_consumption_up_boundary
from datetime import datetime, date
from new_reward_functions import reward_function_w_flexibility
from boptestGymEnv import NormalizedObservationWrapper
from stable_baselines3.common.callbacks import BaseCallback

os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from wrapper_xinlin import DiscretizedActionWrapper_xinlin_,ContinuousActionWrapper_xinlin, ContinuousActionWrapper_xinlin_four_actions
from learning_paras import learning_steps_,test_steps_,learning_rate_,learning_starts_,batch_size_,start_time_
from gymnasium.spaces import MultiDiscrete
url = 'https://api.boptest.net'

seed = 123456
random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(seed)
_ent_coef_='auto'
# Winter period goes from December 21 (day 355) to March 20 (day 79)
excluding_periods = [(79*24*3600, 355*24*3600)]
# Temperature setpoints

energy_consumption_up_bond=energy_consumption_up_boundary()
_n_days_=training_days()
test_typo='peak_cool_day' #'peak_cool_day' #'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
start_date=0
ahead_period=0
time_resolution=1
if test_typo=='peak_heat_day':
    start_date=334-ahead_period
elif test_typo=='peak_cool_day':
    start_date=282-ahead_period
elif test_typo=='typical_heat_day':
    start_date=44-ahead_period
elif test_typo=='typical_cool_day':
    start_date=146-ahead_period
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Air_"+test_typo+'_'+current_time+"_PID.csv"

f_1 = open(result_learning, "w+")
record='indoor_temperature,boundary_0,boundary_1,action_0,action_1,outdoor_air,cool_consumption,heating_consumption,fan_consumption,price,all_consumption,themal_kpi,energy_kpi,cost_kpi\n'
f_1.write(record)
f_1.close()
Last_energy_usage_kpi=0
counter_x=0

###########import the_one_under_test---> low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption
class BoptestGymEnvCustomReward(BoptestGymEnv):
    def compute_reward(self, action):
        R_ = 0
        return R_


env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_air', #'bestest_hydronic_heat_pump',
                                actions               = {
                                                            'con_oveTSetHea_u':(288.15, 296.15), #(15, 23)
                                                            'con_oveTSetCoo_u':(296.15, 303.15),
                                                            # 'con_oveTSetCoo_activate':(1,1)
                                                         },
                                observations          = {
                                                        'zon_reaTRooAir_y': (250, 303),     #'reaTZon_y': (250, 303),# zone air temp
                                                        'zon_weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                                                        # 'zon_weaSta_reaWeaHGloHor_y':(0, 1000),  # solar irradiation
                                                        # 'zon_weaSta_reaWeaRelHum_y':(0, 1),  # Outside relative humidity measurement
                                                        'fcu_reaPCoo_y': (0, energy_consumption_up_bond),# Cooling energy consumption
                                                        'fcu_reaPFan_y': (0, energy_consumption_up_bond),# fan energy consumption
                                                        'fcu_reaPHea_y': (0, energy_consumption_up_bond),# heating energy consumption
                                                        'LowerSetp[1]': (280., 310.),
                                                        'UpperSetp[1]': (280., 310.),
                                                        'PriceElectricPowerDynamic': (0., 1.),
                                                        # 'nTot':(0., 1.)
                                                        # 'Occupancy[1]':(0., 2.),# occupancy
                                                         },

                                random_start_time     = False,
                                predictive_period  =0,
                                start_time=start_date * 24 * 3600,
                                step_period = (60/4) * 60,####7.5 *60????????
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                excluding_periods = [(start_date*24*3600,  start_date*24*3600+14*24*3600)]
                                )

env.actions = []

print('Action space of the wrapped agent:')
print(env.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)


class BaselineModel(object):
    def __init__(self):
        pass
    def predict(self, obs, deterministic=True):
        return [], obs

class SampleModel(object):
    def __init__(self, env):
        self.env = env
        # Seed for action space
        self.env.action_space.seed(123456)
    def predict(self,obs, deterministic=True):
        return self.env.action_space.sample(), obs

learning_steps= int(2400) #learning_steps_()
test_steps= int(14*24*4) #test_steps_()



observations = env.reset()
model=SampleModel(env)
for x in range(0, test_steps):
    action, _ = model.predict(observations, deterministic=True)
    observations, reward, terminated, info = env.step(action)
    indoor_air_temp = observations[0] - 273.15
    out_door_air = observations[1] - 273.15
    cool_consumption = observations[2]
    fan_consumption = observations[3]
    heat_consumption = observations[4]

    kpis = env.get_kpis()
    energy_usage_kpi = kpis['ener_tot']
    thermal_kpi = kpis['tdis_tot']
    cost_kpi = kpis['cost_tot']

    low_boundary = observations[5] - 273.15
    up_bundary = observations[6] - 273.15
    price = observations[7]
    all_consumption = cool_consumption + heat_consumption  # + fan_consumption
    #record='indoor_temperature,boundary_0,boundary_1,action_0,action_1,outdoor_air,cool_consumption,heating_consumption,fan_consumption,price,all_consumption,themal_kpi,energy_kpi,cost_kpi, thermal_weight, energy_weight,edge\n'
    result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
        action[0]) + ',' + str(action[1]) + ',' + str(out_door_air)   + ',' + str(cool_consumption) + ',' + str(heat_consumption) + ',' + str(fan_consumption)+',' + str(price)  +  ',' + str(all_consumption) +',' + str(thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(cost_kpi) +  '\n'
    result_sting.replace('[', '').replace(']', '')
    f_1 = open(result_learning, "a+")
    f_1.write(result_sting)
    f_1.close()

    if x % 10 == 0:
        print(x)

print(kpis)
f_1 = open(result_learning, "a+")
f_1.write(str(kpis))
f_1.close()

f_1 = open(result_learning, "a+")
f_1.write(str(energy_usage_kpi))
f_1.close()

now = datetime.now()
time_ = now.strftime("%H-%M")
print(time_)