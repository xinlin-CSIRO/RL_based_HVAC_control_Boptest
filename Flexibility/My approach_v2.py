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
test_typo='peak_heat_day' #'peak_cool_day' #'typical_heat_day' #'peak_heat_day'
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
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Air_"+test_typo+'_'+current_time+"_my_approach_2.csv"

f_1 = open(result_learning, "w+")
record='indoor_temperature,boundary_0,boundary_1,action_0,action_1,reward,thermal_r,energy_r,outdoor_air,scenario,ref,cool_consumption,heating_consumption,fan_consumption,price,predicted_price,all_consumption,themal_kpi,energy_kpi,cost_kpi, thermal_weight, energy_weight,edge\n'
f_1.write(record)
f_1.close()
Last_energy_usage_kpi=0
counter_x=0
class BoptestGymEnvCustomReward(BoptestGymEnv):

    def compute_reward(self, action):
        # a= self.actions
        u = {}
        # Assign values to inputs if any
        for i, act in enumerate(self.actions):
            # Assign value
            u[act] = action[i]
            # Indicate that the input is active
            u[act.replace('_u', '_activate')] = 1.
        res = requests.post('{0}/advance/{1}'.format(self.url, self.testid), data=u).json()['payload']
        observations = self.get_observations(res)

        indoor_air_temp = observations[0] - 273.15
        out_door_air = observations[1] - 273.15
        # solar = observations[2]
        # humidity = observations[3]

        cool_consumption = observations[2]
        fan_consumption = observations[3]
        heat_consumption = observations[4]
        low_boundary = observations[5] - 273.15
        up_bundary = observations[6] - 273.15
        price = observations[7]
        # occupancy= observations[10]

        action_0 = action[0] - 273.15
        action_1 = action[1] - 273.15
        all_consumption = cool_consumption + heat_consumption + fan_consumption
        reward_function = reward_function_w_flexibility
        global counter_x
        R_, r_t,r_e,target,scenario,predicted_price,thermal_weight, energy_weight, edge = reward_function (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,time_resolution)
        counter_x = (counter_x + 1) % 48

        ###############################################################

        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        cost_kpi = kpis['cost_tot']
        # print('kpi=',energy_usage_kpi)
        result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
            action_0) + ',' + str(action_1) + ',' + str(R_) + ',' + str(r_t) + ',' + str(r_e) + ',' + str(
            out_door_air) + ',' + str(scenario) + ',' + str(target)  + ',' + str(cool_consumption) + ',' + str(heat_consumption) + ',' + str(
            fan_consumption) + ',' + str(price) + ',' + str(predicted_price) + ',' + str(all_consumption) + ',' + str(
            thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(cost_kpi) + ',' + str(thermal_weight) + ',' + str(
            energy_weight) + ',' + str(edge) +'\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()
        return R_




# start_to_train=start_date-100
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
                                step_period = (30/time_resolution) * 60,####7.5 *60????????
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                excluding_periods = [(start_date*24*3600,  start_date*24*3600+14*24*3600)]
                                )

from gymnasium.wrappers import NormalizeObservation

env = NormalizeObservation(env) # dont need it if only indoor air is considered
env = ContinuousActionWrapper_xinlin(env)
# env = DiscretizedObservationWrapper(env, n_bins_obs=5, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
# print('Action space of the original agent:')
# print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)



from stable_baselines3 import DQN,A2C, PPO, SAC, DDPG


model = SAC("MlpPolicy", env, learning_rate=0.0003, batch_size =int(128*time_resolution),  ent_coef=_ent_coef_,learning_starts=2000  ) # ent_coef=0.2
# model = SAC("MlpPolicy", env, learning_rate=0.0005, batch_size =int(24*time_resolution),  ent_coef=_ent_coef_ ) # ent_coef=0.2
# model = DDPG("MlpPolicy", env, learning_rate=0.0003, batch_size =int(128*time_resolution),learning_starts=500 ) # ent_coef=0.2

learning_steps = int(1*time_resolution*24*_n_days_) # (3 weeks--> 5 weeks....)
test_steps = int(1*time_resolution*24*(ahead_period+14))

print('Learning process is started')
# model.learn(total_timesteps=learning_steps, callback=callback)
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')
model.save("sac_v2")

# Loop for one episode of experience (one day)
done = False
obs= env.reset()


for x in range(0, test_steps):
  action, _ = model.predict(obs, deterministic=True) # c
  obs, reward, terminated, info = env.step(action)
  if x % 10 ==0:
      print("current step:", x)

kpis = env.get_kpis()
energy_usage_kpi = kpis['ener_tot']
print('kpi=', energy_usage_kpi)

print(kpis)

f_1 = open(result_learning, "a+")
f_1.write(str(kpis))
f_1.close()

now = datetime.now()
time_= now.strftime("%H-%M")
print(time_)