import os
import requests
import numpy as np
import random
import math
from gym import spaces
from stable_baselines3 import DDPG
from datetime import datetime, date
from reward_function import reward_function_w_flexibility_1,the_one_under_test_best_air_1, the_one_under_test_best_air_2,the_one_under_test_best_air_3, the_one_under_test_best_air_price, reward_function_w_flexibility_2,reward_function_w_flexibility_3
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from wrapper_xinlin import DiscretizedActionWrapper_xinlin_,ContinuousActionWrapper_xinlin
from learning_paras import learning_steps_,test_steps_,learning_rate_,learning_starts_,batch_size_,start_time_
from gymnasium.spaces import MultiDiscrete
url = 'https://api.boptest.net'

# url = 'http://127.0.0.1:5000'
# Instantite environment
# Seed for random starting times of episodes
seed = 123456
random.seed(seed)
# Seed for random exploration and epsilon-greedy schedule
np.random.seed(seed)

# Winter period goes from December 21 (day 355) to March 20 (day 79)
excluding_periods = [(79*24*3600, 355*24*3600)]
# Temperature setpoints

# Instantite environment
energy_consumption_up_bond=4000
_n_days_=300
test_typo='typical_heat_day' #'typical_cool_day' #'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")


result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\RL_w_Price_"+current_time+"_DDPG.csv"


f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor temp,boundary,boundary,action_0, action_1,reward, thermal_r, energy_r, themal kpi ,energy kpi, outdoor air, scenario,ref,cool_consumption,heating consumption,fan consumption,Supply_air_mass_flow_rate,price,predicted price,all_consumption\n'
f_1.write(record)
f_1.close()
# f_2.write(record)
# f_2.close()
Last_energy_usage_kpi=0
counter=0

def predictor_():
    return (0)

def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_ (input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a


energy_couption=[]
couption_all_3, couption_cooling_3,couption_heating_3, couption_fun_3=[],[],[],[]
###########import the_one_under_test---> low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption
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
        out_door_air = 0 #observations[1] - 273.15
        cool_consumption = observations[1]
        fan_consumption = observations[2]
        heat_consumption = observations[3]

        Supply_air_mass_flow_rate = 0 #observations[5]
        # Occupancy = observations[6]
        low_boundary =observations[4] - 273.15 #  action[0] - 273.15
        up_bundary = observations[5]  - 273.15 # action[1] - 273.15
        price = observations[6]
        action_0 = action[0] - 273.15
        action_1 = action[1] - 273.15

        # cool_consumption, heat_consumption, fan_consumption = cool_consumption * price, heat_consumption * price, fan_consumption * price

        couption_cooling_3.append(cool_consumption)
        couption_heating_3.append(heat_consumption)
        couption_fun_3.append(fan_consumption)
        all_consumption = cool_consumption + heat_consumption + fan_consumption
        couption_all_3.append(all_consumption)

        R_, r_t,r_e,target,scenario,predicted_price,occupied_period,load_shaping_mode = reward_function_w_flexibility_1  (low_boundary, up_bundary,indoor_air_temp,out_door_air, price,cool_consumption,  heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3, couption_fun_3, couption_all_3)
        ###############################################################


        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        # print('kpi=',energy_usage_kpi)
        result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
            action_0) + ',' + str(action_1)+','+str(R_) + ',' + str(r_t) + ',' + str(r_e) + ',' + str(
            thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(out_door_air) + ',' + str(scenario) + ','  + str(target) + \
                        ',' + str(cool_consumption)  +  ',' + str(heat_consumption)  + ','+str(fan_consumption)+','+str(Supply_air_mass_flow_rate)+',' + str(price)+',' + str(predicted_price)+','+str(all_consumption)+ '\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()

        return R_

env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_air', #'bestest_hydronic_heat_pump',
                                actions               = {
                                                            # 'fcu_oveFan_u':(0,1),# Heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity)

                                                          #if you wanna change the boundary for the core--> C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym
                                                            # 'fcu_oveTSup_u':(289.15, 303.15) #(12, 40)
                                                            'con_oveTSetHea_u':(288.15, 296.15), #(15, 23)
                                                            'con_oveTSetCoo_u':(296.15, 303.15)
                                                         },
                                observations          = {
                                                        'zon_reaTRooAir_y': (250, 303),     #'reaTZon_y': (250, 303),# zone air temp
                                                        # 'zon_weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                                                        'fcu_reaPCoo_y': (0, energy_consumption_up_bond),# Cooling energy consumption
                                                        'fcu_reaPFan_y': (0, energy_consumption_up_bond),# fan energy consumption
                                                        'fcu_reaPHea_y': (0, energy_consumption_up_bond),# heating energy consumption
                                                        # 'fcu_reaFloSup_y': (0., 1.),
                                                        # 'cloTim': (0, energy_consumption_up_bond),
                                                        # 'zon_reaPPlu_y': (0, energy_consumption_up_bond),
                                                        'LowerSetp[1]': (280., 310.),
                                                        'UpperSetp[1]': (280., 310.),
                                                        'PriceElectricPowerDynamic': (0., 1.)
                                                         },

                                # random_start_time     = True, #False,
                                # regressive_period    = 2*15 * 60,
                                predictive_period  =0,
                                step_period= 30 * 60,
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                )

from gymnasium.wrappers import NormalizeObservation

env = NormalizeObservation(env) # dont need it if only indoor air is considered
# env = DiscretizedActionWrapper(env, n_bins_act=10)
N = [4, 2]
# env = DiscretizedActionWrapper_xinlin_(env, N) #Xinlin_descretized_action_wrapper is used for multi-demensional actions game--established 2024 Jan 10
env = ContinuousActionWrapper_xinlin(env)
# env = DiscretizedObservationWrapper(env, n_bins_obs=5, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
# print('Action space of the original agent:')
# print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)


# print("Action Space:", env.action_space)
# print("Action Space Type:", type(env.action_space))
# print("Action Space Shape:", env.action_space.shape)
# print("Action Space High Bound:", env.action_space.high)
# print("Action Space Low Bound:", env.action_space.low)


from stable_baselines3 import DQN,A2C, PPO, SAC,DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))


# model = DQN('MlpPolicy',
#             env,
#             verbose=1,
#             learning_rate=learning_rates_p,
#             # buffer_size=100000,
#             batch_size= batch_size_p,
#             learning_starts=learning_start_p,
#             # train_freq=1,
#             # gamma=0.99
#             )
#
# model = A2C('MlpPolicy', env, learning_rate=0.1, n_steps=1,ent_coef=0.2  )
# model = SAC("MlpPolicy", env, learning_rate=0.01, batch_size = 24, ent_coef=0.1 )
model = DDPG("MlpPolicy", env,  learning_rate=0.01, action_noise=action_noise, verbose=1)


# model = PPO('MlpPolicy', env, learning_rate=0.1
#              )

# model = DDPG('MlpPolicy', env, verbose=1)


learning_steps= 2400 #learning_steps_()
test_steps= 240 #test_steps_()

print('Learning process is started')
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')

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
f_1.write(str(energy_usage_kpi))
f_1.close()

now = datetime.now()
time_= now.strftime("%H-%M")
print(time_)