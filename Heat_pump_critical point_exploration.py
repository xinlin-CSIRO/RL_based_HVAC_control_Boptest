import os
import requests
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

# Winter period goes from December 21 (day 355) to March 20 (day 79)
excluding_periods = [(79*24*3600, 355*24*3600)]
# Temperature setpoints

# Instantite environment
energy_consumption_up_bond=10000
out_lower_setp = -30 + 273.15
out_upper_setp =  40 + 273.15
lower_setp = 5 + 273.15
upper_setp = 10 + 273.15
_n_days_=50
test_typo='typical_heat_day' #'peak_heat_day'
# test_typo='peak_heat_day'
result_location = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\critical_point_exploration_set_temp_point"+str(_n_days_)+"_"+test_typo+"_.csv"
f_1 = open(result_location, "w+")
record='action, indoor,heat,outdoor,radiation,RH,wend_speed\n'
f_1.write(record)

class my_agent (object):
    def __init__(self, env):
        self.env = env

    def predict(self, obs, deterministic=True):
            a= np.random.choice([a for a in range(env.action_space.n)])
            return a

    def learn(self, total_episodes):
        global result_location  # <-- Add this line

        for i in range(total_episodes):
            # Initialize enviornment
            done = False
            obs= env.reset()[0]
            indoor_air_last=20
            # Print episode number and starting day from beginning of the year:
            print('-------------------------------------------------------------------')
            print('Episode number: {0}, starting day: {1:.1f} ' \
                  '(from beginning of the year)'.format(i + 1, env.unwrapped.start_time / 24 / 3600))
            while not done:


                if(indoor_air_last<20):
                    # action = np.array([1, 1, 1])

                    value=20+273.15
                    action = np.array([value])
                else:
                    # action = np.array([0, 0, 0])
                    value=10+273.15
                    action = np.array([value])
                nxt_obs, rew, done, _ = env.step(action)
                obs = nxt_obs

                print('the current observation is:', obs)
                print('action is: ', action[0] )

                indoor_air=nxt_obs[0]-273.15
                indoor_air_last=indoor_air
                heat_power=nxt_obs[1]
                out_weather=nxt_obs[2]-273.15
                DNI = nxt_obs[3]
                RH = nxt_obs[4]
                wend_speed=nxt_obs[5]
                record_s = str(action[0])+','+str(indoor_air) + ',' + str(heat_power) + ',' + str(out_weather) + ',' + str(DNI) + ',' + str(RH) + ',' + str(wend_speed) + '\n'  # 'set,return,heat, cool, outside/n'
                f_1 = open(result_location, "a+")
                f_1.write(record_s)
                # f_1.write(result_step)
                f_1.close()


env = BoptestGymEnv(            url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump',
                                actions               = {
                                                            # 'oveHeaPumY_u':(0,1),# Heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity)
                                                            # 'oveFan_u':(0,1), # Integer signal to control the heat pump evaporator fan either on or off
                                                            # 'ovePum_u':(0,1), #Integer signal to control the emission circuit pump either on or off
                                                            'oveTSet_u':(278.15, 308.15), # 5, 35
                                                         },
                                observations          = {
                                                         # 'time':(0,604800),#per step one hour
                                                         'reaTZon_y':(250,303),
                                                         'reaQFloHea_y':(0,energy_consumption_up_bond),
                                                         'weaSta_reaWeaTDryBul_y':(250,303),
                                                          'weaSta_reaWeaHDirNor_y':(0, 1000), #radiation
                                                          'weaSta_reaWeaRelHum_y':(0, 100), #Relative humidity
                                                          # 'weaSta_reaWeaCloTim_y':(0, 3600)
                                                           'weaSta_reaWeaWinSpe_y':(0, 3600) #wend speed
                                                         },
                                # random_start_time     = True, #False,
                                start_time = 15*24*3600,
                                # step_period           = 3600,
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                )

from boptestGymEnv import DiscretizedObservationWrapper
# env = NormalizedObservationWrapper(env)
# env = DiscretizedActionWrapper(env, n_bins_act=1)
# env = DiscretizedObservationWrapper(env, n_bins_obs=4, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
print('Action space of the original agent:')
print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)

model = my_agent (env)
print('Starting...')
model.learn(total_episodes=1)