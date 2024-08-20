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
record='set,return,heat,cool,outside,solar,RH,time\n'
# Instantite environment
energy_consumption_up_bond=10000
out_lower_setp = -30 + 273.15
out_upper_setp =  40 + 273.15
lower_setp = 5 + 273.15
upper_setp = 10 + 273.15
_n_days_=14
# test_typo='typical_heat_day' #'peak_heat_day'
test_typo='peak_heat_day'
result_location = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\step_no_control_"+str(_n_days_)+"_"+test_typo+"_.csv"
f_1 = open(result_location, "w+")
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
            obs= env.reset()
            obs=obs[0]
            # Print episode number and starting day from beginning of the year:
            print('-------------------------------------------------------------------')
            print('Episode number: {0}, starting day: {1:.1f} ' \
                  '(from beginning of the year)'.format(i + 1, env.unwrapped.start_time / 24 / 3600))
            while not done:
                # Get action with epsilon-greedy policy and simulate
                # act = self.predict(obs, deterministic = False)
                action = np.array([200, 400]) #200-273
                nxt_obs, rew, done, _ = env.step(action)
                obs = nxt_obs

                # print ('nxt_obs= ', nxt_obs)
                print('the current observation is:', obs)
                print('action is: ', action[0] )
                # print('Set temp is: ', act + 293.15)
                # print('reward is: ', rew)
                supply_sp= 0 #np.array(env.val_bins_act)[0,action[0]]
                return_sp=nxt_obs[0]-273.15
                heat_power=nxt_obs[1]
                cool_power=nxt_obs[2]
                out_weather=nxt_obs[3]-273.15
                DNI = nxt_obs[4]
                RH = nxt_obs[5]
                TIME = nxt_obs[6]
                record_s = str(supply_sp) + ',' + str(return_sp) + ',' + str(heat_power) + ',' + str(
                    cool_power) + ',' + str(out_weather) + ',' + str(DNI) + ',' + str(RH) + ',' + str(TIME) + '\n'  # 'set,return,heat, cool, outside/n'
                f_1 = open(result_location, "a+")
                f_1.write(record_s)
                # f_1.write(result_step)
                f_1.close()


env = BoptestGymEnv(            url                   = url,
                                testcase              = 'bestest_air',
                                actions               = {
                                                            # 'fcu_oveTSup_u':(lower_setp,upper_setp),
                                                            'con_oveTSetHea_u':(200,400), # [K] [min=288.15, max=296.15]: Zone temperature setpoint for heating
                                                            'con_oveTSetCoo_u':(200,400), # [K] [min=296.15, max=303.15]: Zone temperature setpoint for cooling
                                                         },
                                observations          = {
                                                         # 'time':(0,604800),#per step one hour
                                                         'zon_reaTRooAir_y':(250,303),
                                                         'fcu_reaPHea_y':(0,energy_consumption_up_bond),
                                                         'fcu_reaPCoo_y':(0,energy_consumption_up_bond),
                                                         'zon_weaSta_reaWeaTDryBul_y':(250,303),
                                                          'zon_weaSta_reaWeaHDirNor_y':(0, 1000),
                                                          'zon_weaSta_reaWeaRelHum_y':(0, 100),
                                                          'zon_weaSta_reaWeaCloTim_y':(0, 3600)
                                                         },
                                random_start_time     = True, #False,
                                scenario=test_typo,
                                # excluding_periods     = excluding_periods,
                                max_episode_length    = _n_days_*24*3600,
                                warmup_period         = _n_days_*24*3600,
                                step_period           = 3600,
                                # render_episodes       = True
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