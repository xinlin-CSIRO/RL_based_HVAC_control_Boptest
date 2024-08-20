import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date
from reward_function import the_one_under_test_1, the_one_under_test_2
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from learning_paras import learning_steps_,test_steps_,learning_rate_,learning_starts_,batch_size_,start_time_
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
energy_consumption_up_bond=4000
_n_days_=300
test_typo='peak_heat_day' #'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\w_"+current_time+"_ppo_reward_1.csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor temp, boundary,boundary, action, reward, thermal_r, energy_r, themal kpi ,energy kpi, outdoor air, scenario, energy usage, target\n'
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
previous_couption=[]
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

        out_door_air = observations[4] - 273.15
        indoor_air_temp = observations[0] - 273.15

        action_ = action[0] #- 273.15
        current_consumption = observations[1]
        up_bundary = observations[2] - 273.15  # cooling is up
        low_boundary = observations[3] - 273.15


        R_, r_t,r_e, current_indoor_state,wild_boundary,target,scenario=the_one_under_test_1(low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption)
        ###############################################################


        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        # print('kpi=',energy_usage_kpi)
        result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
            action_) + ',' + str(R_) + ',' + str(r_t) + ',' + str(r_e) + ',' + str(
            thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(out_door_air) + ',' + str(scenario) +','+str(current_indoor_state) + ',' + str(
            current_consumption) + ',' + str(target) + '\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()
        previous_couption.append(current_indoor_state)
        energy_couption.append(current_consumption)

        return R_

start_time_p=start_time_()
env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump',
                                actions               = {
                                                            'oveHeaPumY_u':(0,1),# Heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity)
                                                            # 'oveFan_u':(0,1), # Integer signal to control the heat pump evaporator fan either on or off
                                                            # 'ovePum_u':(0,1), #Integer signal to control the emission circuit pump either on or off
                                                            # 'oveTSet_u':(288.15, 308.15), # 20, 26--> 293.15,303.15
                                                            #if you wanna change the boundary for the core--> C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym
                                                             # 'oveTSet_activate':(1),
                                                         },
                                observations          = {
                                    'reaTZon_y': (250, 303),
                                    'reaPHeaPum_y': (0, energy_consumption_up_bond),
                                    'reaTSetCoo_y':(250, 303),
                                    'reaTSetHea_y':(250, 303),
                                    'weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                                    # 'weaSta_reaWeaHDifHor_y': (0, 8400),
                                    # 'weaSta_reaWeaWinSpe_y': (0, 30),
                                                         },
                                # random_start_time     = True, #False,
                                start_time = start_time_p,
                                step_period= 15 * 60,
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                )

from gymnasium.wrappers import NormalizeObservation

env = NormalizeObservation(env) # dont need it if only indoor air is considered
env = DiscretizedActionWrapper(env, n_bins_act=5)
# env = DiscretizedObservationWrapper(env, n_bins_obs=5, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
print('Action space of the original agent:')
print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)

# model = my_agent (env)

from stable_baselines3 import DQN, SAC
from stable_baselines3 import A2C, PPO

learning_rates_p=learning_rate_()
# learning_start_p=learning_starts_()
# batch_size_p=batch_size_()

learning_steps_p= learning_steps_()
test_steps_p=test_steps_()
learning_rates_p=learning_rate_()
learning_start_p=learning_starts_()
batch_size_p=batch_size_()

model = PPO('MlpPolicy',
            env,
            verbose=1,
            learning_rate=learning_rates_p,
            # buffer_size=100000,
            batch_size= batch_size_p,
            # learning_starts=learning_start_p,
            # train_freq=1,
            gamma=0.99
            )

print('Learning process is started')
model.learn(total_timesteps=learning_steps_p)
print('Learning process is completed')

# Loop for one episode of experience (one day)
obs= env.reset()

for x in range(0, test_steps_p):
  action, _ = model.predict(obs, deterministic=True) # c
  obs, reward, terminated, info = env.step(action)
  if x % 10 ==0:
      print(x)
  kpis = env.get_kpis()
  energy_usage_kpi = kpis['ener_tot']
  # print('kpi=', energy_usage_kpi)
print(kpis)

f_1 = open(result_learning, "a+")
f_1.write(str(energy_usage_kpi))
f_1.close()

now = datetime.now()
time_= now.strftime("%H-%M")
print(time_)