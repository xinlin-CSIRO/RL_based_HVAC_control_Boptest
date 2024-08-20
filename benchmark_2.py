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
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\benchmark_DQN_"+current_time+"_different_reward.csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor temp, boundary,boundary, action, reward, outdoor air,  themal kpi ,energy kpi\n'
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
kpi_discomfort=[]
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
        in_door_air = observations[0] - 273.15
        up=observations[2] - 273.15
        low=observations[3] - 273.15
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        kpi_discomfort.append(thermal_kpi)
        if(len(kpi_discomfort)>1):
            previous_discomfot=kpi_discomfort[-2]
        else:
            previous_discomfot=0



        reward = 10 *  (previous_discomfot - thermal_kpi)


        # reward =  thermal_r

        action_=action[0] #-273.15
        result_sting = str(in_door_air) +','+str(up)+ ','+str(low) + ','+ str(action_)+','+str(reward)+','+ str(out_door_air) + ',' + str(thermal_kpi)  +','+str(energy_usage_kpi)+'\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()

        return reward

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

from stable_baselines3 import DQN,A2C, PPO,DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# n_actions=env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)




learning_rates_p=learning_rate_()
learning_start_p=learning_starts_()
batch_size_p=batch_size_()
model = DQN('MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.1,
            buffer_size=100000,
            batch_size= 24,
            learning_starts=350,
            train_freq=1,
            gamma=0.99
            )

# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=0.001,
#             )

learning_steps_p= learning_steps_()
learning_steps=learning_steps_p
test_steps_p=test_steps_()
test_steps=test_steps_p

print('Learning process is started')
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')

# Loop for one episode of experience (one day)
done = False
obs= env.reset()


for x in range(0, test_steps):
  action, _ = model.predict(obs, deterministic=True) # c
  # action, _ = model.predict(obs)  # c
  # a=env.step(action)
  # obs,reward,terminated,truncated,info = env.step(action)
  # initial_action_space=env.val_bins_act[0]
  # action_back=initial_action_space[action]

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