import os
import requests
import numpy as np
import random
import math
from gym import spaces
from datetime import datetime, date
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from wrapper_xinlin import DiscretizedActionWrapper_xinlin_,ContinuousActionWrapper_xinlin
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
import math
def optimized_outdoor_diff(x):
    beta =1
    delta=17.37
    return 1/(1+math.exp(-beta *(x-delta)))
def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a

def reward_(input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a
def reward_function_thermal_only (low_boundary, up_boundary, indoor_temp, outdoor_temp):


    target =  ((up_boundary + low_boundary) / 2)


    diff_curr = abs(indoor_temp - target)


    if (low_boundary < indoor_temp < up_boundary):
            R_ = reward_(diff_curr)

    else:
            R_ = penality_(diff_curr)


    return R_, target
energy_consumption_up_bond=4000
_n_days_=100
test_typo='peak_heat_day' #'peak_cool_day' #'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Air"+test_typo+'_'+current_time+"_comparsion_with_Newcastle_gym.csv"

f_1 = open(result_learning, "w+")
record='indoor_temperature,boundary_0,boundary_1,action_0,action_1,reward,outdoor_air,ref,themal_kpi,energy_kpi,cost_kpi\n'
f_1.write(record)
f_1.close()
Last_energy_usage_kpi=0
counter=0

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
        outdoor_air = observations[1] - 273.15
        low_boundary = observations[2] - 273.15
        up_bundary = observations[3] - 273.15


        action_0 = action[0] - 273.15
        action_1 = action[1] - 273.15

        R_, target = reward_function_thermal_only (low_boundary, up_bundary, indoor_air_temp, outdoor_air)
        ###############################################################

        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        cost_kpi = kpis['cost_tot']
        # print('kpi=',energy_usage_kpi)
        result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
            action_0) + ',' + str(action_1) + ',' + str(R_) + ','  + str(outdoor_air) + ',' + str(target)  + ',' + str( thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(cost_kpi)   + '\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()
        return R_

start_date=0
ahead_period=0
time_resolution=1
if test_typo=='peak_heat_day':
    start_date=334-ahead_period
elif test_typo=='peak_cool_day':
    start_date=282-ahead_period

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
                                                        'LowerSetp[1]': (280., 310.),
                                                        'UpperSetp[1]': (280., 310.),
                                                         },

                                random_start_time     = False,
                                predictive_period  =0,
                                start_time=start_date * 24 * 3600,
                                step_period = (30/time_resolution) * 60,####7.5 *60????????
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*(24*time_resolution)*3600,
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



from stable_baselines3 import DQN,A2C, PPO, SAC,DDPG


model = SAC("MlpPolicy", env, learning_rate=0.001, batch_size =int(24*time_resolution),  ent_coef=_ent_coef_ ) # ent_coef=0.2


learning_steps = int(1*time_resolution*24*_n_days_) # (3 weeks--> 5 weeks....)
test_steps = int(1*time_resolution*24*(ahead_period+10))

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
f_1.write(str(kpis))
f_1.close()

now = datetime.now()
time_= now.strftime("%H-%M")
print(time_)

