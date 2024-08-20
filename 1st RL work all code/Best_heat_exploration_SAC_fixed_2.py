import os
import requests
import numpy as np
import random
import math
from gym import spaces
from stable_baselines3 import DDPG
from datetime import datetime, date
from reward_function import the_one_under_test_1,the_one_under_test_1025_best,the_one_under_test_outdoor_comparsion_fixed,the_one_under_test_2_refined,the_one_under_test_1_further,the_one_under_test_outdoor,the_one_under_test_outdoor_2
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"/")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from wrapper_xinlin import ContinuousActionWrapper_xinlin,ContinuousActionWrapper_xinlin_single_action,ContinuousActionWrapper_xinlin_single_action_setpoint_15_30,ContinuousActionWrapper_xinlin_single_action_setpoint_5_30
from learning_paras import learning_steps_,test_steps_,learning_rate_,learning_starts_,batch_size_,start_time_
from gymnasium.spaces import MultiDiscrete
url = 'https://api.boptest.net'
seed = 123456
random.seed(seed)
np.random.seed(seed)

# Winter period goes from December 21 (day 355) to March 20 (day 79)
excluding_periods = [(79*24*3600, 355*24*3600)]
# Temperature setpoints
alpa=0.5
# Instantite environment
energy_consumption_up_bond=4000
_n_days_=300
test_scenario = 'peak_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Best_heat_our_method_"+current_time+"_"+test_scenario+"_"+str(alpa)+".csv"

f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor temp,boundary,boundary,action,reward,thermal_r,energy_r,themal kpi,energy kpi,outdoor air,scenario,target,energy usage\n'
f_1.write(record)
f_1.close()

energy_couption=[]
class BoptestGymEnvCustomReward(BoptestGymEnv):

    def compute_reward(self, action):
        u = {}
        for i, act in enumerate(self.actions):
            # Assign value
            u[act] = action[i]
            u[act.replace('_u', '_activate')] = 1.
        res = requests.post('{0}/advance/{1}'.format(self.url, self.testid), data=u).json()['payload']
        observations = self.get_observations(res)

        indoor_air_temp = observations[0] - 273.15
        out_door_air = observations[1] - 273.15
        heat_consumption = observations[2]
        low_boundary = observations[3] - 273.15
        up_bundary = observations[4] - 273.15
        action_ = action[0] - 273.15
        # energy_couption.append(heat_consumption)

        R_, r_t, r_e, current_indoor_state, thermal_weight, target, scenario = the_one_under_test_outdoor_comparsion_fixed(
                                                                                                                        low_boundary,
                                                                                                                        up_bundary,
                                                                                                                        indoor_air_temp,
                                                                                                                        heat_consumption,
                                                                                                                        out_door_air,
                                                                                                                        energy_couption,
                                                                                                                        alpa
                                                                                                                        )
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi = kpis['tdis_tot']
        cost_kpi = kpis['cost_tot']
        # print('kpi=',energy_usage_kpi)
        result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
            action_) + ',' + str(R_) + ',' + str(r_t) + ',' + str(r_e) + ',' + str(
            thermal_kpi) + ',' + str(energy_usage_kpi) + ',' + str(out_door_air) + ',' + str(
            scenario) + ',' + str(target) + ',' + str(heat_consumption) + ',' + str(thermal_weight) + '\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()
        return R_
ahead_period=0
if test_scenario =='peak_heat_day':
    start_date=16-ahead_period
else:
    start_date=108-ahead_period
time_resolution=4
env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump', #'bestest_hydronic_heat_pump',
                                actions               = {
                                                            'oveTSet_u':(288.15,303.15),
                                                         },
                                observations          = {
                                                        'reaTZon_y': (250, 303),# zone air temp
                                                        'weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                                                        'reaPHeaPum_y': (0, energy_consumption_up_bond),# heating energy consumption
                                                        'LowerSetp[1]':(280, 310),
                                                        'UpperSetp[1]':(280, 310),
                                                         },

                                random_start_time=False,
                                predictive_period=0,
                                step_period = int(30/time_resolution) * 60,####7.5 *60????????
                                start_time=start_date * 24 * 3600,
                                max_episode_length    = _n_days_*24*3600,
                                warmup_period=3 * 24 * 3600,
                                scenario=test_scenario,
                                excluding_periods = [(start_date*24*3600,  start_date*24*3600+14*24*3600)]
                                )

from gymnasium.wrappers import NormalizeObservation

env = NormalizeObservation(env) # dont need it if only indoor air is considered
# env = ContinuousActionWrapper_xinlin_single_action (env)
env = ContinuousActionWrapper_xinlin_single_action_setpoint_15_30 (env)
# env = ContinuousActionWrapper_xinlin_single_action_setpoint_5_30 (env)



print('Action space of the wrapped agent:')
print(env.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)


from stable_baselines3 import DQN,A2C, PPO, SAC

learning_rates_p=learning_rate_()
learning_start_p=learning_starts_()
batch_size_p=batch_size_()

model = SAC("MlpPolicy", env, learning_rate=0.001, batch_size = 96,  ent_coef=0.1)

learning_steps = int(1*time_resolution*24*50) # (3 weeks--> 5 weeks....)
test_steps = int(1*time_resolution*24*(ahead_period+14))

print('Learning process is started')
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')

# Loop for one episode of experience (one day)
done = False
obs= env.reset()


for x in range(0, int(test_steps)):
  action, _ = model.predict(obs, deterministic=True) # c
  obs, reward, terminated, info = env.step(action)
  if x % 10 == 0:
      print("current step:", x)
      kpis = env.get_kpis()
      energy_usage_kpi = kpis['ener_tot']
      thermal_kpi = kpis['tdis_tot']
      print('energy kpi=', energy_usage_kpi)
      print('thermal discomfort kpi=', thermal_kpi)

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