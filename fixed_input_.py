import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date
from boptestGymEnv import NormalizedObservationWrapper
os.chdir(r"C:\Users\wan397\PycharmProjects\Boptest\project1-boptest-gym")
from boptestGymEnv import BoptestGymEnv_xinlin, DiscretizedActionWrapper
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
_n_days_=50
test_typo='peak_heat_day' #'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\26_based_"+current_time+"_.csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor air, outdoor air, action, consumption, energy kpi\n'#result_sting = str(indoor) + ',' + str(outdoor) + ',' + str(action_)+','+str(energy_consumption) + ',' + str(energy_usage_kpi) +'\n'
f_1.write(record)
f_1.close()
# f_2.write(record)
# f_2.close()
Last_energy_usage_kpi=0


class baseline_controller():
    def __init__(self, TSet=23.15 + 273.15):
        self.TSet = TSet

    def predict(self, obs):
        # if (obs[0] <= self.TSet):
        #     action = np.array(1, dtype=np.int32)
        # else:
        #     action = np.array(0, dtype=np.int32)
        action = np.array(self.TSet , dtype=np.int32)
        return action


env = BoptestGymEnv_xinlin(
        url                  = url,
        testcase             = 'bestest_hydronic_heat_pump',
    actions={
        'oveHeaPumY_u':(0,1),# Heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity)
            },
    observations={
        'reaTZon_y': (250, 303),
        'reaPHeaPum_y': (0, energy_consumption_up_bond),
        'weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                },
    # predictive_period    = 24*3600,
    start_time=25 * 24 * 3600,
    step_period           = 3600/2,
    warmup_period=10 * 24 * 3600,
    scenario=test_typo,
    max_episode_length=_n_days_ * 24 * 3600,

)


from gymnasium.wrappers import NormalizeObservation

# env = NormalizeObservation(env) # dont need it if only indoor air is considered
# env = DiscretizedActionWrapper(env, n_bins_act=1)

print('Action space of the wrapped agent:')
print(env.action_space)
print('Action space of the original agent:')
print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)

model = baseline_controller()

obs= env.reset()


for x in range(0, 1000):
  # action, _ = model.predict(obs, deterministic=True) # c
      action = model.predict(obs)  # c


      obs, reward, done, info = env.step(action)

      x = env.get_kpis()
      energy_usage_kpi = x['ener_tot']
      # print('kpi=', energy_usage_kpi)


      indoor = obs[0] - 273.15
      outdoor=obs[2] - 273.15
      energy_consumption=obs[1]
      action_=action- 273.15
      result_sting = str(indoor) + ',' + str(outdoor) + ',' + str(action_)+','+str(energy_consumption) + ',' + str(energy_usage_kpi) +'\n'

      result_sting.replace('[', '').replace(']', '')

      f_1 = open(result_learning, "a+")
      f_1.write(result_sting)
      f_1.close()


print(x)


f_1 = open(result_learning, "a+")
f_1.write(str(energy_usage_kpi))
f_1.close()

