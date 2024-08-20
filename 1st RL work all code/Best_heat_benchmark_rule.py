import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date
from boptestGymEnv import NormalizedObservationWrapper

os.chdir(r"/")
from boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper
from learning_paras import learning_steps_, test_steps_, learning_rate_, learning_starts_, batch_size_, start_time_
from wrapper_xinlin import DiscretizedActionWrapper_xinlin_, ContinuousActionWrapper_xinlin,ContinuousActionWrapper_xinlin_single_action

url = 'https://api.boptest.net'
seed = 123456
random.seed(seed)
np.random.seed(seed)

excluding_periods = [(79 * 24 * 3600, 355 * 24 * 3600)]

energy_consumption_up_bond = 4000
_n_days_ = 300
test_typo = 'peak_heat_day'  # 'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time = now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\best_heat_rule_" + current_time + "_.csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor_temp,boundary_0,boundary_1,action_0,themal_kpi,energy_kpi,cost_kpi,outdoor_air,heating consumption,fan consumption,price,all_consumption\n'
f_1.write(record)
f_1.close()
Last_energy_usage_kpi = 0
counter = 0

energy_couption = []
previous_couption = []

start_time_test     = 31*24*3600
episode_length_test = 3*24*3600
warmup_period_test  = 3*24*3600

###########import the_one_under_test---> low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption
class BoptestGymEnvCustomReward(BoptestGymEnv):
    def compute_reward(self, action):
        R_ = 0
        return R_



env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump', #'bestest_hydronic_heat_pump',
                                actions               = {
                                                            'oveTSet_u':(288.15, 303.15),
                                                         },
                                observations          = {
                                                        'reaTZon_y': (250, 303),# zone air temp
                                                        'weaSta_reaWeaTDryBul_y': (250, 303),  # outside air temp
                                                        'reaPFan_y': (0, energy_consumption_up_bond),# fan energy consumption
                                                        'reaPHeaPum_y': (0, energy_consumption_up_bond),# heating energy consumption
                                                        'LowerSetp[1]':(280, 310),
                                                        'UpperSetp[1]':(280, 310),
                                                        'PriceElectricPowerDynamic': (0., 1.)
                                                         },

                                # random_start_time     = True, #False,
                                # regressive_period    = 2*15 * 60,
                                predictive_period  =0,
                                step_period= 2* 30 * 60,
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                )

from gymnasium.wrappers import NormalizeObservation

# env = NormalizeObservation(env)  # dont need it if only indoor air is considered
# env = DiscretizedActionWrapper(env, n_bins_act=10)

# env = DiscretizedActionWrapper_xinlin_(env, N) #Xinlin_descretized_action_wrapper is used for multi-demensional actions game--established 2024 Jan 10
env = ContinuousActionWrapper_xinlin_single_action (env)
# env = DiscretizedObservationWrapper(env, n_bins_obs=5, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
# print('Action space of the original agent:')
# print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)

learning_steps = 2400  # learning_steps_()
test_steps = 672  # test_steps_()

# Loop for one episode of experience (one day)



class baseline_controller(object):

    def predict(self, observations):
        low_boundary = observations[4] - 273.15
        up_bundary = observations[5] - 273.15
        print(f"Low: {low_boundary}, Up: {up_bundary}")  # Diagnostic print
        if low_boundary < 20 and up_bundary > 26:
            print("Taking IF path")  # Diagnostic print
            action = np.array([low_boundary], dtype=np.int32)
        else:
            print("Taking ELSE path")  # Diagnostic print
            ref = (low_boundary + up_bundary) / 2
            action = np.array([ref], dtype=np.int32)
        return action

for x in range(0, learning_steps):

    # record = 'indoor_temp,boundary_0,boundary_1,action_0, themal_kpi,energy_kpi,cost_kpi,outdoor_air,cool_consumption,heating consumption,fan consumption,price,all_consumption\n'
    result_sting = str(0) + ',' + str(0) + ',' + str(0) + ',' + str(
        0) + ',' + str(0) + ',' + str(0)  + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(0) + ',' + str(
            0) + ',' + str(0)+',' + str(0) + '\n'
    result_sting.replace('[', '').replace(']', '')
    f_1 = open(result_learning, "a+")
    f_1.write(result_sting)
    f_1.close()

observations = env.reset()

for x in range(0, test_steps):


    low_boundary = observations[4] - 273.15
    up_bundary = observations[5] - 273.15
    indoor_air_temp = observations[0] - 273.15
    # print(f"Low: {low_boundary}, Up: {up_bundary}")  # Diagnostic print
    if low_boundary < 20 and up_bundary > 26:
        # print("Taking IF path")  # Diagnostic print
        action = np.array(low_boundary, dtype=np.int32) + 273.15
    else:
        if(low_boundary< indoor_air_temp<up_bundary):
            ref = (low_boundary + up_bundary) / 2
        else:
            ref=30
        ref = ref + 273.15
        action = np.array(ref, dtype=np.int32)


    # action = model_.predict(obs)
    observations, reward, terminated, info = env.step(action)
    action_=action- 273.15
    indoor_air_temp = observations[0] - 273.15
    out_door_air = observations[1] - 273.15
    fan_consumption = observations[2]
    heat_consumption = observations[3]

    kpis = env.get_kpis()
    energy_usage_kpi = kpis['ener_tot']
    thermal_kpi = kpis['tdis_tot']
    cost_kpi = kpis['cost_tot']

    low_boundary = observations[4] - 273.15
    up_bundary = observations[5] - 273.15
    price = observations[6]
    all_consumption =  heat_consumption  # + fan_consumption
    # record = 'indoor_temp,boundary_0,boundary_1,action_0, themal_kpi,energy_kpi,cost_kpi,outdoor_air,cool_consumption,heating consumption,fan consumption,price,all_consumption\n'
    result_sting = str(indoor_air_temp) + ',' + str(low_boundary) + ',' + str(up_bundary) + ',' + str(
        action_) + ',' + str(thermal_kpi) + ',' + str(energy_usage_kpi)  + ',' + str(cost_kpi) + ',' + str(out_door_air)  + ',' + str(heat_consumption) + ',' + str(
            fan_consumption) + ',' + str(price)+',' + str(all_consumption) + '\n'
    result_sting.replace('[', '').replace(']', '')
    f_1 = open(result_learning, "a+")
    f_1.write(result_sting)
    f_1.close()

    if x % 10 == 0:
        print(x)

print(kpis)

f_1 = open(result_learning, "a+")
f_1.write(str(energy_usage_kpi))
f_1.close()

now = datetime.now()
time_ = now.strftime("%H-%M")
print(time_)