import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date
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
energy_consumption_up_bond=5000
out_lower_setp = -30 + 273.15
out_upper_setp =  40 + 273.15
lower_setp = 5 + 273.15
upper_setp = 10 + 273.15
_n_days_=50
test_typo='typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_"+current_time+".csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor air, action, reward, energy reward, thermal regard, current consumption,energy kpi, themal kpi\n'
f_1.write(record)
f_1.close()
# f_2.write(record)
# f_2.close()
Last_energy_usage_kpi=0


def predictor_():
    return (0)
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
            obs = env.reset()
            obs = obs[0]
            # Print episode number and starting day from beginning of the year:
            print('-------------------------------------------------------------------')
            print('Episode number: {0}, starting day: {1:.1f} ' \
                  '(from beginning of the year)'.format(i + 1, env.unwrapped.start_time / 24 / 3600))
            while not done:
                # Get action with epsilon-greedy policy and simulate
                act = self.predict(obs, deterministic = False)
                action = np.array([200, 400])  # 200-273
                nxt_obs, rew, done, _ = env.step(action)
                obs = nxt_obs

                # print ('nxt_obs= ', nxt_obs)
                print('the current observation is:', obs)
                print('action is: ', action[0])
                # print('Set temp is: ', act + 293.15)
                # print('reward is: ', rew)
                supply_sp = 0  # np.array(env.val_bins_act)[0,action[0]]
                return_sp = nxt_obs[0] - 273.15
                heat_power = nxt_obs[1]
                cool_power = nxt_obs[2]
                out_weather = nxt_obs[3] - 273.15
                DNI = nxt_obs[4]
                RH = nxt_obs[5]
                TIME = nxt_obs[6]
                predicted_outdoor= predictor_()
                record_s = str(supply_sp) + ',' + str(return_sp) + ',' + str(heat_power) + ',' + str(
                    cool_power) + ',' + str(out_weather) + ',' + str(DNI) + ',' + str(RH) + ',' + str(
                    TIME) + '\n'  # 'set,return,heat, cool, outside/n'
                f_1 = open(result_location, "a+")
                f_1.write(record_s)
                # f_1.write(result_step)
                f_1.close()

def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_(input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a

def last_energy_cosnumption ():
    return ()

couption=[]
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
        # Electrical_power_of_the_heat_pump_evaporator_fan=observations[1]
        # Heat_pump_electrical_power = observations[2]
        # Emission_circuit_pump_electrical_power = observations[3]
        # Floor_heating_thermal_power_released_to_the_zone = observations[4]

        # Heat_pump_thermal_power_exchanged_in_the_evaporator = observations[6]
        # out_door_air = observations[1] - 273.15
        #

        # current_consumption = observations[1]
        # couption.append(current_consumption)
        # couption_=np.array(couption)
        # max_=max(couption_)
        # min_=min(couption_)
        # if(len(couption)>1):
        #     last_consumption=couption_[-2]
        #     last_consumption_normalized = (max_ - last_consumption) / (max_ - min_)
        #     current_consumption_normalized = (max_ - current_consumption) / (max_ - min_)
        # else:
        #     last_consumption=0
        #     last_consumption_normalized = 0
        #     current_consumption_normalized = (max_ - current_consumption) / (max_ - min_)
        #
        # energy_changes=-(current_consumption_normalized-last_consumption_normalized)
        # energy_r=energy_changes

        up_bundary=24
        low_boundary=22
        indoor_air_temp=observations[0] - 273.15
        set_indoor_air=action[0]- 273.15
        target=(up_bundary+low_boundary)/2
        diff_real = abs(indoor_air_temp - target)
        diff_set = abs(set_indoor_air - target)
        if (low_boundary < indoor_air_temp < up_bundary):
            indoor_good = 1
        else:
            indoor_good = 0
        if (low_boundary < set_indoor_air < up_bundary):
            indoor_set_good = 1
        else:
            indoor_set_good = 0

        if (indoor_good ==indoor_set_good) and (indoor_set_good==1):
            thermal_r=reward_(diff_real)
        elif (indoor_good ==indoor_set_good) and (indoor_set_good==0):
            thermal_r=penality_(diff_set)
        elif (indoor_good == 0) and (indoor_set_good == 1):
            thermal_r=penality_(diff_real)
        elif (indoor_good == 1) and (indoor_set_good == 0):
            thermal_r=penality_(diff_set)



        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi=kpis['tdis_tot']
        print('kpi=',energy_usage_kpi)


        # weight=0.99
        # if (math.isnan(energy_r)):
        #     reward = 1 * thermal_r
        # else:
        #     reward = weight * thermal_r + (1 - weight) * energy_r
        reward =  thermal_r

        action_=action[0] -273.15
        result_sting = str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str("energy_r") + ',' + str(thermal_r) + ',' + str("current_consumption") + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()
        return reward

env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump',
                                actions               = {
                                                            # 'oveHeaPumY_u':(0,1),# Heat pump modulating signal for compressor speed between 0 (not working) and 1 (working at maximum capacity)
                                                            # 'oveFan_u':(0,1), # Integer signal to control the heat pump evaporator fan either on or off
                                                            # 'ovePum_u':(0,1), #Integer signal to control the emission circuit pump either on or off
                                                            'oveTSet_u':(288.15, 308.15), # 15, 35
                                                             # 'oveTSet_activate': 1,
                                                         },
                                observations          = {
                                                         # 'time':(0,604800),#per step one hour
                                                         'reaTZon_y':(250,303),
                                                         # 'reaTSup_y':(250,303),
                                                        # 'reaPFan_y':(0,energy_consumption_up_bond),

                                                        # 'reaPHeaPum_y':(0,energy_consumption_up_bond),

                                                        # 'reaPPumEmi_y':(0,energy_consumption_up_bond),
                                                        # 'reaQFloHea_y' :(0,energy_consumption_up_bond),
                                                        # 'reaQHeaPumCon_y':(0,energy_consumption_up_bond),
                                                        # 'reaQHeaPumEva_y':(0,energy_consumption_up_bond),
                                                        #
                                                          # 'weaSta_reaWeaTDryBul_y':(250,303),
                                                        #   'weaSta_reaWeaHDirNor_y':(0, 1000), #radiation
                                                        #   'weaSta_reaWeaRelHum_y':(0, 100), #Relative humidity
                                                        #   'weaSta_reaWeaWinSpe_y':(0, 3600) #wend speed
                                                         },
                                random_start_time     = True, #False,
                                start_time = 15*24*3600,
                                # step_period           = 3600,
                                warmup_period  = 10*24*3600,
                                scenario=test_typo,
                                max_episode_length    = _n_days_*24*3600,
                                )

from gymnasium.wrappers import NormalizeObservation

# env = NormalizeObservation(env) # dont need it if only indoor air is considered
env = DiscretizedActionWrapper(env, n_bins_act=5)
# env = DiscretizedObservationWrapper(env, n_bins_obs=5, outs_are_bins=True)
print('Action space of the wrapped agent:')
print(env.action_space)
print('Action space of the original agent:')
print(env.unwrapped.action_space)

print('observation space of the wrapped agent:')
print(env.observation_space)

# model = my_agent (env)

from stable_baselines3 import DQN
from stable_baselines3 import A2C, PPO
#
model = DQN('MlpPolicy', env, verbose=1,
            learning_rate=0.5,
            batch_size= 1*24, #365*24
            learning_starts=24,
            train_freq=1)

# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=0.001,
#             )
learning_steps=1000
print('Learning process is started')
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')

# Loop for one episode of experience (one day)
done = False
obs= env.reset()

test_steps=2000
for x in range(0, test_steps):
  action, _ = model.predict(obs, deterministic=True) # c
  # a=env.step(action)
  # obs,reward,terminated,truncated,info = env.step(action)
  initial_action_space=env.val_bins_act[0]
  action_back=initial_action_space[action]

  obs, reward, terminated, info = env.step(action)


  x = env.get_kpis()
  energy_usage_kpi = x['ener_tot']
  print('kpi=', energy_usage_kpi)


print(x)