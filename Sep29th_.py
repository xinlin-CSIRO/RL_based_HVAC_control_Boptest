import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date
# from Outdoor_temp_predictor import ONE_STEP_AHEAD_predictor
from sklearn.ensemble import GradientBoostingRegressor
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
energy_consumption_up_bond=4000
_n_days_=50
test_typo='typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\version_3_DQN_"+current_time+"_forecasting_.csv"
# result_testing = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\energy_objective_DQN_control_testing"+current_time+".csv"
f_1 = open(result_learning, "w+")
# f_2 = open(result_testing, "w+")  #str(indoor_air_temp) + ','+ str(action_)+','+str(reward)+','+ str(energy_r) + ',' + str(thermal_r) + ',' + str(current_consumption) + ',' +    str(energy_usage_kpi) +','+','+str(thermal_kpi)+'\n'
record='indoor temp, boundary,boundary, action, reward, outdoor air, thermal reward, energy reward, themal kpi ,energy kpi, scenario\n'


f_1.write(record)
f_1.close()
# f_2.write(record)
# f_2.close()
Last_energy_usage_kpi=0


def predictor_():
    return (0)

def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_(input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a

def last_energy_cosnumption ():
    return ()

def ONE_STEP_AHEAD_predictor (for_test, trainX, trainY):

    # trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    # trainY = np.reshape(trainY, (trainY.shape[0], trainY.shape[1], 1))

    trainX_2D = np.reshape(trainX, ( trainX.shape[1],trainX.shape[0]))
    trainY_2D  = trainY #np.reshape(trainY, (trainY.shape[0], trainY.shape[1]))
    # trainY_2D = (trainY_2D[:, 0]).reshape(-1, 1)
    model_ML = GradientBoostingRegressor(loss='squared_error',  n_estimators=5)   #, alpha=alpha)
    model_ML.fit(trainX_2D, trainY_2D)
    # make a one-step prediction
    for_test_1D = np.reshape(for_test, (1, 2))
    y_ = model_ML.predict(for_test_1D)
    #################################
    # model_ML.set_params(alpha=1.0 - alpha)
    # model_ML.fit(trainX_2D, trainY_2D)
    # y_lower = model_ML.predict(for_test_1D)
    # if(y_upper>y_lower):
    #     o_upper=y_upper
    #     o_lower=y_lower
    # else:
    #     o_upper = y_lower
    #     o_lower = y_upper
    return (y_)

couption=[]
outdoor_observation=[]
wind_speed_observation=[]
prediction_couption=[]
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
        out_door_air = observations[2] - 273.15
        wind_speed_observation.append(observations[3])
        outdoor_observation.append(out_door_air)
        for_test=np.array([out_door_air,observations[3]])
        predicted_outdoor=0
        if(len(outdoor_observation)==101):
            outdoor_air_training=outdoor_observation[0:100]
            wind_speed_training=wind_speed_observation[0:100]
            train_X=np.array([outdoor_air_training,wind_speed_training])
            train_Y=np.array(outdoor_observation[1:101])
            outdoor_observation.pop(0)
            wind_speed_observation.pop(0)
            predicted_outdoor=ONE_STEP_AHEAD_predictor(for_test, train_X, train_Y)[0]
            print('poping')
        last_prediction_error=0
        if(len(prediction_couption)>=1):
            last_prediction_error=prediction_couption[-1]-out_door_air
        final_prediction=predicted_outdoor -last_prediction_error
        current_consumption = observations[1]
        couption.append(current_consumption)
        couption_ = np.array(couption)
        max_ = max(couption_)
        min_ = min(couption_)
        if (len(couption) > 1):
            last_consumption = couption_[-2]
            last_consumption_normalized = (max_ - last_consumption) / (max_ - min_)
            current_consumption_normalized = (max_ - current_consumption) / (max_ - min_)
        else:
            last_consumption = 0
            last_consumption_normalized = 0
            current_consumption_normalized = (max_ - current_consumption) / (max_ - min_)

        energy_changes = -(current_consumption_normalized - last_consumption_normalized)
        energy_r = energy_changes

        up_bundary=26
        low_boundary=20
        indoor_air_temp=observations[0] - 273.15
        set_indoor_air=round(action[0]- 273.15)
        target=(up_bundary+low_boundary)/2
        diff_real = abs(indoor_air_temp - target)


        if (low_boundary <= indoor_air_temp <= up_bundary):
            indoor_good = 1
        else:
            indoor_good = 0

        if (indoor_good == 0):
            thermal_r=penality_(diff_real)
            scenario = 2


        elif (indoor_good == 1):
            thermal_r=reward_(diff_real)
            scenario = 3

        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi=kpis['tdis_tot']
        # print('kpi=',energy_usage_kpi)


        weight=1
        if (math.isnan(energy_r)) or (thermal_r<0):
            reward = thermal_r
        else:
            reward = weight * thermal_r + (1 - weight) * energy_r


        # reward =  thermal_r

        action_=action[0] -273.15
        result_sting = str(indoor_air_temp) +','+str(low_boundary)+ ','+str(up_bundary) + ','+ str(action_)+','+str(reward)+','+ str(out_door_air) + ',' + str(thermal_r) +\
                       ',' +  str(energy_r) +','+str(thermal_kpi)+','+str(energy_usage_kpi)+','+str(scenario)+','+str(predicted_outdoor)+','+str(last_prediction_error)+','+str(final_prediction)+'\n'
        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()

        prediction_couption.append(predicted_outdoor)

        return reward

env = BoptestGymEnvCustomReward(url                   = url,
                                testcase              = 'bestest_hydronic_heat_pump',
                                actions               = {
                                                            'oveTSet_u':(293.15, 299.15), # 20, 26--> 293.15,303.15
                                                         },
                                observations          = {
                                                        'reaTZon_y':(250,303),
                                                        'reaPHeaPum_y':(0,energy_consumption_up_bond),
                                                        'weaSta_reaWeaTDryBul_y':(250,303),#outside air temp
                                                        'weaSta_reaWeaWinSpe_y':(0,30),
                                                         },
                                # random_start_time     = True, #False,
                                start_time = 15*24*3600,
                                # step_period           = 3600,
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
learning_steps=500
test_steps=300

print('Learning process is started')
model.learn(total_timesteps=learning_steps)
print('Learning process is completed')

# Loop for one episode of experience (one day)
done = False
obs= env.reset()


for x in range(0, test_steps):
  # action, _ = model.predict(obs, deterministic=True) # c
  action, _ = model.predict(obs)  # c
  # a=env.step(action)
  # obs,reward,terminated,truncated,info = env.step(action)
  initial_action_space=env.val_bins_act[0]
  action_back=initial_action_space[action]

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
