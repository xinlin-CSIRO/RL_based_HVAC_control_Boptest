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
test_typo='peak_heat_day'#'typical_heat_day' #'peak_heat_day'
now = datetime.now()
current_time= now.strftime("%dth-%b-%H-%M")
# current_time2= now.strftime("%H:%M")
result_learning = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion_forecasting_energy_"+current_time+"_forecasting_.csv"
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
previous_states=[]
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

        # forecasting part
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

        # energy part
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

        if (current_consumption_normalized == 1 and last_consumption_normalized == 1):
            energy_changes = 1
        else:
            energy_changes = (current_consumption_normalized - last_consumption_normalized)

        energy_r = energy_changes

        up_bundary=26
        low_boundary=20
        indoor_air_temp=observations[0] - 273.15
        set_indoor_air=round(action[0]- 273.15)
        target=(up_bundary+low_boundary)/2
        diff_real = abs(indoor_air_temp - target)

        edge=0.4
        if ((low_boundary+edge) < indoor_air_temp < (up_bundary-edge)):
            indoor_states = 0
        elif ((up_bundary -edge) <= indoor_air_temp <= up_bundary):
            # last_outdoor=outdoor_observation[-2]
            # if(last_outdoor<out_door_air<final_prediction): # outdoor air is rising
            if (out_door_air <= final_prediction):  # outdoor air is rising
                indoor_states = 1
            else:
                indoor_states = 0
        elif (low_boundary <= indoor_air_temp <= (low_boundary+edge)):
            # last_outdoor = outdoor_observation[-2]
            # if (last_outdoor > out_door_air > final_prediction):  # outdoor air is rising
            if ( out_door_air >= final_prediction):  # outdoor air is declining
                indoor_states = 2
            else:
                indoor_states = 0
        else:
            indoor_states = 3

        if (indoor_states == 0):# good one
            thermal_r = reward_(diff_real)

        elif (indoor_states == 1):
            thermal_r = penality_(diff_real)

        elif (indoor_states == 2):
            thermal_r=penality_(diff_real)

        else:
            thermal_r=2*penality_(diff_real)



        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        energy_usage_kpi = kpis['ener_tot']
        thermal_kpi=kpis['tdis_tot']
        # print('kpi=',energy_usage_kpi)

        if (len(previous_states) >= 1):
            last_state = previous_states[-1]
        else:
            last_state = 1
        weight = 0.3
        if (math.isnan(energy_r)) or (thermal_r< 0):
            reward = thermal_r
        else:
            if ( last_state !=0 ):
                reward = thermal_r
            else:
                reward = weight * thermal_r + (1 - weight) * energy_r

        action_=action[0] -273.15
        result_sting = str(indoor_air_temp) +','+str(low_boundary)+ ','+str(up_bundary) + ','+ str(
            action_)+','+str(reward)+','+ str(out_door_air) + ',' + str(thermal_r) +\
                       ',' +  str(energy_r) +','+str(thermal_kpi)+','+str(energy_usage_kpi)+','+str(indoor_states)+','+\
                       str(predicted_outdoor)+','+str(last_prediction_error)+','+str(final_prediction)+',' + str(
            current_consumption) + ',' + str(current_consumption_normalized) + ',' + str(
            last_consumption_normalized) +'\n'

        result_sting.replace('[', '').replace(']', '')


        result_sting.replace('[', '').replace(']', '')
        f_1 = open(result_learning, "a+")
        f_1.write(result_sting)
        f_1.close()

        prediction_couption.append(predicted_outdoor)
        previous_states.append(indoor_states)

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
                                start_time = 25*24*3600,
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

from stable_baselines3 import DQN, A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# from stable_baselines.ddpg.policies import MlpPolicy

model = DQN('MlpPolicy', env, verbose=1,
            learning_rate=0.5,
            batch_size= 1*24, #365*24
            learning_starts=24,
            train_freq=1)


# model = PPO('MlpPolicy', env, verbose=1,
#             learning_rate=0.001,
#             )
# model = A2C("MlpPolicy", env, verbose=1)

learning_steps=1000
test_steps=1000

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
