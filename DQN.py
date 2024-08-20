import os
import requests
import numpy as np
import random
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
lower_setp = 21 + 273.15
upper_setp = 24 + 273.15

from boptestGymEnv import NormalizedObservationWrapper
from stable_baselines3 import DQN
class BoptestGymEnvCustomReward(BoptestGymEnv):
    def compute_reward(self):
        # Compute BOPTEST core kpis
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        # Calculate objective integrand function as the total discomfort
        objective_integrand = kpis['tdis_tot']
        # Give reward if there is not immediate increment in discomfort
        if objective_integrand == self.objective_integrand:
            reward = 1
        else:
            reward = 0
        # Record current objective integrand for next evaluation
        self.objective_integrand = objective_integrand
        return reward

env = BoptestGymEnvCustomReward(
    url=url,
    actions=['oveHeaPumY_u'],
    observations={'time': (0, 604800),
                  'reaTZon_y': (280., 310.),
                  'TDryBul': (265, 303),
                  'HDirNor': (0, 862),
                  'InternalGainsRad[1]': (0, 219),
                  'PriceElectricPowerHighlyDynamic': (-0.4, 0.4),
                  'LowerSetp[1]': (280., 310.),
                  'UpperSetp[1]': (280., 310.)},
    predictive_period=24 * 3600,
    regressive_period=6 * 3600,
    random_start_time=True,
    max_episode_length=7 * 24 * 3600,
    warmup_period=24 * 3600,
    step_period=900)

env = NormalizedObservationWrapper(env)
env = DiscretizedActionWrapper(env, n_bins_act=10)

print('Observation space of the building environment (dimension):')
print(env.observation_space.shape)
print('Action space of the building environment:')
print(env.action_space)

model = DQN('MlpPolicy', env, verbose=1, gamma=0.99, #seed=seed,
            learning_rate=5e-4, batch_size=24,
            buffer_size=365*24, learning_starts=24, train_freq=1)

# Main training loop
model.learn(total_timesteps=10)

