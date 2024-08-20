import os
import shutil
from git import Repo

# Remove the directory if it exists
if os.path.exists('project1-boptest-gym'):
    shutil.rmtree('project1-boptest-gym')

# Clone the repository
Repo.clone_from('https://github.com/ibpsa/project1-boptest-gym.git', 'project1-boptest-gym', branch='boptest-gym-service')
print ('100%')




import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date

def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_ (input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a
counter=0
def the_one_under_test_time (low_boundary, up_bundary,indoor_air_temp, current_consumption, energy_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        # target = (low_boundary + 3)  # 15+3=18
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2
    global counter

    if(wild_boundary==1):
        # counter+=1
        # target = (low_boundary + 4)+ counter*0.2
        lowest=(up_bundary + low_boundary) / 2 - 3
        back_bound=21
        step_length=(back_bound-lowest)/25
        target =  lowest + counter * step_length
        counter += 1
        loww_b= target #-1
    else:
        # global counter
        counter=0
        loww_b=low_boundary

    diff_curr = abs(indoor_air_temp - target)

    if (loww_b < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    weight_reward_for_thermal = 0.8
    weight_penality_for_thermal = 0.8
    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized

            R_ = r_t
            s_=0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.8
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp): # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized-1
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.6
            R_ = weight_penality_for_thermal*r_t + (1 - weight_penality_for_thermal) *r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp): # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.6
            R_ = weight_penality_for_thermal*r_t + (1 - weight_penality_for_thermal) *r_e
            # R_ = r_t +  r_e
            s_ = 3
        # else:
        #     r_t = penality_(diff_curr)
        #     r_e=0
        #     R_ = r_t
        #     s_ = 4

    return R_, r_t,r_e, current_indoor_state,wild_boundary,target,weight_reward_for_thermal, s_,loww_b


counter_2=0
def the_one_under_test_time_2 (low_boundary, up_bundary,indoor_air_temp, current_consumption, energy_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        # target = (low_boundary + 3)  # 15+3=18
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2
    global counter_2

    if(wild_boundary==1):
        # counter+=1
        # target = (low_boundary + 4)+ counter*0.2
        lowest=(up_bundary + low_boundary) / 2 - 3
        back_bound=21
        step_length=(back_bound-lowest)/25
        target =  lowest + counter_2 * step_length
        counter_2 += 1
        loww_b= target #-1
    else:
        # global counter
        counter_2=0
        loww_b=low_boundary

    diff_curr = abs(indoor_air_temp - target)

    if (loww_b < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    weight_reward_for_thermal = 0.9
    weight_penality_for_thermal = 0.8
    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized

            R_ = r_t
            s_=0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.8
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp): # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized-1
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.6
            R_ = weight_penality_for_thermal*r_t + (1 - weight_penality_for_thermal) *r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp): # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.6
            R_ = weight_penality_for_thermal*r_t + (1 - weight_penality_for_thermal) *r_e
            # R_ = r_t +  r_e
            s_ = 3
        # else:
        #     r_t = penality_(diff_curr)
        #     r_e=0
        #     R_ = r_t
        #     s_ = 4

    return R_, r_t,r_e, current_indoor_state,wild_boundary,target,weight_reward_for_thermal, s_,loww_b



def the_one_under_test_wo_e (low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption):

        if (len(energy_couption) >= 1):
            energy_couption_ = np.array(energy_couption)
            max_ = max(energy_couption_)
            min_ = min(energy_couption_)
        else:
            max_ = 0
            min_ = 0
        if (max_ != min_):
            ###if you wanna normalize a series into [a, b]
            ### (b-a)*(x-min)/(max-min)+a
            current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
        else:
            current_consumption_normalized = 0


        if (up_bundary > 24) and (low_boundary < 21):
            wild_boundary = 1
            target = (low_boundary + 0)  # 15+3=18
        else:
            wild_boundary = 0
            target = (up_bundary + low_boundary) / 2

        diff_curr = abs(indoor_air_temp - target)

        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

        if current_indoor_state == 1:
            if (wild_boundary == 0):  # [21-24]
                r_t = 1.0 * reward_(diff_curr)  # 1.1
                r_e = 1 - current_consumption_normalized
                R_ = r_t
                s_ = 0

            else:  # [15-30]
                r_t = reward_(diff_curr)
                r_e = 1 - current_consumption_normalized
                weight = 0.3
                R_ = weight * r_t + (1 - weight) * r_e
                s_ = 1

        else:
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                # weight = 0.8
                # R_ = weight*r_t + (1 - weight) *r_e
                R_ = r_t + r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                # weight = 0.8
                # R_ = weight*r_t + (1 - weight) *r_e
                R_ = r_t + r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                s_ = 4

        return R_, r_t, r_e, current_indoor_state, wild_boundary, target, s_

def the_one_under_test_1 (low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0


    # if (len(previous_couption)) > 1:
    #     last_step_state = previous_couption[-1]
    # else:
    #     last_step_state = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (low_boundary + 2)  # 15+3=18
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2

    diff_curr = abs(indoor_air_temp - target)

    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized
            R_ = r_t
            s_=0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            weight = 0.4
            R_ = weight * r_t + (1 - weight) * r_e
            s_ = 1

    else:
        if (low_boundary > indoor_air_temp): # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized-1
            # weight = 0.8
            # R_ = weight*r_t + (1 - weight) *r_e
            R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp): # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.8
            # R_ = weight*r_t + (1 - weight) *r_e
            R_ = r_t +  r_e
            s_ = 3
        else:
            r_t = penality_(diff_curr)
            R_ = r_t
            s_ = 4

    return R_, r_t,r_e, current_indoor_state,wild_boundary,target,s_

def the_one_under_test_1025_best (low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0


    # if (len(previous_couption)) > 1:
    #     last_step_state = previous_couption[-1]
    # else:
    #     last_step_state = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (low_boundary + 2)  # 15+3=18
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2

    diff_curr = abs(indoor_air_temp - target)

    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized
            R_ = r_t
            s_=0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            weight = 0.4
            R_ = weight * r_t + (1 - weight) * r_e
            s_ = 1

    else:
        if (low_boundary > indoor_air_temp): # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized-1
            # weight = 0.8
            # R_ = weight*r_t + (1 - weight) *r_e
            R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp): # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.8
            # R_ = weight*r_t + (1 - weight) *r_e
            R_ = r_t +  r_e
            s_ = 3
        else:
            r_t = penality_(diff_curr)
            R_ = r_t
            s_ = 4

    return R_, r_t,r_e, current_indoor_state,wild_boundary,target,s_


def the_one_under_test_1024 (low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption):  #best performane reward function by 1024
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0
    r_e = 1 - current_consumption_normalized

    if (len(previous_couption)) > 1:
        last_step_state = previous_couption[-1]
    else:
        last_step_state = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (low_boundary + 3)  # 15+3=18
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2

    diff_curr = abs(indoor_air_temp - target)

    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            # weight = 0.95
            # R_ = weight * r_t + (1 - weight) * r_e
            R_ = r_t

        else:  # [15-30]
            r_t = reward_(diff_curr)
            weight = 0.6
            R_ = weight * r_t + (1 - weight) * r_e

    else:
        if (last_step_state == 0) and (wild_boundary == 0) and (action_ == 1) and (
                low_boundary > indoor_air_temp):
            r_t = reward_(diff_curr)
            R_ = r_t
        elif (last_step_state == 0) and (wild_boundary == 0) and (action_ == 0) and (
                up_bundary < indoor_air_temp):
            r_t = reward_(diff_curr)
            R_ = r_t
        else:
            r_t = penality_(diff_curr)
            R_ = r_t

    return R_, r_t,r_e, current_indoor_state,wild_boundary,target

def the_one_under_test_1023 (low_boundary, up_bundary,indoor_air_temp, current_consumption, action_, energy_couption,previous_couption):
    if(len(energy_couption)>=1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        ###if you wanna normalize a series into [a, b]
        ### (b-a)*(x-min)/(max-min)+a
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0
    r_e = 1 - current_consumption_normalized

    if (len(previous_couption)) > 1:
        last_step_state = previous_couption[-1]
    else:
        last_step_state = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (low_boundary + 3)
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2

    diff_curr = abs(indoor_air_temp - target)

    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1 * reward_(diff_curr) # 1.1
            # weight = 0.95
            # R_ = weight * r_t + (1 - weight) * r_e
            R_ = r_t

        else:  # [15-30]
            r_t = reward_(diff_curr)
            weight = 0.3
            R_ = weight * r_t + (1 - weight) * r_e

    else:
        # if (last_step_state == 0) and (wild_boundary == 0) and (action_ == 308.15) and (
        #         low_boundary > indoor_air_temp):
        #     r_t = reward_(diff_curr)
        #     R_ = r_t
        # elif (last_step_state == 0) and (wild_boundary == 0) and (action_ == 288.15) and (
        #         up_bundary < indoor_air_temp):
        #     r_t = reward_(diff_curr)
        #     R_ = r_t
        # else:
            r_t = penality_(diff_curr)
            R_ = r_t


    return R_, r_t,r_e, current_indoor_state,wild_boundary
