import os
import requests
import numpy as np
import random
import math
from datetime import datetime, date

energy_max=4500
energy_min=0
def penality_(input):
    a = -((2) / (1 + math.exp(-input))) + 1
    return a


def reward_(input):
    a = -((2) / (1 + math.exp(-input))) + 2
    return a

def penality_new (input):
    a = -((2) / (1 + math.exp(-input/1))) + 1
    return a


def reward_new (input):
    a = -((2) / (1 + math.exp(-input/1))) + 2
    return a

counter = 0


def reward_after_paper_1(observed_low_boundary, observed_up_bundary, indoor_air_temp, outside_air):
    narrow_high_bond = 24
    narrow_low_bond = 21

    global counter
    pre_heating_steps = 4  # 15 mins *4 =1 hour
    if (observed_up_bundary > narrow_high_bond) and (observed_low_boundary < narrow_low_bond) and (
            0 <= counter < (25 - pre_heating_steps)):

        counter += 1
        if (outside_air <= narrow_low_bond):  # cold weather
            saving_interval = - 3
            target = (observed_up_bundary + observed_low_boundary) / 2 + saving_interval
            low_for_reward = observed_low_boundary
            high_for_reward = narrow_high_bond
        else:
            saving_interval = 3
            target = (observed_up_bundary + observed_low_boundary) / 2 + saving_interval
            low_for_reward = narrow_low_bond
            high_for_reward = observed_up_bundary
    elif (observed_up_bundary > narrow_high_bond) and (observed_low_boundary < narrow_low_bond) and (
            (25 - pre_heating_steps) <= counter <= 25):
        counter += 1
        target = (observed_up_bundary + observed_low_boundary) / 2
        low_for_reward = narrow_low_bond
        high_for_reward = narrow_high_bond
    else:
        target = (observed_up_bundary + observed_low_boundary) / 2
        counter = 0
        low_for_reward = observed_low_boundary
        high_for_reward = observed_up_bundary

    diff_curr = abs(indoor_air_temp - target)

    if (low_for_reward < indoor_air_temp < high_for_reward):
        current_indoor_state = 1
        r_t = reward_(diff_curr)
    else:
        current_indoor_state = 0
        r_t = penality_(diff_curr)
    R_ = r_t

    return R_, target, current_indoor_state, low_for_reward, high_for_reward


def the_one_under_test_time(low_boundary, up_bundary, indoor_air_temp, current_consumption, energy_couption,
                            outside_air):
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

    narrow_high_bond = 24
    narrow_low_bond = 21
    if (outside_air < narrow_low_bond):
        global_saving = -0.5
    elif (outside_air > narrow_high_bond):
        global_saving = 0.5
    else:
        global_saving = 0
    global counter
    pre_heating_steps = 5
    if (up_bundary > narrow_high_bond) and (low_boundary < narrow_low_bond):
        wild_boundary = 1
        if (outside_air < narrow_low_bond):
            saving_interval = - 3
            highh_b = narrow_high_bond
            target = (up_bundary + low_boundary) / 2 + global_saving + saving_interval
            back_bound = 21
            step_length = abs(back_bound - target) / pre_heating_steps
            target = target + max(0, (counter - (25 - pre_heating_steps))) * step_length
            counter += 1
            loww_b = target - 1
        else:
            saving_interval = 3
            loww_b = narrow_low_bond
            target = (up_bundary + low_boundary) / 2 + global_saving + saving_interval
            back_bound = 21
            step_length = abs(back_bound - target) / pre_heating_steps
            target = target - max(0, (counter - (25 - pre_heating_steps))) * step_length
            counter += 1
            highh_b = target + 1

    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2 + global_saving
        counter = 0
        loww_b = low_boundary
        highh_b = up_bundary

    diff_curr = abs(indoor_air_temp - target)

    if (loww_b < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    weight_reward_for_thermal = 1
    weight_penality_for_thermal = 1
    weight_penality_for_thermal_2 = 2
    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized

            R_ = r_t
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            R_ = weight_penality_for_thermal_2 * weight_penality_for_thermal * r_t + (
                        1 - weight_penality_for_thermal) * r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            R_ = weight_penality_for_thermal_2 * weight_penality_for_thermal * r_t + (
                        1 - weight_penality_for_thermal) * r_e
            s_ = 3

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, weight_reward_for_thermal, s_, loww_b, highh_b


def the_one_under_test_time_2(low_boundary, up_bundary, indoor_air_temp, current_consumption, energy_couption,
                              outside_air):
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

    narrow_high_bond = 24
    narrow_low_bond = 21
    if (outside_air < narrow_low_bond):
        global_saving = -0.5
    elif (outside_air > narrow_high_bond):
        global_saving = 0.5
    else:
        global_saving = 0
    global counter_2
    pre_heating_steps = 10
    if (up_bundary > narrow_high_bond) and (low_boundary < narrow_low_bond):
        wild_boundary = 1
        if (outside_air < narrow_low_bond):
            saving_interval = - 3
            highh_b = narrow_high_bond
            target = (up_bundary + low_boundary) / 2 + global_saving + saving_interval
            # back_bound = 21
            # step_length = abs(back_bound - target) / pre_heating_steps
            # target = target + max(0, (counter_2 - (25 - pre_heating_steps))) * step_length
            # counter_2 += 1
            loww_b = target - 1
        else:
            saving_interval = 3
            loww_b = narrow_low_bond
            target = (up_bundary + low_boundary) / 2 + global_saving + saving_interval
            # back_bound = 21
            # step_length = abs(back_bound - target) / pre_heating_steps
            # target = target - max(0, (counter_2 - (25 - pre_heating_steps))) * step_length
            # counter_2 += 1
            highh_b = target + 1

    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2 + global_saving
        counter_2 = 0
        loww_b = low_boundary
        highh_b = up_bundary

    diff_curr = abs(indoor_air_temp - target)

    if (loww_b < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    weight_reward_for_thermal = 1
    weight_penality_for_thermal = 0.5
    weight_penality_for_thermal_2 = 1
    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized

            R_ = r_t
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            R_ = weight_penality_for_thermal_2 * weight_penality_for_thermal * r_t + (
                    1 - weight_penality_for_thermal) * r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            R_ = weight_penality_for_thermal_2 * weight_penality_for_thermal * r_t + (
                    1 - weight_penality_for_thermal) * r_e
            s_ = 3

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, weight_reward_for_thermal, s_, loww_b, highh_b


counter_2 = 0


def the_one_under_test_time_Nov_2nd(low_boundary, up_bundary, indoor_air_temp, current_consumption, energy_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        current_consumption_normalized = 1 * (current_consumption - min_) / (max_ - min_) - 0
    else:
        current_consumption_normalized = 0

    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
    else:
        wild_boundary = 0

    global counter_2
    if (wild_boundary == 1):
        lowest = (up_bundary + low_boundary) / 2 - 3
        back_bound = 21
        step_length = (back_bound - lowest) / 25
        # if(counter_2>=4):
        #     a=counter_2-4
        #
        # else:
        #     target = lowest
        target = lowest + counter_2 * step_length
        counter_2 += 1

        loww_b = target - 1  #
        highh_b = 24

    else:
        target = (up_bundary + low_boundary) / 2
        counter_2 = 0
        loww_b = low_boundary
        highh_b = up_bundary

    if (loww_b < indoor_air_temp < highh_b):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    diff_curr = abs(indoor_air_temp - target)

    weight_reward_for_thermal = 0.8
    weight_penality_for_thermal = 0.8
    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized

            R_ = r_t
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.8
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.6
            R_ = weight_penality_for_thermal * r_t + (1 - weight_penality_for_thermal) * r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (highh_b < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.6
            R_ = weight_penality_for_thermal * r_t + (1 - weight_penality_for_thermal) * r_e
            # R_ = r_t +  r_e
            s_ = 3

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, weight_reward_for_thermal, s_, loww_b, highh_b


def the_one_under_test_time_1101_best(low_boundary, up_bundary, indoor_air_temp, current_consumption, energy_couption):
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

    if (wild_boundary == 1):
        # counter+=1
        # target = (low_boundary + 4)+ counter*0.2
        lowest = (up_bundary + low_boundary) / 2 - 3
        back_bound = 21
        step_length = (back_bound - lowest) / 25
        target = lowest + counter * step_length
        counter += 1
        loww_b = target  # -1
    else:
        # global counter
        counter = 0
        loww_b = low_boundary

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
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.8
            R_ = weight_reward_for_thermal * r_t + (1 - weight_reward_for_thermal) * r_e
            s_ = 1

    else:
        if (loww_b > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            # weight = diff_curr/(indoor_air_temp)
            # weight = 0.6
            R_ = weight_penality_for_thermal * r_t + (1 - weight_penality_for_thermal) * r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            # weight = 0.6
            R_ = weight_penality_for_thermal * r_t + (1 - weight_penality_for_thermal) * r_e
            # R_ = r_t +  r_e
            s_ = 3
        # else:
        #     r_t = penality_(diff_curr)
        #     r_e=0
        #     R_ = r_t
        #     s_ = 4

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, weight_reward_for_thermal, s_, loww_b


def the_one_under_test_wo_e(low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption,
                            previous_couption):
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


def the_one_under_test_best_air_1(low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption,
                                  heat_consumption, energy_couption_cooling, energy_couption_heating):
    scenario = 0
    # thermal part:
    thermal_concession = 3
    if (up_bundary > 24) and (low_boundary < 21):

        ref = (up_bundary + low_boundary) / 2 - thermal_concession
    else:
        ref = (up_bundary + low_boundary) / 2

    real_difference = abs(indoor_air_temp - ref)

    # thertical_difference=abs(indoor_air_temp - ref)
    edge = 0  # I dont think put a counpon is a good idea here
    if ((low_boundary + edge) <= indoor_air_temp <= (up_bundary - edge)):
        current_indoor_state = 1
    elif ((low_boundary) <= indoor_air_temp < (low_boundary + edge)) or (
            (up_bundary - edge) <= indoor_air_temp <= (up_bundary)):
        current_indoor_state = 1.5
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_reward = reward_(real_difference)
        energy_reward = 0

    elif current_indoor_state == 1.5:
        thermal_reward = penality_(real_difference)
        if (indoor_air_temp < (low_boundary + edge)) and (cool_consumption == 0) and (
                heat_consumption > 0):  # Too cold--> not heating enough
            couption = np.array(energy_couption_heating)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1
        elif (indoor_air_temp >= (up_bundary - edge)) and (cool_consumption > 0) and (
                heat_consumption == 0):  # Too hot but not cooling enough
            couption = np.array(energy_couption_cooling)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1
        else:
            energy_reward = -1

    else:
        thermal_reward = penality_(real_difference)
        energy_consumption_threshold = 10
        # energy part:
        #####cold scenarios####
        if (indoor_air_temp <= low_boundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption < energy_consumption_threshold):  # Too cold but still cooling
            # couption = np.array(energy_couption_cooling)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu=max(1, (max_-min_))
            energy_reward = -1  # max(-1, -(1 * (cool_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 1

        elif (indoor_air_temp <= low_boundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too cold--> heating but not heating enough
            couption = np.array(energy_couption_heating)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 2

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # not cold but not working

            energy_reward = -1
            scenario = 3

        #####hot scenarios####
        elif (indoor_air_temp >= up_bundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too hot but still heating
            # couption = np.array(energy_couption_heating)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -1  # max(-1,-(1 * (heat_consumption - min_) / (bei_chu_shu) - 0))

            # energy_reward = penality_(thertical_difference)
            scenario = 4

        elif (indoor_air_temp >= up_bundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # Too hot but not cooling enough
            couption = np.array(energy_couption_cooling)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1)

            # energy_reward = penality_(thertical_difference)
            scenario = 5

        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # too hot but do not working
            energy_reward = -1
            # energy_reward = penality_(thertical_difference)
            scenario = 6

        else:
            energy_reward = 0

            # energy_reward = penality_(thertical_difference)
            scenario = 7

    final_reward = thermal_reward  # + energy_reward

    return final_reward, thermal_reward, energy_reward, current_indoor_state, ref, scenario


good_energy_couption_cooling, good_energy_couption_heating, bad_energy_couption_cooling, bad_energy_couption_heating = [], [], [], []


def the_one_under_test_best_air_2(low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption,
                                  heat_consumption, fan_consumption, couption_cooling_3, couption_heating_3,
                                  couption_fun_3, couption_all_3):
    scenario = 0
    # thermal part:

    thermal_concession = 1
    if (up_bundary > 24) and (low_boundary < 21):

        ref = (up_bundary + low_boundary) / 2 - thermal_concession
    else:
        ref = (up_bundary + low_boundary) / 2

    real_difference = abs(indoor_air_temp - ref)

    # thertical_difference=abs(indoor_air_temp - ref)
    if (low_boundary <= indoor_air_temp <= up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0
    saving_mode = 0
    if (up_bundary > 24) and (low_boundary < 20):
        saving_mode = 1
    # reward mechanism
    if current_indoor_state == 1 and saving_mode == 0:

        thermal_reward = reward_(real_difference)
        energy_reward = 0
        final_reward = thermal_reward
        scenario = 0

    elif current_indoor_state == 1 and saving_mode == 1:

        thermal_reward = reward_(real_difference)
        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.8
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 1


    else:  # (current_indoor_state == 0 and saving_mode == 0) and (current_indoor_state == 0 and saving_mode == 0)
        thermal_reward = penality_(real_difference)
        energy_consumption_threshold = 20
        # energy part:
        #####cold scenarios####
        if (indoor_air_temp <= low_boundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption < energy_consumption_threshold):  # Too cold but still cooling
            energy_reward = -1  # max(-1, -(1 * (cool_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 2

        elif (indoor_air_temp <= low_boundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too cold--> heating but not heated enough
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_h = max(-1, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)

            # couption_f = np.array(couption_fun_3)
            # max_f = max(couption_f) if len(couption_f) > 0 else 0
            # min_f = min(couption_f) if len(couption_f) > 0 else 0
            # bei_chu_shu_f = max(1, (max_f - min_f))
            # energy_f = max(-1, ((fan_consumption - min_f) / (bei_chu_shu_f) - 0) - 1)

            energy_reward = energy_h
            scenario = 3

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too cold but not working
            energy_reward = -1
            scenario = 4

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too cold but only working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 5

        #####hot scenarios####
        elif (indoor_air_temp >= up_bundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too hot but still heating
            energy_reward = -1  # max(-1,-(1 * (heat_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 6

        elif (indoor_air_temp >= up_bundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # Too hot but not cooling enough
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 7

        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too hot but do not working
            energy_reward = -1
            scenario = 8
        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too hot but noly working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 9
        else:
            energy_reward = -1
            scenario = 10

        weight_thermal = 0.8
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, ref, scenario


def the_one_under_test_best_air_3(low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption,
                                  heat_consumption, fan_consumption, couption_cooling_3, couption_heating_3,
                                  couption_fun_3, couption_all_3):
    scenario = 0
    # thermal part:

    thermal_concession = 1
    if (up_bundary > 24) and (low_boundary < 21):

        ref = (up_bundary + low_boundary) / 2 - thermal_concession
    else:
        ref = (up_bundary + low_boundary) / 2

    real_difference = abs(indoor_air_temp - ref)

    # thertical_difference=abs(indoor_air_temp - ref)
    if (low_boundary <= indoor_air_temp <= up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0
    saving_mode = 0
    if (up_bundary > 24) and (low_boundary < 20):
        saving_mode = 1
    # reward mechanism
    if current_indoor_state == 1 and saving_mode == 0:

        thermal_reward = reward_(real_difference)
        energy_reward = 0
        final_reward = thermal_reward
        scenario = 0

    elif current_indoor_state == 1 and saving_mode == 1:

        thermal_reward = reward_(real_difference)
        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.5
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 1


    else:  # (current_indoor_state == 0 and saving_mode == 0) and (current_indoor_state == 0 and saving_mode == 0)
        thermal_reward = penality_(real_difference)
        energy_consumption_threshold = 20
        # energy part:
        #####cold scenarios####
        if (indoor_air_temp <= low_boundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption < energy_consumption_threshold):  # Too cold but still cooling
            energy_reward = -1  # max(-1, -(1 * (cool_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 2

        elif (indoor_air_temp <= low_boundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too cold--> heating but not heated enough
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_h = max(-1, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)

            # couption_f = np.array(couption_fun_3)
            # max_f = max(couption_f) if len(couption_f) > 0 else 0
            # min_f = min(couption_f) if len(couption_f) > 0 else 0
            # bei_chu_shu_f = max(1, (max_f - min_f))
            # energy_f = max(-1, ((fan_consumption - min_f) / (bei_chu_shu_f) - 0) - 1)

            energy_reward = energy_h
            scenario = 3

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too cold but not working
            energy_reward = -1
            scenario = 4

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too cold but only working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 5

        #####hot scenarios####
        elif (indoor_air_temp >= up_bundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too hot but still heating
            energy_reward = -1  # max(-1,-(1 * (heat_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 6

        elif (indoor_air_temp >= up_bundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # Too hot but not cooling enough
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 7

        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too hot but do not working
            energy_reward = -1
            scenario = 8
        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too hot but noly working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 9
        else:
            energy_reward = -1
            scenario = 10
        weight_thermal = 0.7
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, current_indoor_state, ref, scenario



counter
cushion_done=0
def the_one_under_test_best_air_price(low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    scenario = 0
    ####price part
    global counter
    global cushion_done
    counter += 1
    if (counter < 25) and (cushion_done==0):
        price_cushion.append(price)
        # price_cushion_np = np.array(price_cushion)
        if(counter==24):
            counter = 0
            # print(price_cushion_np)
            cushion_done=1
    elif (counter < 25) and (cushion_done==1):
        if (counter == 24):
            counter = 0
            cushion_done = 1
    print('counter: ',counter)
    if( cushion_done==1):
        price_cushion_np = np.array(price_cushion)
        predicted_price=price_cushion_np[counter]
    else:
        predicted_price=price

    load_shaping_mode = 0  # 0--> general cases when price not change    1--> the price will rise
    energy_saving = 0  # 0--> not saving    1--> saving wo price concern
    occupied_period = 0  # 0--> unoccupied    1--> occupied

    if (predicted_price - price) > 0:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0

    # thermal part:
    thermal_concession = 1
    if (up_bundary > 24) and (low_boundary < 21):
        # ref = (up_bundary + low_boundary) / 2 - thermal_concession
        occupied_period = 0
    else:
        ref = (up_bundary + low_boundary) / 2
        occupied_period = 1

    # if(occupied_period==0) and (load_shaping_mode==0):
    #     if(out_door_air<15): #winter
    #         ref=low_boundary
    #     else:
    #         ref=up_bundary
    # elif(occupied_period==0) and (load_shaping_mode==1):

    real_difference = abs(indoor_air_temp - ref)

    # thertical_difference=abs(indoor_air_temp - ref)
    if (low_boundary <= indoor_air_temp <= up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0
    saving_mode = 0
    if (up_bundary > 24) and (low_boundary < 20):
        saving_mode = 1
    # reward mechanism
    if current_indoor_state == 1 and saving_mode == 0:

        thermal_reward = reward_(real_difference)
        energy_reward = 0
        final_reward = thermal_reward
        scenario = 0

    elif current_indoor_state == 1 and saving_mode == 1:

        thermal_reward = reward_(real_difference)
        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.5
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 1


    else:  # (current_indoor_state == 0 and saving_mode == 0) and (current_indoor_state == 0 and saving_mode == 0)
        thermal_reward = penality_(real_difference)
        energy_consumption_threshold = 20
        # energy part:
        #####cold scenarios####
        if (indoor_air_temp <= low_boundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption < energy_consumption_threshold):  # Too cold but still cooling
            energy_reward = -1  # max(-1, -(1 * (cool_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 2

        elif (indoor_air_temp <= low_boundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too cold--> heating but not heated enough
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_h = max(-1, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)

            # couption_f = np.array(couption_fun_3)
            # max_f = max(couption_f) if len(couption_f) > 0 else 0
            # min_f = min(couption_f) if len(couption_f) > 0 else 0
            # bei_chu_shu_f = max(1, (max_f - min_f))
            # energy_f = max(-1, ((fan_consumption - min_f) / (bei_chu_shu_f) - 0) - 1)

            energy_reward = energy_h
            scenario = 3

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too cold but not working
            energy_reward = -1
            scenario = 4

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too cold but only working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 5

        #####hot scenarios####
        elif (indoor_air_temp >= up_bundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too hot but still heating
            energy_reward = -1  # max(-1,-(1 * (heat_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 6

        elif (indoor_air_temp >= up_bundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # Too hot but not cooling enough
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 7

        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too hot but do not working
            energy_reward = -1
            scenario = 8
        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too hot but noly working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 9
        else:
            energy_reward = -1
            scenario = 10
        weight_thermal = 0.7
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price


counter
price_cushion_peak_heat_scenario      =np.array([0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444,0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1,1)
price_cushion_heat=np.array([0.2383,  0.2383, 0.2383, 0.2383, 0.2383, 0.2383, 0.2666,  0.2666,  0.2666,  0.2666,  0.2666,  0.2666, 0.2666, 0.2666, 0.2666, 0.2666,  0.2666,  0.2666,  0.2666,  0.2666,0.2666, 0.2666, 0.2383,0.2383]).reshape(-1,1)
def reward_function_w_flexibility (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    scenario = 0
    ####price part
    global counter
    if(counter  <= 21):
        predicted_price = price_cushion [counter + 2]
    elif(counter  == 22):
        predicted_price = price_cushion[0]
    elif (counter == 23):
        predicted_price = price_cushion[1]

    if (counter < 23):
        counter += 1
    else:
        counter=0

    predicted_price=predicted_price[0]
    load_shaping_mode = 0  # 0--> general cases when price not change    1--> the price will rise
    energy_saving = 0  # 0--> not saving    1--> saving wo price concern
    occupied_period = 0  # 0--> unoccupied    1--> occupied

    if predicted_price -  price > 0.0001:
        load_shaping_mode = 1
        # print('predicted_price ',predicted_price)
        # print('price ', price)
        # print (predicted_price -  price)
    else:
        load_shaping_mode = 0

    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    if (occupied_period==0) and (load_shaping_mode==0):
        sendentary_saving   =  1
        dynamic_load_shaping = 0
        ref=low_boundary
        real_difference = abs(indoor_air_temp - ref)
        thermal_reward = reward_(real_difference)

        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.5
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 1

    elif (occupied_period==0) and (load_shaping_mode==1):
        dynamic_load_shaping=1
        sendentary_saving = 0
        ref=24
        real_difference = abs(indoor_air_temp - ref)
        thermal_reward = reward_(real_difference)

        couption = np.array(couption_heating_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.5
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 2

    elif (occupied_period==1) and (load_shaping_mode==0):
        sendentary_saving   =  1
        dynamic_load_shaping = 0
        ref=(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        thermal_reward = reward_(real_difference)
        energy_reward = 0
        final_reward = thermal_reward
        scenario = 3

    elif (occupied_period==1) and (load_shaping_mode==1):
        sendentary_saving   =  1
        dynamic_load_shaping = 0
        ref=up_bundary
        real_difference = abs(indoor_air_temp - ref)
        thermal_reward = reward_(real_difference)

        couption = np.array(couption_heating_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        weight_thermal = 0.5
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 4
    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price,counter,occupied_period,load_shaping_mode

def reward_function_w_flexibility_1 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    ####global param
    global counter_1
    if (counter_1 <= 21):
        predicted_price = price_cushion[counter_1 + 2]
    elif (counter_1 == 22):
        predicted_price = price_cushion[0]
    elif (counter_1 == 23):
        predicted_price = price_cushion[1]

    if (counter_1 < 23):
        counter_1 += 1
    else:
        counter_1 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    # final_reward, thermal_reward, energy_reward,ref, scenario=reward_mechanism_new (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3)

    final_reward, thermal_reward, energy_reward, ref, scenario = reward_mechanism_Feb_29(occupied_period,
                                                                                         load_shaping_mode,
                                                                                         low_boundary,
                                                                                         up_bundary, indoor_air_temp,
                                                                                         all_consumption,
                                                                                         couption_all_3,
                                                                                         heat_consumption,
                                                                                         couption_heating_3,
                                                                                         cool_consumption,
                                                                                         couption_cooling_3
                                                                                         )

    # weight_energy = 1 - weight_thermal
    # final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, occupied_period, load_shaping_mode

def reward_function_w_flexibility_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    ####global param
    global counter_2
    if (counter_2 <= 21):
        predicted_price = price_cushion[counter_2 + 2]
    elif (counter_2 == 22):
        predicted_price = price_cushion[0]
    elif (counter_2 == 23):
        predicted_price = price_cushion[1]

    if (counter_2 < 23):
        counter_2 += 1
    else:
        counter_2 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        if(predicted_price==0.05413):
            load_shaping_mode = 1
        else:
            load_shaping_mode = 2
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1


    # final_reward, thermal_reward, energy_reward, ref, scenario = reward_mechanism_March_08 (occupied_period,
    #                                                                                      load_shaping_mode,
    #                                                                                      low_boundary,
    #                                                                                      up_bundary, indoor_air_temp,
    #                                                                                      all_consumption,
    #                                                                                      couption_all_3,
    #                                                                                      heat_consumption,
    #                                                                                      couption_heating_3,
    #                                                                                      cool_consumption,
    #                                                                                      couption_cooling_3
    #                                                                                      )

    final_reward, thermal_reward, energy_reward, ref, scenario = reward_mechanism_March_12 (price,
                                                                                            occupied_period,
                                                                                           load_shaping_mode,
                                                                                           low_boundary,
                                                                                           up_bundary, indoor_air_temp,
                                                                                           all_consumption,
                                                                                           couption_all_3,
                                                                                           heat_consumption,
                                                                                           couption_heating_3,
                                                                                           cool_consumption,
                                                                                           couption_cooling_3
                                                                                           )

    # weight_energy = 1 - weight_thermal
    # final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, occupied_period, load_shaping_mode

def reward_function_w_flexibility_3 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    #### global param
    global counter_3
    prediction_horizon=2
    if (counter_3 <= 21):
        predicted_price = price_cushion[counter_3 + prediction_horizon]
    elif (counter_3 == 22):
        predicted_price = price_cushion[0]
    elif (counter_3 == 23):
        predicted_price = price_cushion[1]

    if (counter_3 < 23):
        counter_3 += 1
    else:
        counter_3 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
            load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    # final_reward, thermal_reward, energy_reward,ref, scenario=reward_mechanism_new (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3)

    final_reward, thermal_reward, energy_reward, ref, scenario = reward_mechanism_March_12_2 (price,
                                                                                           occupied_period,
                                                                                           load_shaping_mode,
                                                                                           low_boundary,
                                                                                           up_bundary, indoor_air_temp,
                                                                                           all_consumption,
                                                                                           couption_all_3,
                                                                                           heat_consumption,
                                                                                           couption_heating_3,
                                                                                           cool_consumption,
                                                                                           couption_cooling_3,
                                                                                           out_door_air
                                                                                           )

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, occupied_period, load_shaping_mode

counter_0=0
def reward_function_w_flexibility_peak_heat (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption):
    #### global param
    global counter_0
    prediction_horizon = 3
    cushion = np.array(
        [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413,
         0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)

    if (counter_0 <= 20):
        predicted_price = cushion[counter_0 + prediction_horizon]
    elif (counter_0 == 21):
        predicted_price = cushion[0]
    elif (counter_0 == 22):
        predicted_price = cushion[1]
    elif (counter_0 == 23):
        predicted_price = cushion[2]

    if (counter_0 < 23):
        counter_0 += 1
    else:
        counter_0 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 4500
    energy_min = 0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015


    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        edge = 1 * (1-(1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        energy_weight = normalized_price

        if ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption==0):
            # reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = 1 # max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption> 0):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.4
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        energy_weight = 1 - normalized_price
        up_b_load_shaping = 24
        low_b_load_shaping = 21
        # edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_b_load_shaping  # + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)


    elif (occupied_period == 1):
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            energy_weight = normalized_price
            thermal_reward = reward_new(real_difference)
            noramlized_heat = ((heat_consumption - energy_min) / (energy_max - energy_min))
            noramlized_cool = ((cool_consumption - energy_min) / (energy_max - energy_min))
            if (out_door_air > ref):
                scenario = 3.0
                if (noramlized_heat == 0):
                    alpa = 1
                else:
                    alpa = 0
                energy_reward = (1 - noramlized_cool) * alpa
            else:
                scenario = 3.1
                if (noramlized_cool == 0):
                    alpa = 1
                else:
                    alpa = 0
                energy_reward = (1 - noramlized_heat)
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)) * alpa
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, energy_weight

def reward_function_w_flexibility_peak_cool (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption):
    #### global param
    global counter_0
    prediction_horizon = 3
    cushion = np.array(
        [0.0444,0.0444,0.0444,0.0444,0.0444,0.0444,0.0444,0.0444,0.0842,0.0842,0.0842,0.0842,0.0842,0.13814,0.13814,0.13814,0.13814,0.0842,0.0842,0.0842,0.0444,0.0444,0.0444,0.0444]).reshape(-1, 1)

    if (counter_0 <= 20):
        predicted_price = cushion[counter_0 + prediction_horizon]
    elif (counter_0 == 21):
        predicted_price = cushion[0]
    elif (counter_0 == 22):
        predicted_price = cushion[1]
    elif (counter_0 == 23):
        predicted_price = cushion[2]

    if (counter_0 < 23):
        counter_0 += 1
    else:
        counter_0 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 4500
    energy_min = 0

    max_price = max(cushion) + 0.002
    min_price = min(cushion) - 0.015


    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        edge = 1 * (1-(1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        energy_weight = normalized_price

        if ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption==0):
            # reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = 1 # max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption> 0):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.4
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        energy_weight = 1 - normalized_price
        up_b_load_shaping = 24
        low_b_load_shaping = 21
        # edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_b_load_shaping  # + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            if (out_door_air < ref):
                scenario = 2.0
                # reward case--> pre heating
                energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = reward_new(real_difference)
                final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
            else:
                # reward case--> pre cooling
                scenario = 2.1
                # reward case--> pre heating
                energy_reward = max(0, ((cool_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = reward_new(real_difference)
                final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            energy_weight = normalized_price
            thermal_reward = reward_new(real_difference)
            noramlized_heat = ((heat_consumption - energy_min) / (energy_max - energy_min))
            noramlized_cool = ((cool_consumption - energy_min) / (energy_max - energy_min))
            if (out_door_air > ref):
                scenario = 3.0
                if (noramlized_heat == 0):
                    alpa = 1
                else:
                    alpa = 0
                energy_reward = (1 - noramlized_cool) * alpa
            else:
                scenario = 3.1
                if (noramlized_cool == 0):
                    alpa = 1
                else:
                    alpa = 0
                energy_reward = (1 - noramlized_heat)
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)) * alpa
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, energy_weight

counter_2 =0
def reward_function_w_flexibility_peak_heat_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption):
    #### global param
    global counter_2
    prediction_horizon = 3
    cushion = np.array(
        [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413,
         0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)

    if (counter_2 <= 20):
        predicted_price = cushion[counter_2 + prediction_horizon]
    elif (counter_2 == 21):
        predicted_price = cushion[0]
    elif (counter_2 == 22):
        predicted_price = cushion[1]
    elif (counter_2 == 23):
        predicted_price = cushion[2]

    if (counter_2 < 23):
        counter_2 += 1
    else:
        counter_2 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1


    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 4500
    energy_min = 0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        energy_weight = 1 - normalized_price

        if ((low_boundary) < indoor_air_temp < up_bundary):
            # reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        energy_weight = 1 - normalized_price
        up_b_load_shaping = 24
        low_b_load_shaping = 21
        # edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_b_load_shaping  # + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary + 1) <= indoor_air_temp <= (up_bundary - 1):
            scenario = 3.1
            energy_weight = normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (low_boundary) < indoor_air_temp < (low_boundary - 1):
            scenario = 3.2
            energy_weight = 1 - normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (up_bundary - 1) < indoor_air_temp < (up_bundary):
            scenario = 3.2
            energy_weight = 1 - normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, energy_weight

counter_3=0
def reward_function_w_flexibility_peak_heat_3 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption):
    #### global param
    global counter_3
    prediction_horizon = 3
    cushion = np.array(
        [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413,
         0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)

    if (counter_3 <= 20):
        predicted_price = cushion[counter_3 + prediction_horizon]
    elif (counter_3 == 21):
        predicted_price = cushion[0]
    elif (counter_3 == 22):
        predicted_price = cushion[1]
    elif (counter_3 == 23):
        predicted_price = cushion[2]

    if (counter_3 < 23):
        counter_3 += 1
    else:
        counter_3 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 4500
    energy_min = 0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        edge = 1 * (1 - (1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        energy_weight = normalized_price

        if ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption == 0):
            # reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = 1  # max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) < indoor_air_temp < up_bundary) and (all_consumption > 0):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.4
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        energy_weight = 1 - normalized_price
        up_b_load_shaping = 24
        low_b_load_shaping = 21
        # edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_b_load_shaping  # + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)


    elif (occupied_period == 1):
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            energy_weight = normalized_price
            noramlized_heat = ((heat_consumption - energy_min) / (energy_max - energy_min))
            noramlized_cool = ((cool_consumption - energy_min) / (energy_max - energy_min))
            if (out_door_air > ref):
                scenario = 3.0
                if (noramlized_heat == 0):
                    #reward: outside is hot and not using heat
                    energy_reward = (1 - noramlized_cool)
                    thermal_reward = reward_new(real_difference)
                    final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
                else:
                    #penalty: outside is hot and but using heat
                    energy_reward = -noramlized_heat
                    thermal_reward = penality_new(real_difference)
                    final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))

            else:
                scenario = 3.1
                if (noramlized_cool == 0):
                    # reward: outside is cold and not using cooling
                    energy_reward = (1 - noramlized_heat)
                    thermal_reward = reward_new(real_difference)
                    final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
                else:
                    #penalty: outside is cold and using cooling
                    energy_reward = -noramlized_cool
                    thermal_reward = penality_new(real_difference)
                    final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))

        elif (low_boundary >= indoor_air_temp):
                # penalty case
                scenario = 3.2
                thermal_reward = penality_new(real_difference)
                energy_weight = normalized_price
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                ### the more heating we use, the less energy penalty ###
                final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

        else:
                # penalty case
                scenario = 3.3
                thermal_reward = penality_new(real_difference)
                energy_weight = normalized_price
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                ### the more cooling we use, the less energy penalty ###
                final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, energy_weight

def reward_function_w_flexibility_3_heat (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, fan_consumption, all_consumption, couption_cooling_3, couption_heating_3,couption_fun_3, couption_all_3):
    #### global param
    global counter_3
    prediction_horizon=2
    if (counter_3 <= 21):
        predicted_price = price_cushion_heat[counter_3 + prediction_horizon]
    elif (counter_3 == 22):
        predicted_price = price_cushion_heat[0]
    elif (counter_3 == 23):
        predicted_price = price_cushion_heat[1]

    if (counter_3 < 23):
        counter_3 += 1
    else:
        counter_3 = 0
    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
            load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if (up_bundary > 24) and (low_boundary < 21):
        occupied_period = 0
    else:
        occupied_period = 1

    # final_reward, thermal_reward, energy_reward,ref, scenario=reward_mechanism_new (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3)

    final_reward, thermal_reward, energy_reward, ref, scenario = reward_mechanism_March_12_2_heat (price,
                                                                                           occupied_period,
                                                                                           load_shaping_mode,
                                                                                           low_boundary,
                                                                                           up_bundary, indoor_air_temp,
                                                                                           all_consumption,
                                                                                           couption_all_3,
                                                                                           heat_consumption,
                                                                                           couption_heating_3,
                                                                                           cool_consumption,
                                                                                           couption_cooling_3,
                                                                                           out_door_air
                                                                                           )

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, occupied_period, load_shaping_mode

def reward_mechanism (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))

        if (low_boundary < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            weight_thermal = 0.2
        else:
            thermal_reward = penality_(real_difference)
            energy_reward = 0
            weight_thermal = 1


    elif (occupied_period == 0) and (load_shaping_mode == 1):
        scenario = 2
        ref = (up_bundary + low_boundary) / 2
        edge = 1
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < 24):
        # if (low_boundary < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):  # good states and energy consertive
            thermal_reward = reward_(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.1

        else:
            thermal_reward = penality_(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = min(0, ((heat_consumption - min_) / (bei_chu_shu) - 0)-1)
            # energy_reward = 0
            weight_thermal = 0.1

    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_(real_difference)
            weight_thermal = 1.1
        else:
            thermal_reward = penality_(real_difference)
            weight_thermal = 1.1

        energy_reward = 0

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = up_bundary
        real_difference = abs(indoor_air_temp - ref)

        # ref = (up_bundary + low_boundary) / 2
        # edge = 1
        # real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.1

        else:
            # thermal_reward = penality_(real_difference)
            # energy_reward = 0
            # weight_thermal = 1

            thermal_reward = penality_(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = min(0, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)
            # energy_reward = 0
            weight_thermal = 0.1

    return(thermal_reward,energy_reward,weight_thermal,ref,scenario)

def reward_mechanism_new (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))

        if ((low_boundary-1) < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_new (real_difference)
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            weight_thermal = 0.1
        else:
            thermal_reward = penality_new (real_difference)
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))
            weight_thermal = 1

        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    elif (occupied_period == 0) and (load_shaping_mode == 1):
        scenario = 2
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp <= 24):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.0
            final_reward = thermal_reward * (1 + energy_reward)

        elif (low_boundary > indoor_air_temp):
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =  min(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = min(0,  ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_new (real_difference)
            weight_thermal = 1.0
        else:
            thermal_reward = penality_new (real_difference)
            weight_thermal = 1.0

        energy_reward = 0
        weight_energy = 1 - weight_thermal
        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = up_bundary
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.0
            final_reward = thermal_reward * (1 + energy_reward)

        elif (low_boundary > indoor_air_temp):
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = min(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = min(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_Feb_26 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=2
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))

        if ((low_boundary+edge) < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_new (real_difference)
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            # weight_thermal = 0.1
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            thermal_reward = penality_new (real_difference)
            energy_reward = 1 #max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))
            # weight_thermal = 1
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 0) and (load_shaping_mode == 1):
        scenario = 2
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        edge=0
        if (low_boundary < indoor_air_temp <= 21):

            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            if(energy_reward>0):
                final_reward = energy_reward * (1 + thermal_reward)
                scenario = 2.1
            else:
                scenario = 2.12
                thermal_reward = penality_new(real_difference)
                energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
                final_reward = thermal_reward * (1 + energy_reward)
        elif(21 < indoor_air_temp <= 24):
            scenario = 2.13
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (1- (all_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to do NOT use energy **thermal reward is plus
            final_reward = thermal_reward * (1 + energy_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =  max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 3.1
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1- ((cool_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = up_bundary
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 4.1
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.0
            final_reward = thermal_reward * (1 + energy_reward)
            # final_reward = energy_reward * (1 + thermal_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 4.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 4.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1- ((cool_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)

def reward_mechanism_Feb_28 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=2
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))

        if ((low_boundary+edge) < indoor_air_temp < up_bundary):
            indoor_state = 1
        else:
            indoor_state = 0

        if (indoor_state == 1):
            thermal_reward = reward_new (real_difference)
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            thermal_reward = penality_new (real_difference)
            energy_reward = 1
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 0) and (load_shaping_mode == 1):
        scenario = 2
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        edge=0
        if (low_boundary < indoor_air_temp <= 21):

            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            if(energy_reward>0):
                final_reward = energy_reward * (1 + thermal_reward)
                scenario = 2.1
            else:
                scenario = 2.12
                thermal_reward = penality_new(real_difference)
                energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
                final_reward = thermal_reward * (1 + energy_reward)
        elif(21 < indoor_air_temp <= 24):
            scenario = 2.13
            couption = np.array(couption_heating_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            thermal_reward = reward_new(real_difference)
            energy_reward = 1 # max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = 2 #thermal_reward * (1 + energy_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =  max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward_heat = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))

            couption_c = np.array(couption_cooling_3)
            max_c = max(couption_c) if len(couption_c) > 0 else 0
            min_c = min(couption_c) if len(couption_c) > 0 else 0
            bei_chu_shu_c = max(1, (max_c - min_c))
            energy_reward_cool = max(0, 1 - ((cool_consumption - min_c) / (bei_chu_shu_c) - 0))

            energy_reward= min(energy_reward_heat, energy_reward_cool)

            final_reward = thermal_reward * (1 + energy_reward)

        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1- ((cool_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = up_bundary
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 4.1
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
            weight_thermal = 0.0
            final_reward = thermal_reward * (1 + energy_reward)
            # final_reward = energy_reward * (1 + thermal_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 4.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 4.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1- ((cool_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_Feb_29 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=1
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary+edge) < indoor_air_temp < 21):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

        elif((low_boundary+edge) >= indoor_air_temp):
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            thermal_reward = penality_new(real_difference)
            # couption = np.array(couption_cooling_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))

            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 0) and (load_shaping_mode == 1):
        scenario = 2
        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        edge=2

        if (indoor_air_temp < 21):
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        elif (21 < indoor_air_temp < 23):
            scenario = 2.2
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  ((heat_consumption - min_) / (bei_chu_shu) - 0))
            thermal_reward = 1  # reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)


        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)


        # if ((low_boundary+edge)  < indoor_air_temp < 21):
        #     # thermal_reward = penality_new(real_difference)
        #     # couption = np.array(couption_heating_3)
        #     # max_ = max(couption) if len(couption) > 0 else 0
        #     # min_ = min(couption) if len(couption) > 0 else 0
        #     # bei_chu_shu = max(1, (max_ - min_))
        #     # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        #     #     # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
        #     # final_reward = thermal_reward * (1 + energy_reward)
        #
        #
        #     thermal_reward = 1 #reward_new(real_difference)
        #     couption = np.array(couption_heating_3)
        #     max_ = max(couption) if len(couption) > 0 else 0
        #     min_ = min(couption) if len(couption) > 0 else 0
        #     bei_chu_shu = max(1, (max_ - min_))
        #     energy_reward = max(0, (heat_consumption - min_) / bei_chu_shu)
        #         # final_reward = energy_reward * (1 + thermal_reward)
        #     final_reward = thermal_reward * (1 + energy_reward)
        #     scenario = 2.1
        #     # else:
        #     #     scenario = 2.12
        #     #     thermal_reward = penality_new(real_difference)
        #     #     energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        #     #     # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
        #     #     final_reward = thermal_reward * (1 + energy_reward)
        # elif(21 < indoor_air_temp <= 24):
        #     scenario = 2.2
        #     couption = np.array(couption_cooling_3)
        #     max_ = max(couption) if len(couption) > 0 else 0
        #     min_ = min(couption) if len(couption) > 0 else 0
        #     bei_chu_shu = max(1, (max_ - min_))
        #     energy_reward = 1 - ((cool_consumption - min_) / bei_chu_shu)
        #     thermal_reward = 1 #reward_new(real_difference)
        #     final_reward = thermal_reward * (1 + energy_reward)
        #
        # elif ((low_boundary+edge) >= indoor_air_temp):
        #     scenario = 2.3
        #     thermal_reward = penality_new(real_difference)
        #     couption = np.array(couption_heating_3)
        #     max_ = max(couption) if len(couption) > 0 else 0
        #     min_ = min(couption) if len(couption) > 0 else 0
        #     bei_chu_shu = max(1, (max_ - min_))
        #     energy_reward =  max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
        #     # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
        #     final_reward = thermal_reward * (1 + energy_reward)
        # else:
        #     scenario = 2.3
        #     thermal_reward = penality_new(real_difference)
        #     couption = np.array(couption_heating_3)
        #     max_ = max(couption) if len(couption) > 0 else 0
        #     min_ = min(couption) if len(couption) > 0 else 0
        #     bei_chu_shu = max(1, (max_ - min_))
        #     energy_reward = max(0,  ((heat_consumption - min_) / (bei_chu_shu) - 0))
        #     final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = (up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward_heat = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))

            couption_c = np.array(couption_cooling_3)
            max_c = max(couption_c) if len(couption_c) > 0 else 0
            min_c = min(couption_c) if len(couption_c) > 0 else 0
            bei_chu_shu_c = max(1, (max_c - min_c))
            energy_reward_cool = max(0, 1 - ((cool_consumption - min_c) / (bei_chu_shu_c) - 0))

            energy_reward= min(energy_reward_heat, energy_reward_cool)
            final_reward = thermal_reward * (1 + energy_reward)

        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =    1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = up_bundary
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 4.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward_heat = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))

            couption_c = np.array(couption_cooling_3)
            max_c = max(couption_c) if len(couption_c) > 0 else 0
            min_c = min(couption_c) if len(couption_c) > 0 else 0
            bei_chu_shu_c = max(1, (max_c - min_c))
            energy_reward_cool = max(0, 1 - ((cool_consumption - min_c) / (bei_chu_shu_c) - 0))

            energy_reward = min(energy_reward_heat, energy_reward_cool)

            final_reward = thermal_reward * (1 + energy_reward)
            # final_reward = energy_reward * (1 + thermal_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 4.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 4.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1- ((cool_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_March_04 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    edge = 1  # it was 1 before
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        if ((ref) < indoor_air_temp < 21):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

        elif((ref) >= indoor_air_temp):
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)

        if (indoor_air_temp < (ref)):
            scenario = 2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        elif ((ref) <= indoor_air_temp < 24):
            scenario = 2.2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  (1 - ((all_consumption - min_) / bei_chu_shu)))
            thermal_reward = 1  # reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)


    elif (occupied_period == 1) and (load_shaping_mode == 0):
        scenario = 3
        ref = low_boundary #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)


            # couption = np.array(couption_heating_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            # energy_reward_heat = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            #
            # couption_c = np.array(couption_cooling_3)
            # max_c = max(couption_c) if len(couption_c) > 0 else 0
            # min_c = min(couption_c) if len(couption_c) > 0 else 0
            # bei_chu_shu_c = max(1, (max_c - min_c))
            # energy_reward_cool = max(0, 1 - ((cool_consumption - min_c) / (bei_chu_shu_c) - 0))
            #
            # energy_reward= min(energy_reward_heat, energy_reward_cool)
            # final_reward = thermal_reward * (1 + energy_reward)

        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =    1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 1) and (load_shaping_mode == 1):
        scenario = 4
        ref = (up_bundary + low_boundary) / 2 #up_bundary
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary < indoor_air_temp < ref):
            scenario = 4.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

        elif (ref <= indoor_air_temp < up_bundary):
            scenario = 4.3
            thermal_reward = penality_new(real_difference)

            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

        elif (low_boundary >= indoor_air_temp):
            scenario = 4.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)
        else:
            scenario = 4.4
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1- ((cool_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_March_05 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary+edge) < indoor_air_temp < 20):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = energy_reward * (1 + thermal_reward)

        elif((low_boundary+edge) >= indoor_air_temp):
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (indoor_air_temp < 21):
            scenario = 2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        elif (21 <= indoor_air_temp < 22):
            scenario = 2.2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  (1-((all_consumption - min_) / bei_chu_shu)))

            # couption = np.array(couption_cooling_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            # # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # energy_reward_cool = max(0, (1-((cool_consumption - min_) / bei_chu_shu)))
            # energy_reward= energy_reward_ALL* energy_reward_cool

            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)


        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)):
        scenario = 3
        ref = low_boundary
        edge=1  #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < (up_bundary)):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))

            # couption = np.array(couption_cooling_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))


            final_reward = thermal_reward * (1 + energy_reward)


        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 1) and (load_shaping_mode == 2):
            scenario = 5
            ref = (up_bundary + low_boundary) / 2   # up_bundary
            real_difference = abs(indoor_air_temp - ref)

            if (low_boundary < indoor_air_temp < ref):
                scenario = 5.1
                thermal_reward = reward_new(real_difference)

                couption = np.array(couption_heating_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
                final_reward = thermal_reward * (1 + energy_reward)

            elif (ref <= indoor_air_temp < up_bundary):
                scenario = 5.3
                thermal_reward = reward_new(real_difference)

                couption = np.array(couption_all_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                energy_reward = max(0, (1- ((all_consumption - min_) / bei_chu_shu)))

                # couption = np.array(couption_cooling_3)
                # max_ = max(couption) if len(couption) > 0 else 0
                # min_ = min(couption) if len(couption) > 0 else 0
                # bei_chu_shu = max(1, (max_ - min_))
                # energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))


                final_reward = thermal_reward * (1 + energy_reward)

            elif (low_boundary >= indoor_air_temp):
                scenario = 5.2
                thermal_reward = penality_new(real_difference)
                couption = np.array(couption_heating_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                final_reward = thermal_reward * (1 + energy_reward)
            else:
                scenario = 5.4
                thermal_reward = penality_new(real_difference)
                couption = np.array(couption_cooling_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))
                final_reward = thermal_reward * (1 + energy_reward)

    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_March_06 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary+edge
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary+edge) < indoor_air_temp < 20):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

        elif((low_boundary+edge) >= indoor_air_temp):
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  20 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (indoor_air_temp < ref):
            scenario = 2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        elif (ref <= indoor_air_temp < 22):
            scenario = 2.2

            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  (1 - ((all_consumption - min_) / bei_chu_shu)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)


        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)):
        scenario = 3
        ref = low_boundary
        saving_boundary=(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))


            final_reward = thermal_reward * (1 + energy_reward)
            # final_reward = energy_reward * (1 + thermal_reward)


        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)



    return(final_reward, thermal_reward, energy_reward,ref, scenario)

def reward_mechanism_March_08 (occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary) < indoor_air_temp < 20):
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = energy_reward * (1 + thermal_reward)

        elif((low_boundary) >= indoor_air_temp):
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (indoor_air_temp < 21):
            scenario = 2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        elif (21 <= indoor_air_temp < 24):
            scenario = 2.2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  (1-((all_consumption - min_) / bei_chu_shu)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)


        else:
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 2)):
        scenario = 3
        ref = low_boundary
        edge=1  #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < (up_bundary)):
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))

            # couption = np.array(couption_cooling_3)
            # max_ = max(couption) if len(couption) > 0 else 0
            # min_ = min(couption) if len(couption) > 0 else 0
            # bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((cool_consumption - min_) / (bei_chu_shu) - 0))


            final_reward = thermal_reward * (1 + energy_reward)


        elif(low_boundary >= indoor_air_temp):
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)


    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_March_12 (price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3 ):
    #0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    epsilon = 0.0001
    price = round(price, 4)

    max_price = 0.09
    min_price = 0.0432
    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary) < indoor_air_temp < 20):
            #reward case
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0)) * ( normalized_price)
            final_reward = thermal_reward * (1 + energy_reward)


        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            # penalty case
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            final_reward = thermal_reward * (1 + energy_reward)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (20 <= indoor_air_temp < 22):
            # reward case
            scenario = 2.2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0,  (1-((all_consumption - min_) / bei_chu_shu))) *(normalized_price)
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 + energy_reward)

        elif (indoor_air_temp < 20):
            # penalty case
            scenario = 2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            # penalty case
            scenario = 2.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            final_reward = thermal_reward * (1 + energy_reward)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 2)):
        scenario = 3
        ref = low_boundary
        edge=1  #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary) < indoor_air_temp < (up_bundary):
            # reward case
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            if(heat_consumption>10 or cool_consumption>10):
                couption = np.array(couption_all_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0)) * (1-normalized_price)

            else:
                energy_reward = 1*(normalized_price)

            final_reward = thermal_reward * (1 + energy_reward)


        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   1 - ( (cool_consumption - min_) / bei_chu_shu )  # encouage to use more energy **thermal reward is minus
            final_reward = thermal_reward * (1 + energy_reward)


    return(final_reward, thermal_reward, energy_reward,ref, scenario)

def reward_mechanism_March_12_2 (price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3,out_door_air ):
    #0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    epsilon = 0.0001
    price = round(price, 4)

    max_price=0.09
    min_price=0.0432
    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price =  1* (0.0444 - min_price) / (max_price - min_price)

    elif abs(price - 0.05413) < epsilon:
        normalized_price =  1* (0.05413 -min_price) / (max_price - min_price)

    elif abs(price - 0.0888) < epsilon:
        normalized_price =  1* (0.0888 -min_price) / (max_price - min_price)

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1


    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary) < indoor_air_temp < 20):
            #reward case
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1 + energy_reward)
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        else:
            # penalty case
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))

            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (21 <= indoor_air_temp < 22):
            # reward case
            scenario = 2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0, (1 - ((all_consumption - min_) / bei_chu_shu)))
            thermal_reward = reward_new(real_difference)
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

        elif (indoor_air_temp < 21):
            # penalty case
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                # encouage to use more energy **thermal reward is minus, the more heating used, the less penalty
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1- normalized_price) + energy_reward * ( normalized_price)


        else:
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 2)):
        scenario = 3
        ref = low_boundary
        edge=1  #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference =  abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < (up_bundary)):
            # reward case
            scenario = 3.1
            thermal_reward = reward_new(real_difference)
            if (all_consumption <= 10):
                energy_reward = 1
            else:
                couption = np.array(couption_all_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                energy_reward = max(0, (1 - ((all_consumption - min_) / bei_chu_shu)))


            final_reward =  thermal_reward  *(1- normalized_price)  +  energy_reward *( normalized_price )

        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   -(1 - ( (cool_consumption - min_) / bei_chu_shu ))  # encouage to use more energy **thermal reward is minus
            # final_reward = thermal_reward * (1 + energy_reward)
            final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)


    return(final_reward, thermal_reward, energy_reward,ref, scenario)

def reward_mechanism_April_29_peak_heat (price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, heat_consumption, cool_consumption ,out_door_air):
    #0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    edge = 1
    epsilon = 0.0001
    price = round(price, 4)
    energy_max=4500
    energy_min=0
    max_price=0.1    #  max=0.0888
    min_price=0.0332 #  min=0.0444
    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price =  1* (0.0444 - min_price) / (max_price - min_price)

    elif abs(price - 0.05413) < epsilon:
        normalized_price =  1* (0.05413 -min_price) / (max_price - min_price)

    elif abs(price - 0.0888) < epsilon:
        normalized_price =  1* (0.0888 -min_price) / (max_price - min_price)

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1


    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        ##################the parameters for heating or cooling###############################

        ref = low_boundary+ 5* (1-  (1/(1+ math.exp(-out_door_air)))  )
        saving_boundary_during_unoccupied=(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary) < indoor_air_temp < saving_boundary_during_unoccupied):
            #reward case
            thermal_reward = reward_new(real_difference)

            energy_reward = max( 0,  1 - ((all_consumption - energy_min) / (energy_max-energy_min))
                                )
            final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            thermal_reward = penality_new (real_difference)

            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max-energy_min)))
            ### the more heating we use, the less energy penalty ###

            final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        else:
            # penalty case
            thermal_reward = penality_new(real_difference)

            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###

            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        up_b_load_shaping=24
        low_b_load_shaping=21
        ref = (up_bundary + low_boundary) / 2 -edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2

            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min))
                                )
            thermal_reward = reward_new(real_difference)

            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)

            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###

            final_reward = thermal_reward * (1- normalized_price) + energy_reward * ( normalized_price)

        else: #indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)

            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###

            final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)) :

        ref = (up_bundary + low_boundary) / 2 -edge
        ##################the parameters for heating or cooling###############################


        #Note: changed to low boundary for further save energy
        real_difference =  abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < (up_bundary)):
            # reward case
            scenario = 3
            thermal_reward = reward_new(real_difference)

            energy_reward = max( 0,  1 - ((all_consumption - energy_min) / (energy_max-energy_min))
                                )


            final_reward =  thermal_reward  *(1- normalized_price)  +  energy_reward *( normalized_price )

        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###

            final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)


    return(final_reward, thermal_reward, energy_reward,ref, scenario)


def reward_mechanism_flexibility_for_pre_heating (price, predicted_price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, heat_consumption, cool_consumption ,out_door_air):
    #0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    epsilon = 0.0001
    price = round(price, 4)
    energy_max=4500
    energy_min=0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price =  1* (0.0444 - min_price) / (max_price - min_price) #0.00446
        normalized_price_prediction = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price =  1* (0.05413 -min_price) / (max_price - min_price) #0.2216
        normalized_price_prediction = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.0888) < epsilon:
        normalized_price =  1* (0.0888 -min_price) / (max_price - min_price) #0.9955
        normalized_price_prediction = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
        normalized_price_prediction=1
    else:
        normalized_price = 1
        normalized_price_prediction=1

    edge=1*((1/(1+ math.exp(-out_door_air))) )
    ref = low_boundary + edge
    real_difference = abs(indoor_air_temp - ref)
    if (occupied_period == 0) and (load_shaping_mode == 0):#night

        energy_weight=normalized_price
        #saving_boundary_during_unoccupied=(up_bundary + low_boundary) / 2

        if ((low_boundary) < indoor_air_temp < up_bundary):
            #reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = max( 0,  1 - ((all_consumption - energy_min) / (energy_max-energy_min)))
            final_reward = thermal_reward * ( 1-energy_weight) + energy_reward * (energy_weight)
        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new (real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max-energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * ( 1-energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)): #day
        energy_weight=1-normalized_price
        up_b_load_shaping=24
        low_b_load_shaping=21

        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1- energy_weight) + energy_reward * ( energy_weight)
        else: #indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)

    elif  ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 0)):
        energy_weight=normalized_price

        if (low_boundary < indoor_air_temp < (up_bundary)):
            # reward case
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            energy_reward = max( 0,  1 - ((all_consumption - energy_min) / (energy_max-energy_min)))
            final_reward =  thermal_reward  *(1- energy_weight)  +  energy_reward *( energy_weight )
        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)



    return(final_reward, thermal_reward, energy_reward,ref, scenario, energy_weight)

def reward_mechanism_flexibility_for_pre_heating_new_try (price, predicted_price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, heat_consumption, cool_consumption ,out_door_air):
    #0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    epsilon = 0.0001
    price = round(price, 4)
    energy_max=4500
    energy_min=0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price =  1* (0.0444 - min_price) / (max_price - min_price) #0.00446
        normalized_price_prediction = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price =  1* (0.05413 -min_price) / (max_price - min_price) #0.2216
        normalized_price_prediction = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.0888) < epsilon:
        normalized_price =  1* (0.0888 -min_price) / (max_price - min_price) #0.9955
        normalized_price_prediction = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
        normalized_price_prediction=1
    else:
        normalized_price = 1
        normalized_price_prediction=1

    edge=1*((1/(1+ math.exp(-out_door_air))) )
    ref = low_boundary + edge
    real_difference = abs(indoor_air_temp - ref)
    if (occupied_period == 0) and (load_shaping_mode == 0):#night

        energy_weight=1-normalized_price
        #saving_boundary_during_unoccupied=(up_bundary + low_boundary) / 2

        if ((low_boundary) < indoor_air_temp < up_bundary):
            #reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = min(max( 0,  1 - ((heat_consumption - energy_min) / (energy_max-energy_min))), max( 0,  1 - ((cool_consumption - energy_min) / (energy_max-energy_min))))
            final_reward = thermal_reward * ( 1-energy_weight) + energy_reward * (energy_weight)
        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new (real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max-energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * ( 1-energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)): #day
        energy_weight=1-normalized_price
        up_b_load_shaping=24
        low_b_load_shaping=21

        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1- energy_weight) + energy_reward * ( energy_weight)
        else: #indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1-energy_weight) + energy_reward * (energy_weight)

    elif  ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 0)):
        energy_weight=normalized_price

        if (low_boundary < indoor_air_temp < (up_bundary)):
            # reward case
            scenario = 3.1
            thermal_reward = reward_new(real_difference)

            # energy_reward = max( 0,  1 - ((all_consumption - energy_min) / (energy_max-energy_min)))
            energy_reward = min(max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min))),
                                max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min))))
            final_reward =  thermal_reward  *(1- energy_weight)  +  energy_reward *( energy_weight )
        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)



    return(final_reward, thermal_reward, energy_reward,ref, scenario, energy_weight)
def reward_mechanism_flexibility_for_pre_heating_new_try_2 (price, counter, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, heat_consumption, cool_consumption ,out_door_air):
    # 0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413
    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 4500
    energy_min = 0

    max_price = 0.0888 + 0.002
    min_price = 0.0444 - 0.015

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.0444) < epsilon:
        normalized_price = 1 * (0.0444 - min_price) / (max_price - min_price)  # 0.00446

    elif abs(price - 0.05413) < epsilon:
        normalized_price = 1 * (0.05413 - min_price) / (max_price - min_price)  # 0.2216

    elif abs(price - 0.0888) < epsilon:
        normalized_price = 1 * (0.0888 - min_price) / (max_price - min_price)  # 0.9955

    elif abs(price - 0.09) < epsilon:
        normalized_price = 1
    else:
        normalized_price = 1

    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        energy_weight = 1 - normalized_price

        if ((low_boundary) < indoor_air_temp < up_bundary):
            # reward case
            scenario = 1.1
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        energy_weight = 1 - normalized_price
        up_b_load_shaping = 24
        low_b_load_shaping = 21
        # edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_b_load_shaping  # + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = reward_new(real_difference)
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        edge = 1 * ((1 / (1 + math.exp(-out_door_air))))
        ref = low_boundary + edge
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary + 1) <= indoor_air_temp <= (up_bundary - 1):
            scenario = 3.1
            energy_weight = normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (low_boundary) < indoor_air_temp < (low_boundary - 1):
            scenario = 3.2
            energy_weight = 1 - normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (up_bundary - 1) < indoor_air_temp < (up_bundary):
            scenario = 3.2
            energy_weight = 1 - normalized_price
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = (thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight))
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)
        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_weight = normalized_price
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    return(final_reward, thermal_reward, energy_reward,ref, scenario, energy_weight)




def reward_mechanism_March_12_2_heat (price, occupied_period, load_shaping_mode, low_boundary, up_bundary, indoor_air_temp, all_consumption, couption_all_3, heat_consumption, couption_heating_3, cool_consumption,couption_cooling_3,out_door_air ):
    epsilon = 0.0001
    price = round(price, 4)
    #0.2383, 0.2666
    max_price = 0.2666 + 0.02
    min_price = 0.2383 - 0.02

    if abs(price - min_price) < epsilon:
        normalized_price = 0

    elif abs(price - 0.2666) < epsilon:
        normalized_price =  1* (0.2666 - min_price) / (max_price - min_price)

    elif abs(price - 0.2383) < epsilon:
        normalized_price =  1* (0.2383 -min_price) / (max_price - min_price)

    elif abs(price - max_price) < epsilon:
        normalized_price = 1



    if (occupied_period == 0) and (load_shaping_mode == 0):
        scenario = 1
        edge=0 # it was 1 before
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)

        if ((low_boundary) < indoor_air_temp < up_bundary):
            #reward case
            thermal_reward = reward_new(real_difference)
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(0, 1 - ((all_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        elif((low_boundary) >= indoor_air_temp):
            # penalty case
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * ( 1-normalized_price) + energy_reward * (normalized_price)

        # else:
        #     # penalty case
        #     thermal_reward = penality_new(real_difference)
        #     couption = np.array(couption_all_3)
        #     max_ = max(couption) if len(couption) > 0 else 0
        #     min_ = min(couption) if len(couption) > 0 else 0
        #     bei_chu_shu = max(1, (max_ - min_))
        #     energy_reward = -max(0, ((all_consumption - min_) / (bei_chu_shu) - 0))
        #     # final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif (occupied_period == 0) and (load_shaping_mode == 1):

        ref =  21 #(up_bundary + low_boundary) / 2
        real_difference = abs(indoor_air_temp - ref)
        if (21 <= indoor_air_temp < 22):
            # reward case
            scenario = 2
            couption = np.array(couption_all_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            energy_reward = max(0, (1 - ((all_consumption - min_) / bei_chu_shu)))
            thermal_reward = reward_new(real_difference)
            # final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

        elif (indoor_air_temp < 21):
            # penalty case
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1- normalized_price) + energy_reward * ( normalized_price)


        else:
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0, ((heat_consumption - min_) / (bei_chu_shu) - 0))
            # final_reward = thermal_reward * (1-normalized_price) + energy_reward * (normalized_price)

    elif ((occupied_period == 1) and (load_shaping_mode == 0)) or ((occupied_period == 1) and (load_shaping_mode == 1)) or ((occupied_period == 1) and (load_shaping_mode == 2)):
        scenario = 3
        ref = low_boundary
        edge=1  #(up_bundary + low_boundary) / 2
        #Note: changed to low boundary for further save energy
        real_difference =  abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < (up_bundary)):
            # reward case
            scenario = 3.1
            thermal_reward = reward_new(real_difference)
            if (all_consumption <= 10):
                energy_reward = 1
            else:
                couption = np.array(couption_all_3)
                max_ = max(couption) if len(couption) > 0 else 0
                min_ = min(couption) if len(couption) > 0 else 0
                bei_chu_shu = max(1, (max_ - min_))
                # energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))
                energy_reward = max(0, (1 - ((all_consumption - min_) / bei_chu_shu)))
            # final_reward =  thermal_reward  *(1- normalized_price)  +  energy_reward *( normalized_price )

        elif(low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new (real_difference)
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = -max(0,  1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))  # encouage to use more energy **thermal reward is minus
            # final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward =   -(1 - ( (cool_consumption - min_) / bei_chu_shu ))  # encouage to use more energy **thermal reward is minus


    # final_reward = thermal_reward * (1 - normalized_price) + energy_reward * (normalized_price)
    if (occupied_period == 0):
        energy_weight=max(normalized_price, 1-normalized_price)
        thermal_weight = min(normalized_price, 1 - normalized_price)
    else:
        thermal_weight = max(normalized_price, 1 - normalized_price)
        energy_weight = min(normalized_price, 1 - normalized_price)

    final_reward = thermal_reward * (thermal_weight) + energy_reward * (energy_weight)


    return(final_reward, thermal_reward, energy_reward,ref, scenario)

def the_one_under_test_best_air_4(low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption,
                                  heat_consumption, fan_consumption, couption_cooling_3, couption_heating_3,
                                  couption_fun_3, couption_all_3):
    scenario = 0
    # thermal part:

    thermal_concession = 3
    if (up_bundary > 24) and (low_boundary < 21):

        ref = (up_bundary + low_boundary) / 2 - thermal_concession
    else:
        ref = (up_bundary + low_boundary) / 2

    real_difference = abs(indoor_air_temp - ref)

    # thertical_difference=abs(indoor_air_temp - ref)
    if (low_boundary <= indoor_air_temp <= up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    saving_mode = 0
    if (up_bundary > 24) and (low_boundary < 20):
        saving_mode = 1
        weight_thermal = 0.7
        weight_energy = 1 - weight_thermal
    else:
        saving_mode = 0
        weight_thermal = 1
        weight_energy = 1 - weight_thermal

    # reward mechanism
    if current_indoor_state == 1 and saving_mode == 0:

        thermal_reward = reward_(real_difference)
        energy_reward = 0
        final_reward = thermal_reward
        scenario = 0

    elif current_indoor_state == 1 and saving_mode == 1:

        thermal_reward = reward_(real_difference)
        couption = np.array(couption_all_3)
        max_ = max(couption) if len(couption) > 0 else 0
        min_ = min(couption) if len(couption) > 0 else 0
        bei_chu_shu = max(1, (max_ - min_))
        energy_reward = max(0, 1 - ((heat_consumption - min_) / (bei_chu_shu) - 0))

        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward
        scenario = 1


    else:  # (current_indoor_state == 0 and saving_mode == 0) and (current_indoor_state == 0 and saving_mode == 0)
        thermal_reward = penality_(real_difference)
        energy_consumption_threshold = 50
        # energy part:
        #####cold scenarios####
        if (indoor_air_temp <= low_boundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption < energy_consumption_threshold):  # Too cold but still cooling
            energy_reward = -1  # max(-1, -(1 * (cool_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 2

        elif (indoor_air_temp <= low_boundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too cold--> heating but not heated enough
            couption = np.array(couption_heating_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_h = max(-1, ((heat_consumption - min_) / (bei_chu_shu) - 0) - 1)

            # couption_f = np.array(couption_fun_3)
            # max_f = max(couption_f) if len(couption_f) > 0 else 0
            # min_f = min(couption_f) if len(couption_f) > 0 else 0
            # bei_chu_shu_f = max(1, (max_f - min_f))
            # energy_f = max(-1, ((fan_consumption - min_f) / (bei_chu_shu_f) - 0) - 1)

            energy_reward = energy_h
            scenario = 3

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too cold but not working
            energy_reward = -1
            scenario = 4

        elif (indoor_air_temp <= low_boundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too cold but only working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 5

        #####hot scenarios####
        elif (indoor_air_temp >= up_bundary) and (cool_consumption < energy_consumption_threshold) and (
                heat_consumption > energy_consumption_threshold):  # Too hot but still heating
            energy_reward = -1  # max(-1,-(1 * (heat_consumption - min_) / (bei_chu_shu) - 0))
            scenario = 6

        elif (indoor_air_temp >= up_bundary) and (cool_consumption > energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold):  # Too hot but not cooling enough
            couption = np.array(couption_cooling_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, (1 * (cool_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 7

        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption <= energy_consumption_threshold):  # too hot but do not working
            energy_reward = -1
            scenario = 8
        elif (indoor_air_temp >= up_bundary) and (cool_consumption <= energy_consumption_threshold) and (
                heat_consumption <= energy_consumption_threshold) and (
                fan_consumption > energy_consumption_threshold):  # too hot but noly working on fan
            couption = np.array(couption_fun_3)
            max_ = max(couption) if len(couption) > 0 else 0
            min_ = min(couption) if len(couption) > 0 else 0
            bei_chu_shu = max(1, (max_ - min_))
            energy_reward = max(-1, ((fan_consumption - min_) / (bei_chu_shu) - 0) - 1)
            scenario = 9
        else:
            energy_reward = -1
            scenario = 10

        final_reward = weight_thermal * thermal_reward + weight_energy * energy_reward

    return final_reward, thermal_reward, energy_reward, current_indoor_state, ref, scenario


def the_one_under_test_1(low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption):
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
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            weight = 0.5  # 0.4
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

def the_one_under_test_1_refined (low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        current_consumption_normalized =(current_consumption - min_) / (max_ - min_)
    else:
        current_consumption_normalized = 0
    M=2
    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (up_bundary + low_boundary) / 2  -M
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
            thermal_weight =1
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = 1
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + ( energy_weight) * r_e
            s_ = 1

    else:
        if (low_boundary > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            thermal_weight = 0.6
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            thermal_weight = 0.6
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            # R_ = r_t + r_e
            s_ = 3
        else:
            r_t = penality_(diff_curr)
            R_ = r_t
            s_ = 4
    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, s_

def the_one_under_test_1_further (low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption):
    current_consumption_normalized =(current_consumption - energy_min) / (energy_max - energy_min)
    real_bound_up = 24
    real_bound_down = 21
    M=2
    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =low_boundary
    else:
        wild_boundary = 0
        target = (up_bundary + low_boundary) / 2 - 1

    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary==1):
        if(real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized
            thermal_weight =0.9
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = 0.8
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + ( energy_weight) * r_e
            s_ = 1

    else:
        thermal_weight = 0.8
        energy_weight = 1 - thermal_weight
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, s_

def the_one_under_test_2_refined (low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption):
    current_consumption_normalized =(current_consumption - energy_min) / (energy_max - energy_min)
    M = 3
    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = ((up_bundary + low_boundary) / 2) - M
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2) -1

    diff_curr = abs(indoor_air_temp - target)

    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)  # 1.1
            r_e = 1 - current_consumption_normalized
            thermal_weight = 1
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = 0.8
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1

    else:
        if (low_boundary > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            thermal_weight = 0.8
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            # R_ = r_t + r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            thermal_weight = 0.8
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            # R_ = r_t + r_e
            s_ = 3
        else:
            r_t = penality_(diff_curr)
            R_ = r_t
            s_ = 4
    return R_, r_t, r_e, current_indoor_state, wild_boundary, target, s_

def outdoor_function (outdoor_diff, alpa):
    return (1 -  math.exp(-alpa *outdoor_diff)) / (1 +  math.exp(-alpa * outdoor_diff))

def the_one_under_test_4_outdoor (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption):
    if (len(energy_couption) >= 1):
        energy_couption_ = np.array(energy_couption)
        max_ = max(energy_couption_)
        min_ = min(energy_couption_)
    else:
        max_ = 0
        min_ = 0
    if (max_ != min_):
        current_consumption_normalized =(current_consumption - min_) / (max_ - min_)
    else:
        current_consumption_normalized = 0

    ref = (up_bundary + low_boundary) / 2
    outdoor_diff= abs(ref-out_door_air)


    if (up_bundary > 24) and (low_boundary < 21):
        wild_boundary = 1
        target = (up_bundary + low_boundary) / 2
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
            thermal_weight =1
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = outdoor_function (outdoor_diff)
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + ( energy_weight) * r_e
            s_ = 1

    else:
        if (low_boundary > indoor_air_temp):  # too low
            r_t = penality_(diff_curr)
            r_e = current_consumption_normalized - 1
            thermal_weight = 0.6
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 2
        elif (up_bundary < indoor_air_temp):  # too high
            r_t = penality_(diff_curr)
            r_e = -current_consumption_normalized
            thermal_weight = 0.6
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 3
        else:
            r_t = penality_(diff_curr)
            R_ = r_t
            s_ = 4
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def optimized_outdoor_diff(x):
    beta =1
    delta= 17.37
    return 1/(1+math.exp(-beta *(x-delta)))

def optimized_outdoor_diff_delta (x, delta, season):

    if season=='winter': beta =-1
    else: beta =1
    return 1/(1+math.exp(beta *(x-delta)))
def reversed_optimized_outdoor_diff(x):
    beta = 1
    delta = 17.37
    return 1 / (1 + np.exp(beta * (x - delta)))

def the_one_under_test_outdoor_peak_heat (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption):

    current_consumption_normalized =(current_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up=24
    real_bound_down=21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target = ((up_bundary + low_boundary) / 2) - 1
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2) - 1

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = optimized_outdoor_diff (outdoor_diff)
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = optimized_outdoor_diff (outdoor_diff)
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + ( energy_weight) * r_e
            s_ = 1
    else:
        if (wild_boundary == 0): #[21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                thermal_weight = 0.7
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                thermal_weight = 0.7
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                thermal_weight=1
                R_ = r_t
                r_e=0
                s_ = 4
        else:#[14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                thermal_weight = 0.7
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                thermal_weight = 0.7
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                thermal_weight=1
                R_ = r_t
                r_e=0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_


def the_one_under_test_outdoor (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption):
    current_consumption_normalized = (current_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =  low_boundary +0  #((up_bundary + low_boundary) / 2) - 1 #real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2) # - 1 #

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_


def the_one_under_test_outdoor_delta_ (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, delta):
    current_consumption_normalized = (current_consumption - energy_min) / (energy_max - energy_min)
    season = 'winter'
    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        if delta==0:
            target = ((up_bundary + low_boundary) / 2)
        else:
            target =  low_boundary  #((up_bundary + low_boundary) / 2) - 1 #real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2)

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)


    if (low_boundary < indoor_air_temp < up_bundary):
        current_indoor_state = 1
    else:
        current_indoor_state = 0
    thermal_weight = optimized_outdoor_diff_delta(outdoor_diff, delta, season)
    if current_indoor_state == 1:

        r_t = reward_(diff_curr)
        r_e = 1 - current_consumption_normalized

        energy_weight = 1 - thermal_weight
        R_ = thermal_weight * r_t + (energy_weight) * r_e
        s_ = 0

    else:
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_


def the_one_under_test_outdoor_comparsion_fixed (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption, apla):
    current_consumption_normalized = (current_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =low_boundary# ((up_bundary + low_boundary) / 2) - 1  # real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2)  # real_bound_down  #

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = apla
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = apla
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_
def the_one_under_test_outdoor_comparsion_fixed_2 (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption):
    current_consumption_normalized = (current_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target = ((up_bundary + low_boundary) / 2) - 1  # real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2)  # real_bound_down  #

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = 0.1#optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = 0.1#optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def the_one_under_test_outdoor_for_cooling (low_boundary, up_bundary, indoor_air_temp, cool_consumption, out_door_air):
    normalized_cooling = (cool_consumption - energy_min) / (energy_max - energy_min)
    # normalized_heating = (heat_consumption - energy_min) / (energy_max - energy_min)
    # normalized_all=(all_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        # target = real_bound_up  # ((up_bundary + low_boundary) / 2) - 1
        target = up_bundary# ((up_bundary + low_boundary) / 2) +1
    else:
        wild_boundary = 0
        target =  ((up_bundary + low_boundary) / 2)

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        r_t = reward_(diff_curr)
        r_e = 1 - normalized_cooling
        energy_weight = 1 - thermal_weight
        R_ = thermal_weight * r_t + (energy_weight) * r_e
        s_ = 0

    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_



def the_one_under_test_outdoor_for_cooling_detla (low_boundary, up_bundary, indoor_air_temp, cool_consumption, out_door_air, detla):
    normalized_cooling = (cool_consumption - energy_min) / (energy_max - energy_min)
    # normalized_heating = (heat_consumption - energy_min) / (energy_max - energy_min)
    # normalized_all=(all_consumption - energy_min) / (energy_max - energy_min)
    season='summer'
    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        # target = real_bound_up  # ((up_bundary + low_boundary) / 2) - 1
        target = up_bundary# ((up_bundary + low_boundary) / 2) +1
    else:
        wild_boundary = 0
        target =  ((up_bundary + low_boundary) / 2)

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = optimized_outdoor_diff_delta(outdoor_diff, detla,season)
        r_t = reward_(diff_curr)
        r_e = 1 - normalized_cooling
        energy_weight = 1 - thermal_weight
        R_ = thermal_weight * r_t + (energy_weight) * r_e
        s_ = 0

    else:
        thermal_weight = optimized_outdoor_diff_delta(outdoor_diff, detla,season)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def the_one_under_test_outdoor_for_cooling_fixed (low_boundary, up_bundary, indoor_air_temp, cool_consumption, out_door_air, alpa):
    normalized_cooling = (cool_consumption - energy_min) / (energy_max - energy_min)
    # normalized_heating = (heat_consumption - energy_min) / (energy_max - energy_min)
    # normalized_all=(all_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target = up_bundary #((up_bundary + low_boundary) / 2) + 1
    else:
        wild_boundary = 0
        target =  ((up_bundary + low_boundary) / 2)

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        thermal_weight = alpa#reversed_optimized_outdoor_diff(outdoor_diff)
        r_t = reward_(diff_curr)
        r_e = 1 - normalized_cooling
        energy_weight = 1 - thermal_weight
        R_ = thermal_weight * r_t + (energy_weight) * r_e
        s_ = 0

    else:
        thermal_weight = alpa# reversed_optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = -normalized_cooling
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = normalized_cooling-1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def the_one_under_test_outdoor_2 (low_boundary, up_bundary, indoor_air_temp, current_consumption, out_door_air, energy_couption):
    current_consumption_normalized = (current_consumption - energy_min) / (energy_max - energy_min)

    # ref = (up_bundary + low_boundary) / 2
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target = real_bound_down #((up_bundary + low_boundary) / 2) - 1
    else:
        wild_boundary = 0
        target = real_bound_down #((up_bundary + low_boundary) / 2) - 1

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)

    if (wild_boundary == 1):
        if (real_bound_down < indoor_air_temp < real_bound_up):
            current_indoor_state = 1
        else:
            current_indoor_state = 0
    else:
        if (low_boundary < indoor_air_temp < up_bundary):
            current_indoor_state = 1
        else:
            current_indoor_state = 0

    if current_indoor_state == 1:
        if (wild_boundary == 0):  # [21-24]
            r_t = 1.0 * reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = optimized_outdoor_diff_TH (outdoor_diff)
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            thermal_weight = optimized_outdoor_diff(outdoor_diff)
            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = 0.3
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4
        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_(diff_curr)
                r_e = current_consumption_normalized - 1
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_(diff_curr)
                r_e = -current_consumption_normalized
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_



def the_one_under_test_1025_best(low_boundary, up_bundary, indoor_air_temp, current_consumption, action_,
                                 energy_couption):
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
            s_ = 0

        else:  # [15-30]
            r_t = reward_(diff_curr)
            r_e = 1 - current_consumption_normalized
            weight = 0.4
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


def the_one_under_test_1024(low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption,
                            previous_couption):  # best performane reward function by 1024
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

    return R_, r_t, r_e, current_indoor_state, wild_boundary, target


def the_one_under_test_1023(low_boundary, up_bundary, indoor_air_temp, current_consumption, action_, energy_couption,
                            previous_couption):
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
            r_t = 1 * reward_(diff_curr)  # 1.1
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

    return R_, r_t, r_e, current_indoor_state, wild_boundary
