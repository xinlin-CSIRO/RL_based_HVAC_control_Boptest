import numpy as np
import math
import math


def exponential_penality(input, scale=2):
    a = -((2) / (1 + math.exp(-input / scale))) + 1
    return a

def exponential_reward(input, scale=2):
    a = -((2) / (1 + math.exp(-input / scale))) + 2
    return a

def optimized_outdoor_diff(x):
    beta = 1
    delta = 17.37
    return 1 / (1 + math.exp(-beta * (x - delta)))

counter_0 = 0
counter_1 = 0
counter_2 = 0
counter_3 = 0
energy_min = 0
energy_max = 4500


def nonlinear_temperature_response(T, test_typo):
    """
    Calculate the nonlinear response of a system based on outdoor temperature.

    Parameters:
    - T (float): The outdoor temperature.
    - T_mid (float): The midpoint temperature where the function is zero.
    - s (float): The scale factor that controls the transition steepness.

    Returns:
    - float: The calculated value based on the temperature.
    """
    if test_typo == 'peak_heat_day':
        T_mid = 10  # for winter: 0, for summer:10
    else:
        T_mid = 10  # for winter: 0, for summer:10
    # s = 10
    s = 5
    scale = 1.5
    return scale * np.tanh((T - T_mid) / s)


def normalized_thermal_function(T_in, T_ref, b_up, b_down):
    diff_ = abs(T_in - T_ref)
    x_m = abs(b_up - b_down)
    x_normalized = diff_ / x_m
    return (x_normalized)


def penality_normalized_thermal_function(T_in, T_ref, b_up, b_down):
    diff_ = abs(T_in - T_ref)
    x_m = min(abs(T_in - b_up), abs(T_in - b_down))
    x_normalized = - x_m / diff_
    return (x_normalized)





def reward_function_w_flexibility (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

    cushion = np.array(
        [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
         0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(
        -1, 1)
    prediction_horizon = 2


    if test_typo == 'peak_heat_day':
        if(interval==1):
            cushion = np.array(
                [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
                 0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
            prediction_horizon = 2
        elif(interval==2):
            cushion = np.array(
                            [0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444
                ]).reshape(-1, 1)
            prediction_horizon = 3
    elif(test_typo == 'peak_cool_day'):
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        prediction_horizon = 2
    c_=0
    max_price =float(max(cushion)  + c_)
    min_price = float(min(cushion) - c_)
    # real_lowest_price=float(min(cushion))
    if counter_x <= (len(cushion)-1) - prediction_horizon:
        predicted_price = cushion[counter_x + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_x + prediction_horizon) % len(cushion)   ]
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

    edge = nonlinear_temperature_response(out_door_air, test_typo)
    ref = low_boundary
    real_difference = abs(indoor_air_temp - ref)
    penality_cases_weight=0.5
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_reward(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E
            else:
                weight_T =penality_cases_weight# 1 - normalized_price  # min(normalized_price,0)
                weight_E = 1 - weight_T
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_penality(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            # penalty case
            weight_T = penality_cases_weight #1-normalized_price  # min(normalized_price,0)
            weight_E = 1 - weight_T
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))

        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            weight_T = 1  - normalized_price
            weight_E = 1 - weight_T
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            weight_T = penality_cases_weight # 1  # - normalized_price
            weight_E = 1 - weight_T
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)

        THERMAL_EDGE=0.0
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T =min( (1 - normalized_price),normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        # elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
        #     weight_E = min((1 - normalized_price), normalized_price)
        #     weight_T = 1 - weight_E
        #     scenario = 3.1
        #     thermal_reward = exponential_reward(real_difference)
        #     energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
        #     final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        #
        # elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
        #     weight_E = min((1 - normalized_price), normalized_price)
        #     weight_T = 1 - weight_E
        #     scenario = 3.2
        #     thermal_reward = exponential_reward(real_difference)
        #     energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))
        #     final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            weight_T = penality_cases_weight# (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.3
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.4
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge


def reward_function_w_flexibility_fixed (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

    if test_typo == 'peak_heat_day':
        if(interval==1):
            cushion = np.array(
                [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
                 0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
            prediction_horizon = 2
        elif(interval==2):
            cushion = np.array(
                            [0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.0888,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.05413,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444,
                            0.0444
                ]).reshape(-1, 1)
            prediction_horizon = 3
    elif(test_typo == 'peak_cool_day'):
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        prediction_horizon = 2
    c_=0
    max_price =float(max(cushion)  + c_)
    min_price = float(min(cushion) - c_)
    # real_lowest_price=float(min(cushion))
    if counter_x <= (len(cushion)-1) - prediction_horizon:
        predicted_price = cushion[counter_x + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_x + prediction_horizon) % len(cushion)   ]
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

    edge = nonlinear_temperature_response(out_door_air, test_typo)
    ref = low_boundary
    real_difference = abs(indoor_air_temp - ref)
    penality_cases_weight=0.5
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T =penality_cases_weight
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_reward(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E
            else:
                weight_T =penality_cases_weight# 1 - normalized_price  # min(normalized_price,0)
                weight_E = 1 - weight_T
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_penality(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            # penalty case
            weight_T = penality_cases_weight
            weight_E = 1 - weight_T
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))

        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            weight_T = penality_cases_weight
            weight_E = 1 - weight_T
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            weight_T = penality_cases_weight # 1  # - normalized_price
            weight_E = 1 - weight_T
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)

        THERMAL_EDGE=0.0
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T =penality_cases_weight
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


        else:
            weight_T = penality_cases_weight# (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.3
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.4
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge