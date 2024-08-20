import numpy as np
import math
import math

def exponential_penality(input, scale=2):
    a = -((2) / (1 + math.exp(-input/scale))) + 1
    return a

def exponential_reward (input, scale=2):
    a = -((2) / (1 + math.exp(-input/scale))) + 2
    return a

def optimized_outdoor_diff(x):
    beta =1
    delta=17.37
    return 1/(1+math.exp(-beta *(x-delta)))
counter_0=0
counter_1=0
counter_2=0
counter_3=0
energy_min=0
energy_max=4500

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
    if test_typo =='peak_heat_day':
        T_mid = 10  #for winter: 0, for summer:10
    else:
        T_mid = 10  # for winter: 0, for summer:10
    # s = 10
    s = 5
    scale=1.5
    return scale* np.tanh((T - T_mid) / s)

def normalized_thermal_function(T_in, T_ref, b_up,b_down):
    diff_=abs(T_in-T_ref)
    x_m=abs(b_up-b_down)
    x_normalized=diff_/x_m
    return(x_normalized)
def penality_normalized_thermal_function(T_in, T_ref, b_up,b_down):
    diff_=abs(T_in-T_ref)
    x_m=min( abs(T_in-b_up),  abs(T_in-b_down)   )
    x_normalized = - x_m/diff_
    return(x_normalized)
def reward_function_w_flexibility_(low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_1
    prediction_horizon =2
    if test_typo =='peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413, 0.05413,
             0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)

    else:
        cushion = np.array(
        [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
         0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)

    if (counter_1 <= 20):
        predicted_price = cushion[counter_1 + prediction_horizon]
    elif (counter_1== 21):
        predicted_price = cushion[0]
    elif (counter_1== 22):
        predicted_price = cushion[1]
    elif (counter_1 == 23):
        predicted_price = cushion[2]

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

    epsilon = 0.0001
    price = round(price, 4)
    energy_max = 5000
    energy_min = 0

    max_price = 0.0888 + 0.000
    min_price = 0.0444 - 0.000


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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        thermal_weight=normalized_price
        energy_weight = 1-thermal_weight
        T_normalized = normalized_thermal_function(indoor_air_temp, ref)

        if (low_boundary < indoor_air_temp < up_bundary) :
            # reward case
            scenario = 1
            thermal_reward = 1- T_normalized
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * energy_weight

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 1.3
            thermal_reward = - T_normalized  # penality_new(real_difference)

            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * (1 - energy_weight) + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5*(low_boundary+up_bundary)  + edge
        # real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping =  max( (ref + edge), (ref - edge) )
        low_b_load_shaping = min( (ref + edge), (ref - edge) )
        T_normalized=normalized_thermal_function(indoor_air_temp, ref)
        thermal_weight =1- normalized_price
        energy_weight = 1 - thermal_weight
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            # reward case
            scenario = 2
            if test_typo == 'peak_heat_day':
                energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                energy_reward = max(0, ((cool_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward = 1- T_normalized #reward_new(real_difference)
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = - T_normalized #penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = - T_normalized # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        T_normalized = normalized_thermal_function(indoor_air_temp, ref)
        thermal_weight = min( (1 - normalized_price),normalized_price)
        energy_weight = 1 - thermal_weight
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = 1- T_normalized #reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = - T_normalized #penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = - T_normalized #penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight,edge


def reward_function_w_flexibility_saving_ (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_0
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444,
             0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000

    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.001
        min_price = 0.0444 - 0.001

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

    final_reward, thermal_reward, energy_reward, ref, scenario, thermal_weight, energy_weight, edge = the_core_reward_mechanism(
        low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption, heat_consumption, test_typo,
        all_consumption, occupied_period, load_shaping_mode, normalized_price)
    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight, edge

def reward_function_w_flexibility_saving_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_2
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000

    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.001
        min_price = 0.0444 - 0.001

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

    final_reward, thermal_reward, energy_reward, ref, scenario, thermal_weight, energy_weight, edge=the_core_reward_mechanism (low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption, heat_consumption, test_typo, all_consumption, occupied_period, load_shaping_mode, normalized_price)
    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight, edge

def reward_function_w_flexibility_saving_3 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_3
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000

    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.000
        min_price = 0.0444 - 0.000

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        thermal_weight = normalized_price  # min(normalized_price,0)
        energy_weight = 1 - thermal_weight
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)

        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case
            thermal_reward = 1 - T_normalized
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * energy_weight

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 1.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
        thermal_weight = 1 - normalized_price
        energy_weight = 1 - thermal_weight
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = 1 - T_normalized  # reward_new(real_difference)
            energy_reward = 0
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        thermal_weight = min((1 - normalized_price), normalized_price)
        energy_weight = 1 - thermal_weight
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = 1 - T_normalized
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight, edge

def reward_function_w_flexibility_saving_4 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_3
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000

    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.002
        min_price = 0.0444 - 0.002

    if (counter_3 <= 21):
        predicted_price = cushion[counter_3 + prediction_horizon]
    elif (counter_3 == 22):
        predicted_price = cushion[0]
    elif (counter_3 == 23):
        predicted_price = cushion[1]


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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)

        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0#min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case

            thermal_reward = 1 - T_normalized
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
        weight_T = 1# - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = 1 - T_normalized  # reward_new(real_difference)
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        weight_T = min(normalized_price, (1 - normalized_price))
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = 1 - T_normalized
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge
def reward_function_w_flexibility_saving_4_30mins (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_0
    prediction_horizon = 3
    if test_typo == 'peak_heat_day':
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

    max_price = max(cushion) + 0.002
    min_price = min(cushion) - 0.002

    if counter_0 <= 47 - prediction_horizon:
        predicted_price = cushion[counter_0 + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_0 + prediction_horizon) % 48   ]

    counter_0= (counter_0 + 1) % 48

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        weight_T = normalized_price[0]  # min(normalized_price,0)
        weight_E = 1 - weight_T
        # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        real_difference=abs(indoor_air_temp-ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case

            thermal_reward = reward_new(real_difference) #1 - T_normalized
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward =  penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
        weight_T = 1# - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = reward_new(real_difference)
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):

        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        weight_T = min(normalized_price, (1 - normalized_price))[0]
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = reward_new(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward =  penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge


def reward_function_w_flexibility_saving_5_30mins (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_0
    prediction_horizon = 3
    if test_typo == 'peak_heat_day':
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

    max_price = float(max(cushion) + 0.002)
    min_price = float(min(cushion) - 0.002)

    if counter_0 <= 47 - prediction_horizon:
        predicted_price = cushion[counter_0 + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_0 + prediction_horizon) % 48   ]

    counter_0= (counter_0 + 1) % 48

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        real_difference=abs(indoor_air_temp-ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized #reward_new(real_difference) #
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1# - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):

        ref = 0.5 * (low_boundary + up_bundary) + edge
        # real_difference = abs(indoor_air_temp - ref)
        weight_T = min(normalized_price, (1 - normalized_price))
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_saving_6_30mins (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_1
    prediction_horizon = 3
    if test_typo == 'peak_heat_day':
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



    max_price = float(max(cushion) + 0.002)
    min_price = float(min(cushion) - 0.002)

    if counter_1 <= 47 - prediction_horizon:
        predicted_price = cushion[counter_1 + prediction_horizon]
    else:
        predicted_price = cushion[(counter_1 + prediction_horizon) % 48]

    counter_1 = (counter_1 + 1) % 48

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        real_difference = abs(indoor_air_temp - ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption == 0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0  # min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):

        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        weight_T = min(normalized_price, (1 - normalized_price))
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary,
                                                                  low_boundary)  # penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge


def reward_function_w_flexibility_30_min_interval_ (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,margin_price):
    #### global param
    prediction_horizon = 3
    if test_typo == 'peak_heat_day':
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
    max_price = float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
    # real_lowest_price=float(min(cushion))
    if counter_x <= 47 - prediction_horizon:
        predicted_price = cushion[counter_x + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_x + prediction_horizon) % 48   ]
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :  # night

        ref = low_boundary
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        real_difference=abs(indoor_air_temp-ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized #reward_new(real_difference) #
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1# - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        # real_difference = abs(indoor_air_temp - ref)
        weight_T =min(normalized_price, (1 - normalized_price))
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:

            # penalty case
            scenario = 3.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_30_min_interval_07_16 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,margin_price):
    #### global param
    prediction_horizon = 3
#     if test_typo == 'peak_heat_day':
#         cushion = np.array(
#             [0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.0888,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.05413,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444,
#             0.0444
# ]).reshape(-1, 1)
    if test_typo == 'peak_heat_day':
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413,0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
    max_price = 0.1#float(max(cushion) + margin_price)
    min_price = 0#float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward=exponential_reward(real_difference)
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
        else:
            # penalty case
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
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
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
        weight_T =(1 - normalized_price)
        weight_E = 1 - weight_T
        if (low_boundary) < indoor_air_temp < (up_bundary):
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            # weight_T = max((1 - normalized_price), normalized_price)
            # weight_E = 1 - weight_T
            if (low_boundary >= indoor_air_temp):
                scenario = 3.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_30_min_interval_07_17_test_1 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):
    #### global param

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
    max_price = 0.1#float(max(cushion) + margin_price)
    min_price = 0#float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward=exponential_reward(real_difference)
            final_reward = thermal_reward * weight_T + energy_reward * weight_E
        else:
            # penalty case
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
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
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
        weight_T =1 #(1 - normalized_price)
        weight_E = 1 - weight_T
        if (low_boundary) < indoor_air_temp < (up_bundary):
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_07_23 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

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
    max_price =float(max(cushion) + 0.0001)
    min_price = float(min(cushion) - 0.0001)
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
                weight_T = 1 - normalized_price  # min(normalized_price,0)
                weight_E = 1 - weight_T
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_penality(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            # penalty case
            weight_T = 1-normalized_price  # min(normalized_price,0)
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
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
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

        THERMAL_EDGE=0.1
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_E= (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_E = (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            weight_T = (1 - normalized_price)
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


def reward_function_w_flexibility_07_24 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

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
    max_price =float(max(cushion) + 0.0001)
    min_price = float(min(cushion) - 0.0001)
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
    penality_cases_weight=0.5
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)
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
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
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

        THERMAL_EDGE=0.2
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_E= (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_E = (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))
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


def reward_function_w_flexibility_07_25 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

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
    max_price =float(max(cushion)  + 0.0002)
    min_price = float(min(cushion) - 0.0002)
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
            weight_T = 1  # - normalized_price
            weight_E = 1 - weight_T
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
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

        THERMAL_EDGE=0.2
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T =min( (1 - normalized_price),normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_T = max((1 - normalized_price), normalized_price)
            weight_E = 1 - weight_T
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_T = max((1 - normalized_price), normalized_price)
            weight_E = 1 - weight_T
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))
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

def reward_function_w_flexibility_07_25_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

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
    max_price =float(max(cushion) + 0.0001)
    min_price = float(min(cushion) - 0.0001)
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
            weight_T = 1  # - normalized_price
            weight_E = 1 - weight_T
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
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

        THERMAL_EDGE=0.2
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T =min( (1 - normalized_price),normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_T = min((1 - normalized_price), normalized_price)
            weight_E = 1 - weight_T
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_T = min((1 - normalized_price), normalized_price)
            weight_E = 1 - weight_T
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))
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

def reward_function_w_flexibility_07_23_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):

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
    max_price =float(max(cushion) + 0.0001)
    min_price = float(min(cushion) - 0.0001)
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
                weight_T = 1 - normalized_price  # min(normalized_price,0)
                weight_E = 1 - weight_T
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_penality(real_difference)
                final_reward = thermal_reward * weight_T + energy_reward * weight_E

        else:
            # penalty case
            weight_T = 1-normalized_price  # min(normalized_price,0)
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
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
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

        THERMAL_EDGE=0.1
        if (low_boundary+THERMAL_EDGE) < indoor_air_temp < (up_bundary-THERMAL_EDGE):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_E= (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1-((heat_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_E = (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  1-((cool_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            weight_T = (1 - normalized_price)
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


def reward_function_w_flexibility_30_min_interval_07_17_test_3 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):
    #### global param

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
    margin_price=0.002
    max_price =float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward=exponential_reward(real_difference)

            main_factor=energy_reward
            sec_factor=thermal_reward
            weight_for_sec_factor=normalized_price
            final_reward=main_factor*(1 + weight_for_sec_factor *sec_factor)

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = energy_reward
            sec_factor = thermal_reward
            weight_for_sec_factor = normalized_price
            final_reward=main_factor*(1 - weight_for_sec_factor *sec_factor)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0,((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor*(1 + weight_for_sec_factor * sec_factor)

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor*(1 - weight_for_sec_factor * sec_factor)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary) < indoor_air_temp < (up_bundary):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor*(1 + weight_for_sec_factor * sec_factor)


        else:
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor*(1 - weight_for_sec_factor * sec_factor)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_30_min_interval_07_17_test_4 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):
    #### global param

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
    margin_price=0.002
    max_price =float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_reward(real_difference)
            else:
                scenario = 1.1
                thermal_reward = exponential_penality(real_difference)
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))


            main_factor=energy_reward
            sec_factor=thermal_reward
            weight_for_sec_factor=normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = energy_reward
            sec_factor = thermal_reward
            weight_for_sec_factor = normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0,((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)

        if (low_boundary) < indoor_air_temp < (up_bundary):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor


        else:
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge

def reward_function_w_flexibility_30_min_interval_07_19_test_1 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):
    #### global param

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
    margin_price=0.002
    max_price =float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            thermal_reward=exponential_reward(real_difference)

            main_factor=energy_reward
            sec_factor=thermal_reward
            weight_for_sec_factor=normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = energy_reward
            sec_factor = thermal_reward
            weight_for_sec_factor = normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0,((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        THERMAL_EDGE = 0.10
        if (low_boundary + THERMAL_EDGE) < indoor_air_temp < (up_bundary - THERMAL_EDGE):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        elif ((low_boundary) < indoor_air_temp <= (low_boundary + THERMAL_EDGE)) or (((up_bundary - THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = 1-normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        else:
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge


def reward_function_w_flexibility_30_min_interval_07_19_test_2 (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption, counter_x,interval):
    #### global param

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
    margin_price=0.002
    max_price =float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_reward(real_difference)
            else:
                scenario = 1.1
                energy_reward = min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = exponential_penality(real_difference)


            main_factor=energy_reward
            sec_factor=thermal_reward
            weight_for_sec_factor=normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 1.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))

            else:
                # penalty case
                scenario = 1.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = energy_reward
            sec_factor = thermal_reward
            weight_for_sec_factor = normalized_price
            final_reward=main_factor + weight_for_sec_factor *sec_factor

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1  # - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = exponential_reward(real_difference)
            # T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            # thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = max(0,((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        else:
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_b_load_shaping >= indoor_air_temp):
                scenario = 2.1
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 2.2
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        THERMAL_EDGE = 0.1
        if (low_boundary + THERMAL_EDGE) < indoor_air_temp < (up_bundary - THERMAL_EDGE):
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            scenario = 3
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor
        elif ((low_boundary) < indoor_air_temp <= (low_boundary+THERMAL_EDGE)) :
            weight_E= (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.1
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        elif (((up_bundary-THERMAL_EDGE) <= indoor_air_temp < (up_bundary))):
            weight_E = (1 - normalized_price)
            weight_T = 1 - weight_E
            scenario = 3.2
            thermal_reward = exponential_reward(real_difference)
            energy_reward = max(0,  ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor

        else:
            weight_T = (1 - normalized_price)
            weight_E = 1 - weight_T
            # penalty case
            thermal_reward = exponential_penality(real_difference)
            if (low_boundary >= indoor_air_temp):
                scenario = 3.2
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            else:
                # penalty case
                scenario = 3.3
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))

            main_factor = thermal_reward
            sec_factor = energy_reward
            weight_for_sec_factor = normalized_price
            final_reward = main_factor + weight_for_sec_factor * sec_factor


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge



counter_y=0
def reward_function_w_flexibility_saving_30min_interval_redesigned_ppo (low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption,occupancy):
    #### global param
    global counter_y
    prediction_horizon = 3
    if test_typo == 'peak_heat_day':
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
    margin_price=0.002
    max_price = float(max(cushion) + margin_price)
    min_price = float(min(cushion) - margin_price)
    # real_lowest_price=float(min(cushion))
    if counter_y <= 47 - prediction_horizon:
        predicted_price = cushion[counter_y + prediction_horizon]
    else:
        predicted_price = cushion[ (counter_y + prediction_horizon) % 48   ]

    counter_y= (counter_y + 1) % 48

    predicted_price = predicted_price[0]
    if predicted_price - price > 0.0001:
        load_shaping_mode = 1
    else:
        load_shaping_mode = 0
    if occupancy==0:
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
    if (occupied_period == 0) and (load_shaping_mode == 0) :  # night

        ref = low_boundary
        weight_T = normalized_price  # min(normalized_price,0)
        weight_E = 1 - weight_T
        real_difference=abs(indoor_air_temp-ref)
        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption==0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0 #min(0, - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized #reward_new(real_difference) #
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * weight_E

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:
            # penalty case
            scenario = 1.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        weight_T = 1 - normalized_price
        weight_E = 1 - weight_T
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #
            energy_reward = 0
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        # real_difference = abs(indoor_air_temp - ref)
        weight_T = min(normalized_price, (1 - normalized_price))
        # weight_T = (1 - normalized_price)
        weight_E = 1 - weight_T
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_reward = 1 - T_normalized  # reward_new(real_difference) #

            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward =penality_normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)# penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * weight_T + energy_reward * (weight_E)


    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, weight_T, weight_E, edge
def the_core_reward_mechanism (low_boundary, up_bundary, indoor_air_temp, out_door_air, cool_consumption, heat_consumption, test_typo, all_consumption, occupied_period, load_shaping_mode, normalized_price):

        edge = nonlinear_temperature_response(out_door_air, test_typo)
        if (occupied_period == 0) and (load_shaping_mode == 0):  # night

            ref = low_boundary
            thermal_weight = 1-normalized_price  # min(normalized_price,0)
            energy_weight = 1 - thermal_weight
            T_normalized = normalized_thermal_function(indoor_air_temp, ref,up_bundary, low_boundary)

            if (low_boundary < indoor_air_temp < up_bundary):
                # reward case
                scenario = 1
                thermal_reward = 1 - T_normalized
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                final_reward = thermal_reward * thermal_weight + energy_reward * energy_weight

            elif ((low_boundary) >= indoor_air_temp):
                # penalty case
                scenario = 1.2
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                ### the more heating we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

            else:
                # penalty case
                scenario = 1.3
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                ### the more cooling we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
            ref = 0.5 * (low_boundary + up_bundary) + edge
            up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
            low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
            T_normalized = normalized_thermal_function(indoor_air_temp, ref,up_b_load_shaping, low_b_load_shaping)
            thermal_weight = 1# - normalized_price
            energy_weight = 1 - thermal_weight
            if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
                scenario = 2
                # if test_typo == 'peak_heat_day':
                #     energy_reward = max(0, ((heat_consumption - energy_min) / (energy_max - energy_min)))
                # else:
                #     energy_reward = max(0, ((cool_consumption - energy_min) / (energy_max - energy_min)))
                thermal_reward = 1 - T_normalized  # reward_new(real_difference)
                energy_reward=0
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

            elif (indoor_air_temp < low_b_load_shaping):
                scenario = 2.1
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                ### the more heating we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
            else:  # indoor_air_temp > up_bundary
                # penalty case
                scenario = 2.2
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                ### the more cooling we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif (occupied_period == 1):
            ref = 0.5 * (low_boundary + up_bundary) + edge
            T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
            thermal_weight = min((1 - normalized_price), normalized_price)
            energy_weight = 1 - thermal_weight
            if (low_boundary) <= indoor_air_temp <= (up_bundary):
                scenario = 3
                thermal_reward = 1 - T_normalized
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
            elif (low_boundary >= indoor_air_temp):
                # penalty case
                scenario = 3.2
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                ### the more heating we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

            else:
                # penalty case
                scenario = 3.3
                thermal_reward = - T_normalized  # penality_new(real_difference)
                energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                ### the more cooling we use, the less energy penalty ###
                final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        return final_reward, thermal_reward, energy_reward, ref, scenario, thermal_weight, energy_weight, edge


def reward_function_w_flexibility_v_1_2(low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_2
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444,
             0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000
    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.000
        min_price = 0.0444 - 0.000

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night
        ref = low_boundary
        real_difference = abs(indoor_air_temp - ref)
        thermal_weight = normalized_price  # min(normalized_price,0)
        energy_weight = 1 - thermal_weight
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)

        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption == 0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0  # max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case

            thermal_reward =reward_new(real_difference) # 1 - T_normalized
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * energy_weight

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 1.3
            thermal_reward =  penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary)+ edge
        real_difference = abs(indoor_air_temp - ref)
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
        thermal_weight = 1 - normalized_price
        energy_weight = 1 - thermal_weight
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward =  reward_new(real_difference)
            energy_reward = 0
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward =  penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary) + edge
        real_difference = abs(indoor_air_temp - ref)
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        thermal_weight = min((1 - normalized_price), normalized_price)
        energy_weight = 1 - thermal_weight
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = reward_new(real_difference) #1 - T_normalized
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward =  penality_new(real_difference) #- T_normalized  #
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight, edge


def reward_function_w_flexibility_v_1_1(low_boundary, up_bundary, indoor_air_temp, out_door_air, price, cool_consumption,heat_consumption, test_typo, all_consumption):
    #### global param
    global counter_1
    prediction_horizon = 2
    if test_typo == 'peak_heat_day':

        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.05413, 0.05413, 0.05413, 0.05413,
             0.05413, 0.0888, 0.0888, 0.0888, 0.0888, 0.05413, 0.05413, 0.05413, 0.0444, 0.0444, 0.0444,
             0.0444]).reshape(-1, 1)
        max_price = 0.0888 + 0.000
        min_price = 0.0444 - 0.000

    else:
        cushion = np.array(
            [0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0444, 0.0842, 0.0842, 0.0842, 0.0842, 0.0842,
             0.13814, 0.13814, 0.13814, 0.13814, 0.0842, 0.0842, 0.0842, 0.0444, 0.0444, 0.0444, 0.0444]).reshape(-1, 1)
        max_price = 0.0842 + 0.000
        min_price = 0.0444 - 0.000

    if (counter_1 <= 20):
        predicted_price = cushion[counter_1 + prediction_horizon]
    elif (counter_1 == 21):
        predicted_price = cushion[0]
    elif (counter_1 == 22):
        predicted_price = cushion[1]
    elif (counter_1 == 23):
        predicted_price = cushion[2]

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
    if (occupied_period == 0) and (load_shaping_mode == 0):  # night

        ref = low_boundary
        thermal_weight = normalized_price  # min(normalized_price,0)
        energy_weight = 1 - thermal_weight
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)

        if (low_boundary < indoor_air_temp < up_bundary):
            if all_consumption == 0:
                scenario = 1
                energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            else:
                scenario = 1.1
                energy_reward = 0  # max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            # reward case

            thermal_reward = 1 - T_normalized
            # energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * energy_weight

        elif ((low_boundary) >= indoor_air_temp):
            # penalty case
            scenario = 1.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 1.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif ((occupied_period == 0) and (load_shaping_mode == 1)):  # day
        ref = 0.5 * (low_boundary + up_bundary)# + edge
        up_b_load_shaping = 24  # max((ref + edge), (ref - edge))
        low_b_load_shaping = 21  # min((ref + edge), (ref - edge))
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_b_load_shaping, low_b_load_shaping)
        thermal_weight = 1 - normalized_price
        energy_weight = 1 - thermal_weight
        if (low_b_load_shaping <= indoor_air_temp < up_b_load_shaping):
            scenario = 2
            thermal_reward = 1 - T_normalized  # reward_new(real_difference)
            energy_reward = 0
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        elif (indoor_air_temp < low_b_load_shaping):
            scenario = 2.1
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        else:  # indoor_air_temp > up_bundary
            # penalty case
            scenario = 2.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    elif (occupied_period == 1):
        ref = 0.5 * (low_boundary + up_bundary)# + edge
        T_normalized = normalized_thermal_function(indoor_air_temp, ref, up_bundary, low_boundary)
        thermal_weight = min((1 - normalized_price), normalized_price)
        energy_weight = 1 - thermal_weight
        if (low_boundary) <= indoor_air_temp <= (up_bundary):
            scenario = 3
            thermal_reward = 1 - T_normalized
            energy_reward = max(0, 1 - ((all_consumption - energy_min) / (energy_max - energy_min)))
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)
        elif (low_boundary >= indoor_air_temp):
            # penalty case
            scenario = 3.2
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
            ### the more heating we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

        else:
            # penalty case
            scenario = 3.3
            thermal_reward = - T_normalized  # penality_new(real_difference)
            energy_reward = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
            ### the more cooling we use, the less energy penalty ###
            final_reward = thermal_reward * thermal_weight + energy_reward * (energy_weight)

    return final_reward, thermal_reward, energy_reward, ref, scenario, predicted_price, thermal_weight, energy_weight, edge


def the_one_under_test_outdoor (low_boundary, up_bundary, indoor_air_temp, cool_consumption, heat_consumption, all_consumption, out_door_air):
    current_consumption_normalized = (all_consumption - energy_min) / (energy_max - energy_min)

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
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_new(diff_curr)
                r_e = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_new(diff_curr)
                r_e = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4

        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_new(diff_curr)
                r_e =  -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = penality_new(diff_curr)
                r_e =-max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_


def the_one_under_test_outdoor_v1_1 (low_boundary, up_bundary, indoor_air_temp, cool_consumption, heat_consumption, all_consumption, out_door_air):
    current_consumption_normalized = (all_consumption - energy_min) / (energy_max - energy_min)

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
    T_normalized = normalized_thermal_function(indoor_air_temp, target, up_bundary, low_boundary)

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
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = 1- T_normalized #reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = 1- T_normalized #reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = - T_normalized # penality_new(diff_curr)
                r_e = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = - T_normalized # penality_new(diff_curr)
                r_e = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = - T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4

        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = - T_normalized # penality_new(diff_curr)
                r_e =  -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = - T_normalized # penality_new(diff_curr)
                r_e =-max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = - T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def the_one_under_test_outdoor_v1_2 (low_boundary, up_bundary, indoor_air_temp, cool_consumption, heat_consumption, all_consumption, out_door_air):
    current_consumption_normalized = (all_consumption - energy_min) / (energy_max - energy_min)
    edge = nonlinear_temperature_response(out_door_air, 'peak_heat_day')
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =  low_boundary +0  #((up_bundary + low_boundary) / 2) - 1 #real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2) +edge

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)
    T_normalized = normalized_thermal_function(indoor_air_temp, target, up_bundary, low_boundary)

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
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = reward_new(diff_curr)#1- T_normalized #reward_new(diff_curr)#
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t =reward_new(diff_curr)# 1- T_normalized #reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = penality_new(diff_curr)#- T_normalized # penality_new(diff_curr)#
                r_e = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = penality_new(diff_curr)#- T_normalized # penality_new(diff_curr)
                r_e = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = penality_new(diff_curr)#- T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4

        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = penality_new(diff_curr)#- T_normalized # penality_new(diff_curr)
                r_e =  -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t =penality_new(diff_curr)# - T_normalized # penality_new(diff_curr)
                r_e =-max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = penality_new(diff_curr)#- T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_

def the_one_under_test_outdoor_v1_3 (low_boundary, up_bundary, indoor_air_temp, cool_consumption, heat_consumption, all_consumption, out_door_air):
    current_consumption_normalized = (all_consumption - energy_min) / (energy_max - energy_min)
    edge = nonlinear_temperature_response(out_door_air, 'peak_heat_day')
    real_bound_up = 24
    real_bound_down = 21

    if (up_bundary > real_bound_up) and (low_boundary < real_bound_down):
        wild_boundary = 1
        target =  low_boundary +0  #((up_bundary + low_boundary) / 2) - 1 #real_bound_down  #
    else:
        wild_boundary = 0
        target = ((up_bundary + low_boundary) / 2) +edge

    outdoor_diff = abs(target - out_door_air)
    diff_curr = abs(indoor_air_temp - target)
    T_normalized = normalized_thermal_function(indoor_air_temp, target, up_bundary, low_boundary)

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
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21-24]
            r_t = 1- T_normalized #reward_new(diff_curr)#
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 0

        else:  # [15-30]
            r_t = 1- T_normalized #reward_new(diff_curr)
            r_e = 1 - current_consumption_normalized

            energy_weight = 1 - thermal_weight
            R_ = thermal_weight * r_t + (energy_weight) * r_e
            s_ = 1
    else:
        thermal_weight = optimized_outdoor_diff(outdoor_diff)
        if (wild_boundary == 0):  # [21, 24]
            if (low_boundary > indoor_air_temp):  # too low
                r_t = - T_normalized # penality_new(diff_curr)#
                r_e = -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 2
            elif (up_bundary < indoor_air_temp):  # too high
                r_t = - T_normalized # penality_new(diff_curr)
                r_e = -max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 3
            else:
                r_t = - T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 4

        else:  # [14, 30]
            if (real_bound_down >= indoor_air_temp):  # too low
                r_t = - T_normalized # penality_new(diff_curr)
                r_e =  -max(0, 1 - ((heat_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 5
            elif (real_bound_up < indoor_air_temp):  # too high
                r_t = - T_normalized # penality_new(diff_curr)
                r_e =-max(0, 1 - ((cool_consumption - energy_min) / (energy_max - energy_min)))
                energy_weight = 1 - thermal_weight
                R_ = thermal_weight * r_t + (energy_weight) * r_e
                s_ = 6
            else:
                r_t = - T_normalized # penality_new(diff_curr)
                R_ = r_t
                r_e = 0
                s_ = 7
    return R_, r_t, r_e, current_indoor_state, thermal_weight, target, s_