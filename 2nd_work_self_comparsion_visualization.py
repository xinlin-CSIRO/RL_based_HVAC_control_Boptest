import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


debug_model=1
occup=1
cool_or_heat='heat'#or 'heat'
file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\Second_work\BOPTEST_best_air_flexibity.xlsx"
data_source=pd.ExcelFile(file)
if(cool_or_heat=='heat'):
    #peak heat scenario
    #our method

    our_method = pd.read_excel(data_source, 'Peak_heat_our_test')
    indoor_our_approach=np.array(our_method['indoor_temperature']).astype(float)
    all_conumsption_our_approach=np.array(our_method['all_consumption']).astype(float)
    thermal_kpi_our_list=np.array(our_method['themal_kpi']).astype(float)
    thermal_kpi_our_approach=float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
    energy_kpi_our_list=np.array(our_method['energy_kpi']).astype(float)
    energy_kpi_our_approach=float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])
    cost_kpi_our_list=np.array(our_method['cost_kpi']).astype(float)
    cost_kpi_our_approach=float(cost_kpi_our_list[-1]) - float(cost_kpi_our_list[0])

    outdoor_temp=np.array(our_method['outdoor_air']).astype(float)

    #rule_based
    rule_based = pd.read_excel(data_source, 'Peak_heat_rule')
    indoor_benchmark_rule=np.array(rule_based['indoor_temp'])
    all_conumsption_benchmark_rule=np.array(rule_based['all_consumption'])
    thermal_kpi_benchmark_rule=np.array(rule_based['themal_kpi'])
    price=np.array(rule_based['price'])
    up_boundary=np.array(rule_based['boundary_1'])
    low_boundary=np.array(rule_based['boundary_0'])

    thermal_kpi_rule_list=np.array(rule_based['themal_kpi']).astype(float)
    thermal_kpi_rule=float(thermal_kpi_rule_list[-1]) - float(thermal_kpi_rule_list[0])
    energy_kpi_rule_list=np.array(rule_based['energy_kpi']).astype(float)
    energy_kpi_rule=float(energy_kpi_rule_list[-1]) - float(energy_kpi_rule_list[0])
    cost_kpi_rule_list=np.array(rule_based['cost_kpi']).astype(float)
    cost_kpi_rule=float(cost_kpi_rule_list[-1]) - float(cost_kpi_rule_list[0])


    #benchmark_rl
    benchmark_rl = pd.read_excel(data_source, 'Peak_heat_our_test_DDPG')
    indoor_benchmark_rl=np.array(benchmark_rl['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl=np.array(benchmark_rl['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list=np.array(benchmark_rl['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach=float(thermal_kpi_benchmark_rl_list[-1]) - float(thermal_kpi_benchmark_rl_list[0])
    energy_kpi_benchmark_rl_list=np.array(benchmark_rl['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach=float(energy_kpi_benchmark_rl_list[-1]) - float(energy_kpi_benchmark_rl_list[0])
    cost_kpi_benchmark_rl_list=np.array(benchmark_rl['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach=float(cost_kpi_benchmark_rl_list[-1]) - float(cost_kpi_benchmark_rl_list[0])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle('Test scenario: Peak heat days',y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_our_approach, label='our method: SAC', color='cyan', linewidth=4)
    ax1.plot(indoor_benchmark_rl, label='our method: DDPG', color='green', linewidth=4)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1: rule-based', color='blue')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

    # Second row spanning all columns
    ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    ax2.set_title('Outdoor temperature')
    ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc="lower right")

    specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax2.set_xticks(index_dates)
    ax2.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['cyan', 'green','blue' ]
    labels = ['our method: SAC', 'our method: DDPG','Benchmark 1: rule-based' ]
    datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
    benchmarks = [[thermal_kpi_benchmark_rl_approach,thermal_kpi_rule ],
                  [energy_kpi_benchmark_rl_approach, energy_kpi_rule ],
                  [cost_kpi_benchmark_rl_approach, cost_kpi_rule ]]

    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_title(titles[i])
        all_values = [datasets[i]] + benchmarks[i]
        rects = ax.bar(labels, all_values, color=colors, linewidth=2)
        # ax.legend(loc="lower right")
        ax.set_ylabel(titles[i])

        # Adding value annotations
        for rect, value in zip(rects, all_values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)

        # Adjust y-limits to accommodate text annotations
        max_height = max(all_values)
        ax.set_ylim(0, max_height + 0.1 * max_height)

    plt.show()


else:
    ###############################################################################################################################################
    ##########peak cool################################################################
    our_method = pd.read_excel(data_source, 'Peak_cool_our_test')
    indoor_our_approach = np.array(our_method['indoor_temperature']).astype(float)
    all_conumsption_our_approach = np.array(our_method['all_consumption']).astype(float)
    thermal_kpi_our_list = np.array(our_method['themal_kpi']).astype(float)
    thermal_kpi_our_approach = float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
    energy_kpi_our_list = np.array(our_method['energy_kpi']).astype(float)
    energy_kpi_our_approach = float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])
    cost_kpi_our_list = np.array(our_method['cost_kpi']).astype(float)
    cost_kpi_our_approach = float(cost_kpi_our_list[-1]) - float(cost_kpi_our_list[0])

    outdoor_temp = np.array(our_method['outdoor_air']).astype(float)

    # rule_based
    rule_based = pd.read_excel(data_source, 'Peak_cool_rule')
    indoor_benchmark_rule = np.array(rule_based['indoor_temp'])
    all_conumsption_benchmark_rule = np.array(rule_based['all_consumption'])
    thermal_kpi_benchmark_rule = np.array(rule_based['themal_kpi'])
    price = np.array(rule_based['price'])
    up_boundary = np.array(rule_based['boundary_1'])
    low_boundary = np.array(rule_based['boundary_0'])

    thermal_kpi_rule_list = np.array(rule_based['themal_kpi']).astype(float)
    thermal_kpi_rule = float(thermal_kpi_rule_list[-1]) - float(thermal_kpi_rule_list[0])
    energy_kpi_rule_list = np.array(rule_based['energy_kpi']).astype(float)
    energy_kpi_rule = float(energy_kpi_rule_list[-1]) - float(energy_kpi_rule_list[0])
    cost_kpi_rule_list = np.array(rule_based['cost_kpi']).astype(float)
    cost_kpi_rule = float(cost_kpi_rule_list[-1]) - float(cost_kpi_rule_list[0])

    # benchmark_rl
    benchmark_rl = pd.read_excel(data_source, 'Peak_cool_our_test_DDPG')
    indoor_benchmark_rl = np.array(benchmark_rl['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl = np.array(benchmark_rl['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list = np.array(benchmark_rl['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach = float(thermal_kpi_benchmark_rl_list[-1]) - float(
        thermal_kpi_benchmark_rl_list[0])
    energy_kpi_benchmark_rl_list = np.array(benchmark_rl['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach = float(energy_kpi_benchmark_rl_list[-1]) - float(energy_kpi_benchmark_rl_list[0])
    cost_kpi_benchmark_rl_list = np.array(benchmark_rl['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach = float(cost_kpi_benchmark_rl_list[-1]) - float(cost_kpi_benchmark_rl_list[0])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle('Test scenario: Peak cool days', y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_our_approach, label='our method: SAC', color='cyan', linewidth=4)
    ax1.plot(indoor_benchmark_rl, label='our method: DDPG', color='green', linewidth=4)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1: rule-based', color='blue')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

    # Second row spanning all columns
    ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    ax2.set_title('Outdoor temperature')
    ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc="lower right")

    specific_dates = ['Day 282', 'Day 284', 'Day 286', 'Day 288', 'Day 290', 'Day 292', 'Day 294', 'Day 296']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax2.set_xticks(index_dates)
    ax2.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['cyan', 'green', 'blue']
    labels = ['our method: SAC', 'our method: DDPG', 'Benchmark 1: rule-based']
    datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
    benchmarks = [[thermal_kpi_benchmark_rl_approach, thermal_kpi_rule],
                  [energy_kpi_benchmark_rl_approach, energy_kpi_rule],
                  [cost_kpi_benchmark_rl_approach, cost_kpi_rule]]

    for i in range(3):
        ax = fig.add_subplot(gs[2, i])
        ax.set_title(titles[i])
        all_values = [datasets[i]] + benchmarks[i]
        rects = ax.bar(labels, all_values, color=colors, linewidth=2)
        # ax.legend(loc="lower right")
        ax.set_ylabel(titles[i])

        # Adding value annotations
        for rect, value in zip(rects, all_values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.2f}',
                    ha='center', va='bottom', fontsize=9)

        # Adjust y-limits to accommodate text annotations
        max_height = max(all_values)
        ax.set_ylim(0, max_height + 0.1 * max_height)

    plt.show()