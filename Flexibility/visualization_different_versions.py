import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


debug_model=1
occup=1
cool_or_heat='heat'#or 'heat'
file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\Second_work\BOPTEST_best_air_different_versions_comparsion_code.xlsx"
data_source=pd.ExcelFile(file)
scenario = 'heat'
if scenario == 'heat':
    our_method_v1 = pd.read_excel(data_source, 'Peak_heat_v1.1')
    length=len(our_method_v1)
    indoor_v1=np.array(our_method_v1['indoor_temperature'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_our_list=np.array(our_method_v1['themal_kpi'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_v1=float(thermal_kpi_our_list[-1])
    energy_kpi_our_list=np.array(our_method_v1['energy_kpi'])[(length-1-24*14):(length-1)].astype(float)
    energy_kpi_v1=float(energy_kpi_our_list[-1])
    cost_kpi_our_list=np.array(our_method_v1['cost_kpi'])[(length-1-24*14):(length-1)].astype(float)
    cost_kpi_v1=float(cost_kpi_our_list[-1])

    our_method_v2 = pd.read_excel(data_source, 'Peak_heat_v1.2')
    length=len(our_method_v2)
    indoor_v2=np.array(our_method_v2['indoor_temperature'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_our_list2=np.array(our_method_v2['themal_kpi'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_v2=float(thermal_kpi_our_list2[-1])
    energy_kpi_our_list2=np.array(our_method_v2['energy_kpi'])[(length-1-24*14):(length-1)].astype(float)
    energy_kpi_v2=float(energy_kpi_our_list2[-1])
    cost_kpi_our_list2=np.array(our_method_v2['cost_kpi'])[(length-1-24*14):(length-1)].astype(float)
    cost_kpi_v2=float(cost_kpi_our_list2[-1])

    our_method_v3 = pd.read_excel(data_source, 'Peak_heat_v1.4')
    length=len(our_method_v3)
    indoor_v3=np.array(our_method_v3['indoor_temperature'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_our_list3=np.array(our_method_v3['themal_kpi'])[(length-1-24*14):(length-1)].astype(float)
    thermal_kpi_v3=float(thermal_kpi_our_list3[-1])
    energy_kpi_our_list3=np.array(our_method_v3['energy_kpi'])[(length-1-24*14):(length-1)].astype(float)
    energy_kpi_v3=float(energy_kpi_our_list3[-1])
    cost_kpi_our_list3=np.array(our_method_v3['cost_kpi'])[(length-1-24*14):(length-1)].astype(float)
    cost_kpi_v3=float(cost_kpi_our_list3[-1])

    our_method_v0 = pd.read_excel(data_source, 'Peak_heat_v1.0')
    indoor_v0=np.array(our_method_v0['indoor_temperature']).astype(float)
    thermal_kpi_vo=np.array(our_method_v0['themal_kpi']).astype(float)
    thermal_kpi_v0=float(thermal_kpi_vo[-1]) - float(thermal_kpi_vo[0])
    energy_kpi_our_list0=np.array(our_method_v0['energy_kpi']).astype(float)
    energy_kpi_v0=float(energy_kpi_our_list0[-1]) - float(energy_kpi_our_list0[0])
    cost_kpi_our_list0=np.array(our_method_v0['cost_kpi']).astype(float)
    cost_kpi_v0=float(cost_kpi_our_list0[-1]) - float(cost_kpi_our_list0[0])




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



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle('Test scenario: Peak heat days',y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_v0, label='version-1.0', color='orange', linewidth=3)
    ax1.plot(indoor_v1, label='version-1.1', color='cyan', linewidth=3)
    ax1.plot(indoor_v2, label='version-1.2', color='green', linewidth=3)
    ax1.plot(indoor_v3, label='version-1.3', color='blue', linewidth=3)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1: rule-based', color='black')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')

    specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax1.set_xticks(index_dates)
    ax1.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['orange','cyan', 'green','blue','black']
    labels = ['version-1', 'version-1.1', 'version-1.2','version-1.3','rule-based' ]
    datasets = [thermal_kpi_v0, energy_kpi_v0, cost_kpi_v0]
    benchmarks = [[thermal_kpi_v1,thermal_kpi_v2,thermal_kpi_v3,thermal_kpi_rule ],
              [energy_kpi_v1, energy_kpi_v2, energy_kpi_v3, energy_kpi_rule ],
              [cost_kpi_v1, cost_kpi_v2, cost_kpi_v3, cost_kpi_rule ]]

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(titles[i])
        all_values = [datasets[i]] + benchmarks[i]
        rects = ax.bar(labels, all_values, color=colors, linewidth=2)
        # ax.legend(loc="lower right")
        ax.set_ylabel(titles[i])

        # Adding value annotations
        for rect, value in zip(rects, all_values):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height, f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9)

        # Adjust y-limits to accommodate text annotations
        max_height = max(all_values)
        ax.set_ylim(0, max_height + 0.1 * max_height)

    plt.show()

else:

    our_method_v3 = pd.read_excel(data_source, 'Peak_cool_v1.3')
    length = len(our_method_v3)
    indoor_v3 = np.array(our_method_v3['indoor_temperature'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    thermal_kpi_our_list3 = np.array(our_method_v3['themal_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    thermal_kpi_v3 = float(thermal_kpi_our_list3[-1])
    energy_kpi_our_list3 = np.array(our_method_v3['energy_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    energy_kpi_v3 = float(energy_kpi_our_list3[-1])
    cost_kpi_our_list3 = np.array(our_method_v3['cost_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    cost_kpi_v3 = float(cost_kpi_our_list3[-1])

    our_method_v0 = pd.read_excel(data_source, 'Peak_cool_v1.0')
    length = len(our_method_v0)
    indoor_v0 = np.array(our_method_v0['indoor_temperature'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    thermal_kpi_our_list0 = np.array(our_method_v0['themal_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    thermal_kpi_v0 = float(thermal_kpi_our_list0[-1])-float(thermal_kpi_our_list0[0])
    energy_kpi_our_list0 = np.array(our_method_v0['energy_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    energy_kpi_v0 = float(energy_kpi_our_list0[-1])- float(energy_kpi_our_list0[0])
    cost_kpi_our_list0 = np.array(our_method_v0['cost_kpi'])[(length - 1 - 24 * 14):(length - 1)].astype(float)
    cost_kpi_v0 = float(cost_kpi_our_list0[-1]) -float(cost_kpi_our_list0[0])

    # rule_based
    rule_based = pd.read_excel(data_source, 'Peak_cool_rule')
    indoor_benchmark_rule = np.array(rule_based['indoor_temp'])
    thermal_kpi_benchmark_rule = np.array(rule_based['themal_kpi'])
    up_boundary = np.array(rule_based['boundary_1'])
    low_boundary = np.array(rule_based['boundary_0'])

    thermal_kpi_rule_list = np.array(rule_based['themal_kpi']).astype(float)
    thermal_kpi_rule = float(thermal_kpi_rule_list[-1]) - float(thermal_kpi_rule_list[0])
    energy_kpi_rule_list = np.array(rule_based['energy_kpi']).astype(float)
    energy_kpi_rule = float(energy_kpi_rule_list[-1]) - float(energy_kpi_rule_list[0])
    cost_kpi_rule_list = np.array(rule_based['cost_kpi']).astype(float)
    cost_kpi_rule = float(cost_kpi_rule_list[-1]) - float(cost_kpi_rule_list[0])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    fig.suptitle('Test scenario: Peak cool days', y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_v0, label='version-1.0', color='orange', linewidth=3)
    # ax1.plot(indoor_v1, label='version-1.1', color='cyan', linewidth=3)
    # ax1.plot(indoor_v2, label='version-1.2', color='green', linewidth=3)
    ax1.plot(indoor_v3, label='version-1.3', color='blue', linewidth=3)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1: rule-based', color='black')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')

    specific_dates = ['Day 282', 'Day 284', 'Day 286', 'Day 288', 'Day 290', 'Day 292', 'Day 294', 'Day 296']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax1.set_xticks(index_dates)
    ax1.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['orange', 'blue', 'black']
    labels = ['version-1', 'version-1.3', 'rule-based']
    datasets = [thermal_kpi_v0, energy_kpi_v0, cost_kpi_v0]
    benchmarks = [[ thermal_kpi_v3, thermal_kpi_rule],
                  [energy_kpi_v3, energy_kpi_rule],
                  [ cost_kpi_v3, cost_kpi_rule]]

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
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
