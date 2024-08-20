import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#
# max_price = 0.0888 + 0.002  # 0.1    #  max=0.0888
# min_price = 0.0444 - 0.002  # 0.0332 #  min=0.0444
#
# normalized_price =  1* (0.0444 - min_price) / (max_price - min_price) #0.00446
# print(normalized_price)
#
#
# normalized_price =  1* (0.05413 -min_price) / (max_price - min_price) #0.2216
# print(normalized_price)
#
#
# normalized_price =  1* (0.0888 -min_price) / (max_price - min_price) #0.9955
# print(normalized_price)



head=2786-2
end=3120

debug_model=1
occup=1
cool_or_heat='cool'#or 'heat'
file=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\Second_work\BOPTEST_best_air_flexibity_all.xlsx"
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


    #benchmark_rl——2
    benchmark_rl = pd.read_excel(data_source, 'Peak_heat_benchmark_2_test')
    indoor_benchmark_rl=np.array(benchmark_rl['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl=np.array(benchmark_rl['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list=np.array(benchmark_rl['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach=float(thermal_kpi_benchmark_rl_list[-1]) - float(thermal_kpi_benchmark_rl_list[0])
    energy_kpi_benchmark_rl_list=np.array(benchmark_rl['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach=float(energy_kpi_benchmark_rl_list[-1]) - float(energy_kpi_benchmark_rl_list[0])
    cost_kpi_benchmark_rl_list=np.array(benchmark_rl['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach=float(cost_kpi_benchmark_rl_list[-1]) - float(cost_kpi_benchmark_rl_list[0])

    # benchmark_rl——3
    benchmark_rl_3 = pd.read_excel(data_source, 'Peak_heat_benchmark_3_test')
    indoor_benchmark_rl_3 = np.array(benchmark_rl_3['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl_3 = np.array(benchmark_rl_3['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach_3 = float(thermal_kpi_benchmark_rl_list_3[-1]) - float(
        thermal_kpi_benchmark_rl_list_3[0])
    energy_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach_3 = float(energy_kpi_benchmark_rl_list_3[-1]) - float(energy_kpi_benchmark_rl_list_3[0])
    cost_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach_3 = float(cost_kpi_benchmark_rl_list_3[-1]) - float(cost_kpi_benchmark_rl_list_3[0])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle('Test scenario: Peak heat days',y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0:2, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_our_approach, label='our method', color='cyan', linewidth=3)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1', color='blue')
    ax1.plot(indoor_benchmark_rl, label='Benchmark 2', color='green')
    ax1.plot(indoor_benchmark_rl_3, label='Benchmark 3', color='olive')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')
    # ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

    # # Second row spanning all columns
    # ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    # ax2.set_title('Outdoor temperature')
    # ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    # ax2.set_ylabel('Temperature (°C)')
    # ax2.legend(loc="lower right")

    specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax1.set_xticks(index_dates)
    ax1.set_xticklabels(specific_dates)

    # Second row spanning all columns
    # ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    # ax2.set_title('Outdoor temperature')
    # ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    # ax2.set_ylabel('Temperature (°C)')
    # ax2.legend(loc="lower right")
    #
    # specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    # index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    # ax2.set_xticks(index_dates)
    # ax2.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['cyan', 'blue', 'green', 'olive']
    labels = ['our method', 'Benchmark 1', 'Benchmark 2', 'Benchmark 3']
    datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
    benchmarks = [[thermal_kpi_rule, thermal_kpi_benchmark_rl_approach,thermal_kpi_benchmark_rl_approach_3],
                  [energy_kpi_rule, energy_kpi_benchmark_rl_approach,energy_kpi_benchmark_rl_approach_3],
                  [cost_kpi_rule, cost_kpi_benchmark_rl_approach,cost_kpi_benchmark_rl_approach_3]]

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

    # plt.show()


    def thermal(t,coo,hea):
        result=(max((max((t-coo),0)+max((hea-t),0)),1))
        return result


    c_our,c_rule,c_ben_rl,c_ben_rl_3=0,0,0,0
    for x in range(len(all_conumsption_benchmark_rule)-1):
        c_our += price[x] * all_conumsption_our_approach[x]
        c_rule   += price[x] * all_conumsption_benchmark_rule[x]
        c_ben_rl += price[x] * all_conumsption_benchmark_rl[x]
        c_ben_rl_3+= price[x] * all_conumsption_benchmark_rl_3[x]

    FI_our =1-c_our/c_rule
    FI_bench_rl =1-c_ben_rl/c_rule
    FI_bench_rl_3 = 1 - c_ben_rl_3 / c_rule
    print(FI_our)
    print(FI_bench_rl)
    print(FI_bench_rl_3)

    c_0_NEW, c_our_NEW ,p_0_new, p_our_new, p_rl,c_rl_new=0,0,0,0,0,0

        # c_0_NEW   += price[x] * all_conumsption_benchmark_rule[x] * xx
    print('new FI index is: ')

    # FI_our_NEW =1-(c_our_NEW+p_our_new)/(c_0_NEW+p_0_new)
    # FI_rl_NEW =1-(p_rl+c_rl_new)/(c_0_NEW+p_0_new)
    penalty_our=energy_kpi_our_approach*(1+thermal_kpi_our_approach * occup)
    penalty_rl=energy_kpi_benchmark_rl_approach*(1+thermal_kpi_benchmark_rl_approach * occup)
    penalty_rl_3 = energy_kpi_benchmark_rl_approach_3 * (1 + thermal_kpi_benchmark_rl_approach_3 * occup)
    penalty_rule=energy_kpi_rule*(1+thermal_kpi_rule * occup)
    FI_our_NEW = 1 - (c_our * (1 + penalty_our)) / \
                 (c_rule * (1 + penalty_rule))
    FI_rl_NEW = 1 - (c_ben_rl * (1 + penalty_rl)) / \
                (c_rule * (1 + penalty_rule))
    FI_rl_NEW_3 = 1 - (c_ben_rl_3 * (1 + penalty_rl_3)) / \
                (c_rule * (1 + penalty_rule))

    print(FI_our_NEW)
    print(FI_rl_NEW)
    print(FI_rl_NEW_3)



    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    FIs = [FI_our, FI_bench_rl,FI_bench_rl_3]
    FIs_new = [FI_our_NEW, FI_rl_NEW,FI_rl_NEW_3]
    methods = ['Our approach', 'Benchmark 2', 'Benchmark 3']
    bar_color = ['cyan', 'green','olive']

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle('Quantitative assessment of flexibility index (FI) enhancements against rule-based controller',y=1)  # Set a title for the whole figure

    ax1 = fig.add_subplot(gs[0, 0]) # ax1 takes the entire first row
    ax2 = fig.add_subplot(gs[0, 1]) # ax2 takes the entire second row

    # Function to add value labels above each bar
    def add_value_labels(ax, x, y):
        for i in range(len(x)):
            ax.text(i, y[i] + 0.002, f'{y[i]:.2f}', ha='center', va='bottom')

    # First plot - original
    ax1.bar(methods, FIs, color=bar_color)
    add_value_labels(ax1, methods, FIs)
    ax1.set_ylabel('Original FI')
    ax1.set_title('Original FI')

    # Second plot - new
    ax2.bar(methods, FIs_new, color=bar_color)
    add_value_labels(ax2, methods, FIs_new)
    ax2.set_ylabel('The proposed FI')
    ax2.set_title('The proposed FI')

    # Adjust layout at the end
    fig.tight_layout()
    plt.show()
else:
    ###############################################################################################################################################
    ##########peak cool################################################################
    #our method
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

    # benchmark_rl——2
    benchmark_rl = pd.read_excel(data_source, 'Peak_cool_benchmark_2_test')
    indoor_benchmark_rl = np.array(benchmark_rl['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl = np.array(benchmark_rl['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list = np.array(benchmark_rl['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach = float(thermal_kpi_benchmark_rl_list[-1]) - float(
        thermal_kpi_benchmark_rl_list[0])
    energy_kpi_benchmark_rl_list = np.array(benchmark_rl['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach = float(energy_kpi_benchmark_rl_list[-1]) - float(energy_kpi_benchmark_rl_list[0])
    cost_kpi_benchmark_rl_list = np.array(benchmark_rl['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach = float(cost_kpi_benchmark_rl_list[-1]) - float(cost_kpi_benchmark_rl_list[0])

    # benchmark_rl——3
    benchmark_rl_3 = pd.read_excel(data_source, 'Peak_cool_benchmark_3_test')
    indoor_benchmark_rl_3 = np.array(benchmark_rl_3['indoor_temperature']).astype(float)
    all_conumsption_benchmark_rl_3 = np.array(benchmark_rl_3['all_consumption']).astype(float)
    thermal_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['themal_kpi']).astype(float)
    thermal_kpi_benchmark_rl_approach_3 = float(thermal_kpi_benchmark_rl_list_3[-1]) - float(
        thermal_kpi_benchmark_rl_list_3[0])
    energy_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['energy_kpi']).astype(float)
    energy_kpi_benchmark_rl_approach_3 = float(energy_kpi_benchmark_rl_list_3[-1]) - float(
        energy_kpi_benchmark_rl_list_3[0])
    cost_kpi_benchmark_rl_list_3 = np.array(benchmark_rl_3['cost_kpi']).astype(float)
    cost_kpi_benchmark_rl_approach_3 = float(cost_kpi_benchmark_rl_list_3[-1]) - float(cost_kpi_benchmark_rl_list_3[0])

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle('Test scenario: Peak cool days', y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0:2, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Indoor temperature')
    ax1.plot(indoor_our_approach, label='our method', color='cyan', linewidth=3)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1', color='blue')
    ax1.plot(indoor_benchmark_rl, label='Benchmark 2', color='green')
    ax1.plot(indoor_benchmark_rl_3, label='Benchmark 3', color='olive')
    ax1.plot(up_boundary, color='gray')
    ax1.plot(low_boundary, color='gray')
    ax1.legend(loc="lower right")
    ax1.set_ylabel('Temperature (°C)')
    # ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

    # # Second row spanning all columns
    # ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    # ax2.set_title('Outdoor temperature')
    # ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    # ax2.set_ylabel('Temperature (°C)')
    # ax2.legend(loc="lower right")

    specific_dates = ['Day 282', 'Day 284', 'Day 286', 'Day 288', 'Day 290', 'Day 292', 'Day 294', 'Day 296']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax1.set_xticks(index_dates)
    ax1.set_xticklabels(specific_dates)

    # Second row spanning all columns
    # ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    # ax2.set_title('Outdoor temperature')
    # ax2.plot(outdoor_temp, color='navy', label='Outdoor air temperature')
    # ax2.set_ylabel('Temperature (°C)')
    # ax2.legend(loc="lower right")
    #
    # specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    # index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    # ax2.set_xticks(index_dates)
    # ax2.set_xticklabels(specific_dates)

    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['cyan', 'blue', 'green', 'olive']
    labels = ['our method', 'Benchmark 1', 'Benchmark 2', 'Benchmark 3']
    datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
    benchmarks = [[thermal_kpi_rule, thermal_kpi_benchmark_rl_approach, thermal_kpi_benchmark_rl_approach_3],
                  [energy_kpi_rule, energy_kpi_benchmark_rl_approach, energy_kpi_benchmark_rl_approach_3],
                  [cost_kpi_rule, cost_kpi_benchmark_rl_approach, cost_kpi_benchmark_rl_approach_3]]

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


    # plt.show()

    def thermal(t, coo, hea):
        result = (max((max((t - coo), 0) + max((hea - t), 0)), 1))
        return result


    c_our, c_rule, c_ben_rl, c_ben_rl_3 = 0, 0, 0, 0
    for x in range(len(all_conumsption_benchmark_rule) - 1):
        c_our += price[x] * all_conumsption_our_approach[x]
        c_rule += price[x] * all_conumsption_benchmark_rule[x]
        c_ben_rl += price[x] * all_conumsption_benchmark_rl[x]
        c_ben_rl_3 += price[x] * all_conumsption_benchmark_rl_3[x]

    FI_our = 1 - c_our / c_rule
    FI_bench_rl = 1 - c_ben_rl / c_rule
    FI_bench_rl_3 = 1 - c_ben_rl_3 / c_rule
    print(FI_our)
    print(FI_bench_rl)
    print(FI_bench_rl_3)

    c_0_NEW, c_our_NEW, p_0_new, p_our_new, p_rl, c_rl_new = 0, 0, 0, 0, 0, 0

    # c_0_NEW   += price[x] * all_conumsption_benchmark_rule[x] * xx
    print('new FI index is: ')

    # FI_our_NEW =1-(c_our_NEW+p_our_new)/(c_0_NEW+p_0_new)
    # FI_rl_NEW =1-(p_rl+c_rl_new)/(c_0_NEW+p_0_new)
    penalty_our = energy_kpi_our_approach * (1 + thermal_kpi_our_approach * occup)
    penalty_rl = energy_kpi_benchmark_rl_approach * (1 + thermal_kpi_benchmark_rl_approach * occup)
    penalty_rl_3 = energy_kpi_benchmark_rl_approach_3 * (1 + thermal_kpi_benchmark_rl_approach_3 * occup)
    penalty_rule = energy_kpi_rule * (1 + thermal_kpi_rule * occup)
    FI_our_NEW = 1 - (c_our * (1 + penalty_our)) / \
                 (c_rule * (1 + penalty_rule))
    FI_rl_NEW = 1 - (c_ben_rl * (1 + penalty_rl)) / \
                (c_rule * (1 + penalty_rule))
    FI_rl_NEW_3 = 1 - (c_ben_rl_3 * (1 + penalty_rl_3)) / \
                  (c_rule * (1 + penalty_rule))

    print(FI_our_NEW)
    print(FI_rl_NEW)
    print(FI_rl_NEW_3)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    FIs = [FI_our, FI_bench_rl, FI_bench_rl_3]
    FIs_new = [FI_our_NEW, FI_rl_NEW, FI_rl_NEW_3]
    methods = ['Our approach', 'Benchmark 2', 'Benchmark 3']
    bar_color = ['cyan', 'green', 'olive']

    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    fig.suptitle('Quantitative assessment of flexibility index (FI) enhancements against rule-based controller',
                 y=1)  # Set a title for the whole figure

    ax1 = fig.add_subplot(gs[0, 0])  # ax1 takes the entire first row
    ax2 = fig.add_subplot(gs[0, 1])  # ax2 takes the entire second row


    # Function to add value labels above each bar
    def add_value_labels(ax, x, y):
        for i in range(len(x)):
            ax.text(i, y[i] + 0.002, f'{y[i]:.2f}', ha='center', va='bottom')


    # First plot - original
    ax1.bar(methods, FIs, color=bar_color)
    add_value_labels(ax1, methods, FIs)
    ax1.set_ylabel('Original FI')
    ax1.set_title('Original FI')

    # Second plot - new
    ax2.bar(methods, FIs_new, color=bar_color)
    add_value_labels(ax2, methods, FIs_new)
    ax2.set_ylabel('The proposed FI')
    ax2.set_title('The proposed FI')

    # Adjust layout at the end
    fig.tight_layout()
    plt.show()