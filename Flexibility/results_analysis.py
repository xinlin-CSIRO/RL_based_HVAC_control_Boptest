import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




debug_model=1
occup=1
cool_or_heat='boptest'#'boptest'#or 'CSIRO'

if(cool_or_heat=='boptest'):
    file = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\2025 Conference\all_results_in_one.xlsx"
    data_source = pd.ExcelFile(file)
    #peak heat scenario
    #our method
    our_method = pd.read_excel(data_source, 'Peak_heat_our')
    head = 14400
    end = len(our_method)-1
    indoor_our_approach=np.array(our_method['indoor_temperature'][head:end]).astype(float)
    all_conumsption_our_approach=np.array(our_method['all_consumption'][head:end]).astype(float)
    thermal_kpi_our_list=np.array(our_method['themal_kpi'][head:end]).astype(float)
    thermal_kpi_our_approach=float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
    energy_kpi_our_list=np.array(our_method['energy_kpi'][head:end]).astype(float)
    energy_kpi_our_approach=float(energy_kpi_our_list[-1]) - float(energy_kpi_our_list[0])
    cost_kpi_our_list=np.array(our_method['cost_kpi'][head:end]).astype(float)
    cost_kpi_our_approach=float(cost_kpi_our_list[-1]) - float(cost_kpi_our_list[0])
    outdoor_temp=np.array(our_method['outdoor_air'][head:end]).astype(float)
    price=np.array(our_method['price'][head:end]).astype(float)
    step_rewards = np.array(our_method['reward'][0:head]).astype(float)
    #rule_based

    rule_based = pd.read_excel(data_source, 'Peak_heat_pid')
    head = 0
    end = len(rule_based) - 1
    indoor_benchmark_rule=np.array(rule_based['indoor_temperature'][head:end])
    all_conumsption_benchmark_rule=np.array(rule_based['all_consumption'][head:end])
    thermal_kpi_benchmark_rule=np.array(rule_based['themal_kpi'][head:end])
    price=np.array(rule_based['price'][head:end])
    up_boundary=np.array(rule_based['boundary_1'][head:end])
    low_boundary=np.array(rule_based['boundary_0'][head:end])

    thermal_kpi_rule_list=np.array(rule_based['themal_kpi'][head:end]).astype(float)
    thermal_kpi_rule=float(thermal_kpi_rule_list[-1])
    energy_kpi_rule_list=np.array(rule_based['energy_kpi'][head:end]).astype(float)
    energy_kpi_rule=float(energy_kpi_rule_list[-1])
    cost_kpi_rule_list=np.array(rule_based['cost_kpi'][head:end]).astype(float)
    cost_kpi_rule=float(cost_kpi_rule_list[-1])

    rl_1 = pd.read_excel(data_source, 'Peak_heat_rl_1')
    head = int(14400+24*0)
    end = len(rl_1) - 1
    print(end)
    indoor_rl_1 = np.array(rl_1['indoor_temperature'][head:end]).astype(float)
    all_conumsption_rl_1 = np.array(rl_1['all_consumption'][head:end]).astype(float)
    thermal_kpi_rl_1 = np.array(rl_1['themal_kpi'][head:end]).astype(float)
    thermal_kpi_rl_1 = float(thermal_kpi_rl_1[-1]) - float(thermal_kpi_rl_1[0])
    energy_kpi_rl_1 = np.array(rl_1['energy_kpi'][head:end]).astype(float)
    energy_kpi_rl_1 = float(energy_kpi_rl_1[-1]) - float(energy_kpi_rl_1[0])
    cost_kpi_rl_1 = np.array(rl_1['cost_kpi'][head:end]).astype(float)
    cost_kpi_rl_1 = float(cost_kpi_rl_1[-1]) - float(cost_kpi_rl_1[0])

    rl_2 = pd.read_excel(data_source, 'Peak_heat_rl_2')
    head = int(14400 + 24 * 0)
    end = len(rl_2) - 1
    print(end)
    indoor_rl_2 = np.array(rl_2['indoor_temperature'][head:end]).astype(float)
    all_conumsption_rl_2 = np.array(rl_2['all_consumption'][head:end]).astype(float)
    thermal_kpi_rl_2 = np.array(rl_2['themal_kpi'][head:end]).astype(float)
    thermal_kpi_rl_2 = float(thermal_kpi_rl_2[-1]) - float(thermal_kpi_rl_2[0])
    energy_kpi_rl_2 = np.array(rl_2['energy_kpi'][head:end]).astype(float)
    energy_kpi_rl_2 = float(energy_kpi_rl_2[-1]) - float(energy_kpi_rl_2[0])
    cost_kpi_rl_2 = np.array(rl_2['cost_kpi'][head:end]).astype(float)
    cost_kpi_rl_2 = float(cost_kpi_rl_2[-1]) - float(cost_kpi_rl_2[0])
    print(1)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(16, 18), constrained_layout=True)
    # fig = plt.figure(figsize=(18, 17))
    gs = gridspec.GridSpec(5, 3, figure=fig)
    # fig.suptitle('Bestest air: Peak heat days',y=0.915)  # Set a title for the whole figure

    # First row spanning all columns
    ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
    ax1.set_title('Bestest air: Peak heat days', fontsize=16)
    ax1.plot(indoor_our_approach, label='our method', color='plum', linewidth=2)
    ax1.plot(indoor_benchmark_rule, label='Benchmark 1: rule-based', color='royalblue', linewidth=2)
    ax1.plot(indoor_rl_1, label='Benchmark 2: RL-based', color='darkgreen', linewidth=2)
    ax1.plot(indoor_rl_2, label='Benchmark 3: RL-based', color='darkgoldenrod', linewidth=2)
    ax1.plot(up_boundary, ':', color='gray')
    ax1.plot(low_boundary, ':', color='gray')
    ax1.legend(loc="upper right")
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_xticklabels([])  # Disabling x-axis labels for ax1
    # ax1.text(0.5, -0.05, '(a)', ha='center', va='center', transform=ax1.transAxes, fontsize=12)

    # Second row spanning all columns
    ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
    ax2.set_title('Outdoor temperature')
    ax2.plot(outdoor_temp, color='black', label='Outdoor air temperature')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(loc="lower right")

    specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax2.set_xticks(index_dates)
    # ax2.set_xticklabels(specific_dates)
    ax2.set_xticklabels([])  # Disabling x-axis labels for ax1
    # ax2.text(0.5, -0.05, '(b)', ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    # third row spanning all columns
    ax3 = fig.add_subplot(gs[2, :])  # This adds a subplot that spans the second row entirely
    ax3.set_title('Energy use and ToU')
    # Plot the price on the right y-axis
    ax3_right = ax3.twinx()
    ax3_right.plot(price, color='black', label='ToU')
    ax3_right.set_ylabel('Price (units)')

    # Plot the energy consumption values on the left y-axis
    ax3.plot(all_conumsption_our_approach, label='our method', color='plum', linewidth=2)
    ax3.plot(all_conumsption_benchmark_rule, label='Benchmark 1: rule-based', color='royalblue', linewidth=2)
    ax3.plot(all_conumsption_rl_1, label='Benchmark 2: RL-based', color='darkgreen', linewidth=2)
    ax3.plot(all_conumsption_rl_2, label='Benchmark 3: RL-based', color='darkgoldenrod', linewidth=2)
    ax3.set_ylabel('Power (W)')

    # Combine legends from both axes
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_right.get_legend_handles_labels()
    ax3.legend(lines + lines2, labels + labels2, loc="upper right")

    specific_dates = ['Day 334', 'Day 336', 'Day 338', 'Day 340', 'Day 342', 'Day 344', 'Day 346', 'Day 348']
    index_dates = [i * 24 * 2 for i in range(len(specific_dates))]
    ax3.set_xticks(index_dates)
    ax3.set_xticklabels(specific_dates)

    ax4 = fig.add_subplot(gs[3, :])
    batch_size = 128
    num_batches2 = len(step_rewards) // batch_size
    batch_rewards2 = [
        np.sum(step_rewards[i * batch_size:(i + 1) * batch_size])
        for i in range(num_batches2)
    ]
    timesteps = np.arange(1, num_batches2 + 1) * batch_size
    ma_window = 10  # Adjust the window size for smoother curve
    smoothed_rewards = pd.Series(batch_rewards2).rolling(window=ma_window).mean().to_numpy()
    # Calculate standard deviation for confidence intervals (optional)
    rolling_std = pd.Series(batch_rewards2).rolling(window=ma_window).std().to_numpy()

    decay_factor = np.linspace(1, 0.3, num_batches2)
    adjusted_std = rolling_std * decay_factor

    ax4.plot(timesteps, smoothed_rewards, label='Cumulative reward of the proposed method', color='plum', linewidth=3)

    # Add confidence interval (optional)
    # ax4.fill_between(timesteps, smoothed_rewards - adjusted_std, smoothed_rewards + adjusted_std, color='cyan',
    #                  alpha=0.2)
    # ax4.set_xlim([timesteps[0], timesteps[-1]])
    # ax4.plot(timesteps, batch_rewards)
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Cumulative reward', fontsize=12)
    ax4.legend(loc="lower right", fontsize=12)
    ax4.set_title('Convergence plot of the proposed method', fontsize=12)




    # Third row with three individual plots
    titles = ['Thermal discomfort KPI', 'Energy usage KPI', 'Operational cost KPI']
    colors = ['plum', 'royalblue', 'darkgreen', 'darkgoldenrod']
    labels = ['our method', 'Benchmark 1', 'Benchmark 2', 'Benchmark 3']
    datasets = [thermal_kpi_our_approach, energy_kpi_our_approach, cost_kpi_our_approach]
    benchmarks = [[thermal_kpi_rule, thermal_kpi_rl_1, thermal_kpi_rl_2],
                  [energy_kpi_rule, energy_kpi_rl_1, energy_kpi_rl_2],
                  [cost_kpi_rule, cost_kpi_rl_1, cost_kpi_rl_2]]

    for i in range(3):
        ax = fig.add_subplot(gs[4, i])
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
    fig.tight_layout()
    fig.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\2025 Conference\\BOPTEST_results.png')


    def thermal(t,coo,hea):
        result=(max((max((t-coo),0)+max((hea-t),0)),1))
        return result


    c_our,c_rule,c_rl1,c_rl2=0,0,0,0
    for x in range(len(all_conumsption_benchmark_rule)-1):
        c_our += price[x] * all_conumsption_our_approach[x]
        c_rule   += price[x] * all_conumsption_benchmark_rule[x]
        c_rl1   +=price[x] * all_conumsption_rl_1[x]
        c_rl2 += price[x] * all_conumsption_rl_2[x]

    FI_our =1-c_our/c_rule
    FI_rl_1 =1-c_rl1/c_rule
    FI_rl_2 = 1 - c_rl2 / c_rule
    print('orginial FI our:',FI_our)
    print('orginial FI rl1:',FI_rl_1)
    print('orginial FI rl2:',FI_rl_2)

    c_0_NEW, c_our_NEW ,p_0_new, p_our_new, p_rl,c_rl_new=0,0,0,0,0,0

    print('new FI index is: ')

    penalty_our=energy_kpi_our_approach*(1+thermal_kpi_our_approach )
    penalty_rl_1=energy_kpi_rl_1*(1+thermal_kpi_rl_1)
    penalty_rl_2 = energy_kpi_rl_2*(1+thermal_kpi_rl_2)
    penalty_rule=energy_kpi_rule*(1+thermal_kpi_rule )
    FI_our_NEW = 1 - (c_our * (1 + penalty_our)) / \
                 (c_rule * (1 + penalty_rule))

    FI_our_NEW = 1 - (1 * (1 + penalty_our)) / \
                 (1 * (1 + penalty_rule))

    FI_rl_1_NEW = 1 - (c_rl1 * (1 + penalty_rl_1)) / \
                 (c_rule * (1 + penalty_rule))
    FI_rl_2_NEW = 1 - (c_rl2 * (1 + penalty_rl_2)) / \
                  (c_rule * (1 + penalty_rule))


    print(FI_our_NEW)
    print(FI_rl_1_NEW)
    print(FI_rl_2_NEW)




    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec


    FIs = [FI_our, FI_rl_1,FI_rl_2]
    FIs_new = [FI_our_NEW, FI_rl_1_NEW,FI_rl_2_NEW]
    bar_color = ['plum',  'darkgreen', 'darkgoldenrod']
    methods = ['Our approach', 'Benchmark 2', 'Benchmark 3']


    fig = plt.figure(constrained_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0]) # ax1 takes the entire first row
    ax2 = fig.add_subplot(gs[0, 1]) # ax2 takes the entire second row

    # Function to add value labels above each bar
    def add_value_labels(ax, x, y):
        for i in range(len(x)):
            ax.text(i, y[i] + 0.002, f'{y[i]:.2f}', ha='center', va='bottom')

    # First plot - original
    ax1.bar(methods, FIs, color=bar_color)
    add_value_labels(ax1, methods, FIs)
    ax1.set_ylabel('The conventional FI')
    ax1.set_title('The conventional FI')

    # Second plot - new
    ax2.bar(methods, FIs_new, color=bar_color)
    add_value_labels(ax2, methods, FIs_new)
    ax2.set_ylabel('The proposed $FI_{improved}$')
    ax2.set_title('The proposed $FI_{improved}$')

    # Adjust layout at the end
    fig.tight_layout()
    plt.show()
else:
    if (cool_or_heat == 'CSIRO'):
        file = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RL_WORK_SUMMARY\2025 Conference\CSIRO_GYM_results_1.xlsx"
        data_source = pd.ExcelFile(file)
        our_method = pd.read_excel(data_source, 'test')
        head = 0
        end = len(our_method)
        indoor_our_approach = np.array(our_method['indoor_temp'][head:end]).astype(float)
        all_conumsption_our_approach = np.array(our_method['cooling_usage'][head:end]).astype(float)*3
        up_boundary = np.array(our_method['boundary_1'][head:end])
        low_boundary = np.array(our_method['boundary_0'][head:end])
        price = np.array(our_method['price'][head:end])
        outdoor_temp = np.array(our_method['outdoor air'][head:end]).astype(float)
        reward = np.array(pd.read_excel(data_source, 'New_Newcastle_site_gym_19th-Oct')['reward'][head:int(174816/4.1)]).astype(float)

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # Create a figure with GridSpec layout
        fig = plt.figure(figsize=(16, 14))
        gs = gridspec.GridSpec(4, 3, figure=fig)
        # fig.suptitle('Bestest air: Peak heat days',y=0.915)  # Set a title for the whole figure

        # First row spanning all columns
        ax1 = fig.add_subplot(gs[0, :])  # This adds a subplot that spans the first row entirely
        ax1.set_title('ReaLMLSim')
        ax1.plot(indoor_our_approach, label='our method', color='plum', linewidth=2)
        ax1.plot(up_boundary, ':', color='gray')
        ax1.plot(low_boundary, ':', color='gray')
        ax1.legend(loc="upper right")
        ax1.set_ylabel('Temperature (°C)')

        ax1.set_xticklabels([])  # Disabling x-axis labels for ax1

        # Second row spanning all columns
        ax2 = fig.add_subplot(gs[1, :])  # This adds a subplot that spans the second row entirely
        ax2.set_title('Outdoor temperature')
        ax2.plot(outdoor_temp, color='black', label='Outdoor air temperature')
        ax2.set_ylabel('Temperature (°C)')
        ax2.legend(loc="lower right")

        ax2.set_xticklabels([])

        # third row spanning all columns
        ax3 = fig.add_subplot(gs[2, :])  # This adds a subplot that spans the second row entirely
        ax3.set_title('Energy use and ToU')
        # Plot the price on the right y-axis
        ax3_right = ax3.twinx()
        ax3_right.plot(price, color='black', label='ToU')
        ax3_right.set_ylabel('Price (units)')

        # Plot the energy consumption values on the left y-axis
        ax3.plot(all_conumsption_our_approach, label='our method', color='plum', linewidth=2)
        ax3.set_ylabel('Power (kW)')

        # Combine legends from both axes
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_right.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc="upper right")

        specific_dates = ['Dec 20th', 'Dec 22th','Dec 24th', 'Dec 26th', 'Dec 28th', 'Dec 30th', 'Jan 1st', 'Jan 3rd']
        index_dates = [i * 144 * 2 for i in range(len(specific_dates))]
        ax3.set_xticks(index_dates)
        ax3.set_xticklabels(specific_dates)


        ax4 = fig.add_subplot(gs[3, :])
        batch_size = 144
        num_batches2 = len(reward) // batch_size
        batch_rewards2 = [
            np.sum(reward[i * batch_size:(i + 1) * batch_size])
            for i in range(num_batches2)
        ]
        timesteps = np.arange(1, num_batches2 + 1) * batch_size
        ma_window =30  # Adjust the window size for smoother curve
        smoothed_rewards = pd.Series(batch_rewards2).rolling(window=ma_window).mean().to_numpy()
        # Calculate standard deviation for confidence intervals (optional)
        rolling_std = pd.Series(batch_rewards2).rolling(window=ma_window).std().to_numpy()

        decay_factor = np.linspace(0.6, 0.6, num_batches2)
        adjusted_std = rolling_std * decay_factor

        ax4.plot(timesteps, smoothed_rewards, label='Cumulative reward of the proposed method', color='plum', linewidth=3)

        # Add confidence interval (optional)
        # ax4.fill_between(timesteps, smoothed_rewards - adjusted_std, smoothed_rewards + adjusted_std, color='cyan',
        #                  alpha=0.2)
        # ax4.plot(timesteps, batch_rewards)
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Cumulative reward', fontsize=12)
        ax4.legend(loc="lower right", fontsize=12)
        ax4.set_title('Convergence plot of the proposed method', fontsize=12)

        # Explicitly clear automatic date locators or formatters
        ax4.xaxis.set_major_locator(plt.NullLocator())
        ax4.xaxis.set_minor_locator(plt.NullLocator())

        #Manually set custom ticks and labels
        # specific_data = [' ', '2000', '4000', '6000', '8000', '10000', '12000', '14000', '16000']
        # length_ = len(timesteps)-6
        # index_loc = [i * int(length_ / (len(specific_data) - 1)) for i in range(len(specific_data))]
        #
        # ax4.set_xticks(timesteps[index_loc])
        # ax4.set_xticklabels(specific_data, fontsize=10)


        # Ensure the ticks match the data points properly

        fig.tight_layout()
        fig.savefig('C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\2025 Conference\\csiro_1.png')
        plt.show()