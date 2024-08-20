import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
head=2408
end=2640


epsilon = 0.0001
price = 0.2383# 0.2666
#0.2383, 0.2666
max_price=0.2666+0.02
min_price=0.2383-0.02

if abs(price - min_price) < epsilon:
    normalized_price = 0

elif abs(price - 0.2666) < epsilon:
    normalized_price =  1* (0.2666 - min_price) / (max_price - min_price)

elif abs(price - 0.2383) < epsilon:
    normalized_price =  1* (0.2383 -min_price) / (max_price - min_price)

elif abs(price - max_price) < epsilon:
    normalized_price = 1

print(normalized_price)
print(1-normalized_price)



#new reward_SAC
file="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\best_air_comparsion\\"
our_approach=file+'our_approach_2.csv'
indoor_our_approach=np.array(pd.read_csv(our_approach)['indoor_temp'])[head: end]
all_conumsption_our_approach=np.array(pd.read_csv(our_approach)['all_consumption'])[head: end]
thermal_kpi_our_approach=np.array(pd.read_csv(our_approach)['themal_kpi'])[end-1]-np.array(pd.read_csv(our_approach)['themal_kpi'])[head]
energy_kpi_our_approach=np.array(pd.read_csv(our_approach)['energy_kpi'])[end-1]-np.array(pd.read_csv(our_approach)['energy_kpi'])[head]
cost_kpi_our_approach=np.array(pd.read_csv(our_approach)['cost_kpi'])[end-1]-np.array(pd.read_csv(our_approach)['cost_kpi'])[head]
print('thermal_kpi_our_approach= ',thermal_kpi_our_approach)
print('energy_kpi_our_approach =',energy_kpi_our_approach)
print('cost_kpi_our_approach =',cost_kpi_our_approach)

up_boundary=np.array(pd.read_csv(our_approach)['boundary_1'])[head: end]
low_boundary=np.array(pd.read_csv(our_approach)['boundary_0'])[head: end]
price=np.array(pd.read_csv(our_approach)['price'])[head: end]
outdoor=np.array(pd.read_csv(our_approach)['outdoor_air'])[head: end]

# #new reward_A2C
# new_reward_with_A2C="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\new_reward_A2C.csv"
# indoor_new_reward_with_A2C=np.array(pd.read_csv(new_reward_with_A2C)['indoor temp'])[head: end]
# all_conumsption_new_reward_with_A2C=np.array(pd.read_csv(new_reward_with_A2C)['all_consumption'])[head: end]
# thermal_kpi_new_reward_with_A2C=np.array(pd.read_csv(new_reward_with_A2C)[' themal kpi '])[end-1]-np.array(pd.read_csv(new_reward_with_A2C)[' themal kpi '])[head]
# energy_kpi_new_reward_with_A2C=np.array(pd.read_csv(new_reward_with_A2C)['energy kpi'])[end-1]-np.array(pd.read_csv(new_reward_with_A2C)['energy kpi'])[head]
#
# #old reward_SAC
# old_reward_with_SAC="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\old_reward_SAC.csv"
# indoor_old_reward_with_SAC=np.array(pd.read_csv(old_reward_with_SAC)['indoor temp'])[head: end]
# all_conumsption_old_reward_with_SAC=np.array(pd.read_csv(old_reward_with_SAC)['all_consumption'])[head: end]
# thermal_kpi_old_reward_with_SAC=np.array(pd.read_csv(old_reward_with_SAC)[' themal kpi '])[end-1]-np.array(pd.read_csv(old_reward_with_SAC)[' themal kpi '])[head]
# energy_kpi_old_reward_with_SAC=np.array(pd.read_csv(old_reward_with_SAC)['energy kpi'])[end-1]-np.array(pd.read_csv(old_reward_with_SAC)['energy kpi'])[head]
#
# #old reward_A2C
# old_reward_with_A2C="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\old_reward_A2C.csv"
# indoor_old_reward_with_A2C=np.array(pd.read_csv(old_reward_with_A2C)['indoor temp'])[head: end]
# all_conumsption_old_reward_with_A2C=np.array(pd.read_csv(old_reward_with_A2C)['all_consumption'])[head: end]
# thermal_kpi_old_reward_with_A2C=np.array(pd.read_csv(old_reward_with_A2C)[' themal kpi '])[end-1]-np.array(pd.read_csv(old_reward_with_A2C)[' themal kpi '])[head]
# energy_kpi_old_reward_with_A2C=np.array(pd.read_csv(old_reward_with_A2C)['energy kpi'])[end-1]-np.array(pd.read_csv(old_reward_with_A2C)['energy kpi'])[head]


#rl benchmark
benchmark_rl=file+"benchmark_rl.csv"
indoor_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['indoor_temp'])[head: end]
all_conumsption_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['all_consumption'])[head: end]
thermal_kpi_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['themal_kpi'])[head]
energy_kpi_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['energy_kpi'])[head]
cost_kpi_benchmarl_rl=np.array(pd.read_csv(benchmark_rl)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['cost_kpi'])[head]
print('thermal_kpi_benchmark_rl= ',thermal_kpi_benchmark_rl)
print('energy_kpi_benchmark_rl =',energy_kpi_benchmark_rl)
print('cost_kpi_benchmarl_rl =',cost_kpi_benchmarl_rl)
############################################################################################################################
#rule benchmark
benchmark_rule=file+"benchmark_rule.csv"
indoor_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['indoor_temp'])[head: end]
all_conumsption_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['all_consumption'])[head: end]
thermal_kpi_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rule)['themal_kpi'])[head]
energy_kpi_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rule)['energy_kpi'])[head]
cost_kpi_benchmarl_rule=np.array(pd.read_csv(benchmark_rule)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rule)['cost_kpi'])[head]
print('thermal_kpi_benchmark_rule= ',thermal_kpi_benchmark_rule)
print('energy_kpi_benchmark_rule =',energy_kpi_benchmark_rule)
print('cost_kpi_benchmarl_rule =',cost_kpi_benchmarl_rule)
############################################################################################################################

fig, ax1 = plt.subplots()
ax1.plot(indoor_our_approach,'r-',  label='our method')
ax1.plot(indoor_benchmark_rl, 'b-', label='benchmark_rl')
ax1.plot(indoor_benchmark_rule, 'y-', label='benchmark_rule')
ax1.plot(up_boundary, 'g:')
ax1.plot(low_boundary, 'g:')
ax1.plot(outdoor,'navy',  label='outdoor air temp')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Temperature C', color='k')
ax1.plot(up_boundary, 'g:')
ax1.plot(low_boundary, 'g:')

plt.legend(loc="lower right")

plt.show()
#
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'old_reward_with_SAC']
# counts = [energy_kpi_new_reward_with_SAC, energy_kpi_old_reward_with_SAC]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Energy Usage KPI')
# ax.set_title('Energy Usage KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'old_reward_with_SAC']
# counts = [thermal_kpi_new_reward_with_SAC, thermal_kpi_old_reward_with_SAC]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Thermal discomfort KPI')
# ax.set_title('Thermal discomfort KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax1 = plt.subplots()
# ax1.plot(indoor_new_reward_with_A2C,'g-', label='new_reward_with_A2C')
# ax1.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Temperature C', color='g')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
#
# # ax2 = ax1.twinx()
# # ax2.plot(price, 'b-') # 使用蓝色线条
# # ax2.set_ylabel('Price', color='b')
#
# # plt.plot(indoor_new_reward_with_SAC, label='new_reward_with_SAC')
# # # plt.plot(indoor_new_reward_with_A2C, label='new_reward_with_A2C')
# # # plt.plot(indoor_old_reward_with_SAC, label='old_reward_with_SAC')
# # # plt.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# # # plt.plot(indoor_inherent, label='Benchmark')
# plt.legend(loc="upper right")
#
# plt.show()
#
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_A2C', 'old_reward_with_A2C']
# counts = [energy_kpi_new_reward_with_A2C, energy_kpi_old_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Energy Usage KPI')
# ax.set_title('Energy Usage KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_A2C', 'old_reward_with_A2C']
# counts = [thermal_kpi_new_reward_with_A2C, thermal_kpi_old_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Thermal discomfort KPI')
# ax.set_title('Thermal discomfort KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax1 = plt.subplots()
# ax1.plot(indoor_new_reward_with_SAC,'g-', label='new_reward_with_SAC')
# ax1.plot(indoor_new_reward_with_A2C, label='new_reward_with_A2C')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Temperature C', color='g')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
#
# # ax2 = ax1.twinx()
# # ax2.plot(price, 'b-') # 使用蓝色线条
# # ax2.set_ylabel('Price', color='b')
#
# # plt.plot(indoor_new_reward_with_SAC, label='new_reward_with_SAC')
# # # plt.plot(indoor_new_reward_with_A2C, label='new_reward_with_A2C')
# # # plt.plot(indoor_old_reward_with_SAC, label='old_reward_with_SAC')
# # # plt.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# # # plt.plot(indoor_inherent, label='Benchmark')
# plt.legend(loc="upper right")
#
# plt.show()
#
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'new_reward_with_A2C']
# counts = [energy_kpi_new_reward_with_SAC, energy_kpi_new_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Energy Usage KPI')
# ax.set_title('Energy Usage KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'new_reward_with_A2C']
# counts = [thermal_kpi_new_reward_with_SAC, thermal_kpi_new_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Thermal discomfort KPI')
# ax.set_title('Thermal discomfort KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
# #########################################################################################
# fig, ax1 = plt.subplots()
# ax1.plot(indoor_new_reward_with_SAC,'g-', label='new_reward_with_SAC')
# ax1.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Temperature C', color='g')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
#
# # ax2 = ax1.twinx()
# # ax2.plot(price, 'b-') # 使用蓝色线条
# # ax2.set_ylabel('Price', color='b')
#
# # plt.plot(indoor_new_reward_with_SAC, label='new_reward_with_SAC')
# # # plt.plot(indoor_new_reward_with_A2C, label='new_reward_with_A2C')
# # # plt.plot(indoor_old_reward_with_SAC, label='old_reward_with_SAC')
# # # plt.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# # # plt.plot(indoor_inherent, label='Benchmark')
# plt.legend(loc="upper right")
#
# plt.show()
#
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'old_reward_with_A2C']
# counts = [energy_kpi_new_reward_with_SAC, energy_kpi_old_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Energy Usage KPI')
# ax.set_title('Energy Usage KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax = plt.subplots()
#
# fruits = ['new_reward_with_SAC', 'old_reward_with_A2C']
# counts = [thermal_kpi_new_reward_with_SAC, thermal_kpi_old_reward_with_A2C]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Thermal discomfort KPI')
# ax.set_title('Thermal discomfort KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
# #########################################################################################
#
#
# #########################################################################################
# fig, ax1 = plt.subplots()
# ax1.plot(indoor_new_reward_with_SAC,'g-', label='Our method')
# ax1.plot(indoor_inherent, label='Benchmark')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Temperature C', color='g')
# ax1.plot(up_boundary, 'g:')
# ax1.plot(low_boundary, 'g:')
#
# # ax2 = ax1.twinx()
# # ax2.plot(price, 'b-') # 使用蓝色线条
# # ax2.set_ylabel('Price', color='b')
#
# # plt.plot(indoor_new_reward_with_SAC, label='new_reward_with_SAC')
# # # plt.plot(indoor_new_reward_with_A2C, label='new_reward_with_A2C')
# # # plt.plot(indoor_old_reward_with_SAC, label='old_reward_with_SAC')
# # # plt.plot(indoor_old_reward_with_A2C, label='old_reward_with_A2C')
# # # plt.plot(indoor_inherent, label='Benchmark')
# plt.legend(loc="upper right")
#
# plt.show()
#
#
#
# fig, ax = plt.subplots()
#
# fruits = ['Our approach', 'benchmark']
# counts = [energy_kpi_new_reward_with_SAC, energy_kpi_inherent]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Energy Usage KPI')
# ax.set_title('Energy Usage KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
#
#
# fig, ax = plt.subplots()
#
# fruits = ['Our approach', 'benchmark']
# counts = [thermal_kpi_new_reward_with_SAC, thermal_kpi_inherent]
# bar_labels = ['red', 'blue']
# bar_colors = ['tab:green', 'tab:blue']
#
# ax.bar(fruits, counts, label=bar_labels, color=bar_colors)
#
# ax.set_ylabel('Thermal discomfort KPI')
# ax.set_title('Thermal discomfort KPI')
# # ax.legend(title='Energy Usage KPI')
# plt.show()
# #########################################################################################

# print(1)



wo_u=all_conumsption_benchmark_rl #pd.read_excel(source_file)['all_consumption_wo_Price']
wo_u_2=all_conumsption_benchmark_rule #pd.read_excel(source_file)['all_consumption_wo_Price']
w_u= all_conumsption_our_approach #pd.read_excel(source_file)['all_consumption_w_Price']
price=price #pd.read_excel(source_file)['price']
c_1,c_0=0,0
outputs="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\Flexibility_.csv"
fig, ax1 = plt.subplots()
line1, = ax1.plot(w_u, 'r-', label='our approach')  # Note the comma, which unpacks the list of lines returned by plot
line2, = ax1.plot(wo_u, 'b-', label='benchmark_rl')
line3, = ax1.plot(wo_u_2, 'y-', label='benchmark_rule')

ax1.set_xlabel('Time step')
ax1.set_ylabel('Energy consumption', color='k')
ax2 = ax1.twinx()
line4, = ax2.plot(price, 'navy', label='price')  # Note the comma here as well

ax2.set_ylabel('Price', color='navy')

# Combining handles (lines) and labels from both axes
lines = [line1, line2, line3, line4]
labels = [l.get_label() for l in lines]

# You can choose the appropriate axis for the legend. Here, ax1 is used
ax1.legend(lines, labels, loc="upper center")

plt.show()

# f_1 = open(outputs, "w+")
# record='FI(t), Price(t),C_1(t), C_0(t)\n'
# f_1.write(record)
# f_1.close()
###general FI calculation########################
c_1,c_0,c_0_2, FI_x, FI_x_2 = 0, 0, 0, 0, 0
for x in range(len(wo_u)):
        c_1 += price[x] * w_u[x]
        c_0 += price[x] * wo_u[x]
        c_0_2 += price[x] * wo_u_2[x]
if (c_0 != 0):
    c_1_thermal =  c_1
    c_0_thermal =  c_0
    FI_x = 1 - c_1_thermal / c_0_thermal

    c_0_thermal_2 = c_0_2
    FI_x_2=1 - c_1_thermal / c_0_thermal_2
else:
    FI_x = 0
print(FI_x)
print(FI_x_2)



###new FI calculation########################
c_our, c_ben_rl, c_ben_ru=0,0,0
penality=10
failed_times_our,failed_times_rl,failed_times_rule=0,0,0
for x in range(len(wo_u)):

    if (  low_boundary[x] <indoor_our_approach[x]< up_boundary [x]):
        c_our+= price[x] * w_u[x]
    else:
        c_our += penality*(price[x] * w_u[x])
    if (  low_boundary[x] < indoor_benchmark_rl[x] < up_boundary [x]):
        c_ben_rl += price[x] * wo_u[x]
    else:
        c_ben_rl +=  penality*(price[x] * wo_u[x])
    if ( low_boundary[x] < indoor_benchmark_rule[x]< up_boundary[x]):
            c_ben_ru += price[x] * wo_u_2[x]
    else:
        c_ben_ru += penality*(price[x] * wo_u_2[x])


FI_x = 1 - (c_our)/(c_ben_rl)
print(FI_x)

FI_x = 1 - (c_our)/(c_ben_ru)
print(FI_x)