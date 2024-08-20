import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# max_price = 0.0888 + 0.002  # 0.1    #  max=0.0888
# min_price = 0.0444 - 0.015  # 0.0332 #  min=0.0444
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

#new reward_SAC
file="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Best_air_peak_heat_day_16th-May-17-04_3_smaller_co.csv"
indoor_our_approach=np.array(pd.read_csv(file)['indoor_temperature'])[head: end].astype(float)
all_conumsption_our_approach=np.array(pd.read_csv(file)['all_consumption'])[head: end].astype(float)
thermal_kpi_our_list=np.array(pd.read_csv(file)['themal_kpi'])[head: end].astype(float)
thermal_kpi_our_approach=float(thermal_kpi_our_list[-1]) - float(thermal_kpi_our_list[0])
# energy_kpi_our_approach=np.array(pd.read_csv(file)['energy_kpi_our'])
# cost_kpi_our_approach=np.array(pd.read_csv(file)['cost_kpi'])
# print('thermal_kpi_our_approach= ',thermal_kpi_our_approach)
# print('energy_kpi_our_approach =',energy_kpi_our_approach)
# print('cost_kpi_our_approach =',cost_kpi_our_approach)


# outdoor=np.array(pd.read_csv(file)['outdoor_air'])
# ############################################################################################################################
# #rl benchmark   1
# benchmark_rl=file+"benchmark_rl_1.csv"
# indoor_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['themal_kpi'])[head]
# energy_kpi_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['energy_kpi'])[head]
# cost_kpi_benchmark_rl=np.array(pd.read_csv(benchmark_rl)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl= ',thermal_kpi_benchmark_rl)
# print('energy_kpi_benchmark_rl =',energy_kpi_benchmark_rl)
# print('cost_kpi_benchmarl_rl =',cost_kpi_benchmark_rl)
# ############################################################################################################################
#
# #rl benchmark  2
# benchmark_rl_2=file+"benchmark_rl_2.csv"
# indoor_benchmark_rl_2=np.array(pd.read_csv(benchmark_rl_2)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl_2=np.array(pd.read_csv(benchmark_rl_2)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl_2=np.array(pd.read_csv(benchmark_rl_2)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_2)['themal_kpi'])[head]
# energy_kpi_benchmark_rl_2=np.array(pd.read_csv(benchmark_rl_2)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_2)['energy_kpi'])[head]
# cost_kpi_benchmark_rl_2=np.array(pd.read_csv(benchmark_rl_2)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_2)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl_2= ',thermal_kpi_benchmark_rl_2)
# print('energy_kpi_benchmark_rl_2 =',energy_kpi_benchmark_rl_2)
# print('cost_kpi_benchmarl_rl_2 =',cost_kpi_benchmark_rl_2)
# ############################################################################################################################
#
# #rl benchmark   3
# benchmark_rl_3=file+"benchmark_rl_3_2.csv"
# indoor_benchmark_rl_3=np.array(pd.read_csv(benchmark_rl_3)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl_3=np.array(pd.read_csv(benchmark_rl_3)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl_3=np.array(pd.read_csv(benchmark_rl_3)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_3)['themal_kpi'])[head]
# energy_kpi_benchmark_rl_3=np.array(pd.read_csv(benchmark_rl_3)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_3)['energy_kpi'])[head]
# cost_kpi_benchmark_rl_3=np.array(pd.read_csv(benchmark_rl_3)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_3)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl_3= ',thermal_kpi_benchmark_rl_3)
# print('energy_kpi_benchmark_rl_3 =',energy_kpi_benchmark_rl_3)
# print('cost_kpi_benchmarl_rl_3 =',cost_kpi_benchmark_rl_3)
# ############################################################################################################################
# ###########################################################################################################################
# #rl benchmark   4
# benchmark_rl_4=file+"benchmark_rl_4.csv"
# indoor_benchmark_rl_4=np.array(pd.read_csv(benchmark_rl_4)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl_4=np.array(pd.read_csv(benchmark_rl_4)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl_4=np.array(pd.read_csv(benchmark_rl_4)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_4)['themal_kpi'])[head]
# energy_kpi_benchmark_rl_4=np.array(pd.read_csv(benchmark_rl_4)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_4)['energy_kpi'])[head]
# cost_kpi_benchmark_rl_4=np.array(pd.read_csv(benchmark_rl_4)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_4)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl= ',thermal_kpi_benchmark_rl_4)
# print('energy_kpi_benchmark_rl =',energy_kpi_benchmark_rl_4)
# print('cost_kpi_benchmarl_rl =',cost_kpi_benchmark_rl_4)
# ############################################################################################################################
#
# #rl benchmark  5
# benchmark_rl_5=file+"benchmark_rl_5.csv"
# indoor_benchmark_rl_5=np.array(pd.read_csv(benchmark_rl_5)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl_5=np.array(pd.read_csv(benchmark_rl_5)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl_5=np.array(pd.read_csv(benchmark_rl_5)[' themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_5)[' themal_kpi'])[head]
# energy_kpi_benchmark_rl_5=np.array(pd.read_csv(benchmark_rl_5)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_5)['energy_kpi'])[head]
# cost_kpi_benchmark_rl_5=np.array(pd.read_csv(benchmark_rl_5)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_5)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl_5= ',thermal_kpi_benchmark_rl_5)
# print('energy_kpi_benchmark_rl_5 =',energy_kpi_benchmark_rl_5)
# print('cost_kpi_benchmarl_rl_5 =',cost_kpi_benchmark_rl_5)
# ############################################################################################################################
#
# #rl benchmark   6
# benchmark_rl_6=file+"benchmark_rl_6.csv"
# indoor_benchmark_rl_6=np.array(pd.read_csv(benchmark_rl_6)['indoor_temp'])[head: end]
# all_conumsption_benchmark_rl_6=np.array(pd.read_csv(benchmark_rl_6)['all_consumption'])[head: end]
# thermal_kpi_benchmark_rl_6=np.array(pd.read_csv(benchmark_rl_6)['themal_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_6)['themal_kpi'])[head]
# energy_kpi_benchmark_rl_6=np.array(pd.read_csv(benchmark_rl_6)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_6)['energy_kpi'])[head]
# cost_kpi_benchmark_rl_6=np.array(pd.read_csv(benchmark_rl_6)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rl_6)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rl_6= ',thermal_kpi_benchmark_rl_6)
# print('energy_kpi_benchmark_rl_6 =',energy_kpi_benchmark_rl_6)
# print('cost_kpi_benchmarl_rl_6=',cost_kpi_benchmark_rl_6)
# ############################################################################################################################


#rule benchmark


file_rule="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Second_work\\rule_based_benchmark.csv"
indoor_benchmark_rule=np.array(pd.read_csv(file_rule)['indoor_temp'])
all_conumsption_benchmark_rule=np.array(pd.read_csv(file_rule)['all_consumption'])
thermal_kpi_benchmark_rule=np.array(pd.read_csv(file_rule)['themal_kpi'])
price=np.array(pd.read_csv(file_rule)['price'])
up_boundary=np.array(pd.read_csv(file_rule)['boundary_1'])
low_boundary=np.array(pd.read_csv(file_rule)['boundary_0'])
# energy_kpi_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['energy_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rule)['energy_kpi'])[head]
# cost_kpi_benchmark_rule=np.array(pd.read_csv(benchmark_rule)['cost_kpi'])[end-1]-np.array(pd.read_csv(benchmark_rule)['cost_kpi'])[head]
# print('thermal_kpi_benchmark_rule= ',thermal_kpi_benchmark_rule)
# print('energy_kpi_benchmark_rule =',energy_kpi_benchmark_rule)
# print('cost_kpi_benchmarl_rule =',cost_kpi_benchmark_rule)
############################################################################################################################


# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
#
# fig = plt.figure(constrained_layout=True, figsize=(10, 15))
#
# # Create a grid layout
# gs = gridspec.GridSpec(4, 3, figure=fig)
#
# # Specify the subplot placements
# ax1 = fig.add_subplot(gs[0:2, :]) # ax1 takes the entire first row
# ax2 = fig.add_subplot(gs[2, :]) # ax2 takes the entire second row
# ax3 = fig.add_subplot(gs[3, 0]) # ax3 takes the first column of the third row
# ax4 = fig.add_subplot(gs[3, 1]) # ax4 takes the second column of the third row
# ax5 = fig.add_subplot(gs[3, 2]) # ax5 takes the third column of the third row
#
# def add_value_labels(ax, spacing=5, buffer=0.1):
#     # Find the maximum y_value to adjust the ylim later
#     max_height = max([rect.get_height() for rect in ax.patches])
#     for rect in ax.patches:
#         y_value = rect.get_height()
#         x_value = rect.get_x() + rect.get_width() / 2
#         label = "{:.3f}".format(y_value)
#         ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
#                     textcoords="offset points", ha='center', va='bottom')
#     # Adjust the ylim to give some space for the labels
#     ax.set_ylim(0, max_height * (1 + buffer))
#
#
# # First plot - indoor benchmarks
# ax1.plot(indoor_our_approach, 'red', label='our method')
# ax1.plot(indoor_benchmark_rl, 'blue', label='b:rl-1')
# ax1.plot(indoor_benchmark_rl_2, 'dimgrey', label='b:rl-2')
# ax1.plot(indoor_benchmark_rl_3, 'silver', label='b:rl-3')
# ax1.plot(indoor_benchmark_rl_4, 'm', label='b:rl-4')
# ax1.plot(indoor_benchmark_rl_5, 'c', label='b:rl-5')
# ax1.plot(indoor_benchmark_rl_6, 'orange', label='b:rl-6')
# ax1.plot(indoor_benchmark_rule, 'yellow', label='b:rule-based')
# ax1.plot(up_boundary, 'g:', label='Upper boundary')
# ax1.plot(low_boundary, 'g:', label='Lower boundary')
# ax1.set_xlabel('Time step')
# ax1.set_ylabel('Temperature °C', color='k')
# ax1.legend(loc="lower right")
#
# # Second plot - outdoor
# ax2.plot(outdoor, 'navy', label='outdoor air temp')
# ax2.set_xlabel('Time step')
# ax2.set_ylabel('Temperature °C', color='k')
# ax2.legend(loc="lower right")
#
# # Third plot - Thermal discomfort KPI
# name = ['Our approach', 'b:rl-1', 'b:rl-2', 'b:rl-3', 'b:rl-4', 'b:rl-5', 'b:rl-6','b:rule']
# discomfort_kpis = [thermal_kpi_our_approach, thermal_kpi_benchmark_rl, thermal_kpi_benchmark_rl_2, thermal_kpi_benchmark_rl_3,  thermal_kpi_benchmark_rl_4, thermal_kpi_benchmark_rl_5,thermal_kpi_benchmark_rl_6, thermal_kpi_benchmark_rule]
# bar_c = ['red', 'blue', 'dimgrey', 'silver','m','c','orange','yellow']
#
# ax3.bar(name, discomfort_kpis, color=bar_c)
# ax3.set_ylabel('Thermal discomfort KPI')
# ax3.set_title('Thermal discomfort KPI')
# add_value_labels(ax3)
#
# # Fourth plot - Energy usage KPI
# energy_kpis = [energy_kpi_our_approach, energy_kpi_benchmark_rl, energy_kpi_benchmark_rl_2, energy_kpi_benchmark_rl_3,energy_kpi_benchmark_rl_4, energy_kpi_benchmark_rl_5, energy_kpi_benchmark_rl_6,energy_kpi_benchmark_rule]
# ax4.bar(name, energy_kpis, color=bar_c)
# ax4.set_ylabel('Energy usage KPI')
# ax4.set_title('Energy usage KPI')
# add_value_labels(ax4)
#
# # Fifth plot - Cost KPI
# cost_kpis = [cost_kpi_our_approach, cost_kpi_benchmark_rl, cost_kpi_benchmark_rl_2, cost_kpi_benchmark_rl_3, cost_kpi_benchmark_rl_4, cost_kpi_benchmark_rl_5, cost_kpi_benchmark_rl_6,cost_kpi_benchmark_rule]
# ax5.bar(name, cost_kpis, color=bar_c)
# ax5.set_ylabel('Cost KPI')
# ax5.set_title('Cost KPI')
# add_value_labels(ax5)
#
# if(debug_model==0):
#     plt.show()
#
#
# import plotly.graph_objects as go
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# metrics = ['Thermal comfort', 'Energy efficiency', 'Cost efficiency']
#
# data_=[
#           [thermal_kpi_our_approach,	energy_kpi_our_approach,	cost_kpi_our_approach],# our method
#           [thermal_kpi_benchmark_rl,    energy_kpi_benchmark_rl,    cost_kpi_benchmark_rl],#method 1
#           [thermal_kpi_benchmark_rl_2, energy_kpi_benchmark_rl_2, cost_kpi_benchmark_rl_2],#method 2
#           [thermal_kpi_benchmark_rl_3, energy_kpi_benchmark_rl_3, cost_kpi_benchmark_rl_3],  # method 3
#           [thermal_kpi_benchmark_rl_4, energy_kpi_benchmark_rl_4, cost_kpi_benchmark_rl_4],  # method 4
#           [thermal_kpi_benchmark_rl_5, energy_kpi_benchmark_rl_5, cost_kpi_benchmark_rl_5],  # method 5
#           [thermal_kpi_benchmark_rl_6, energy_kpi_benchmark_rl_6, cost_kpi_benchmark_rl_6],# method 6
#           [thermal_kpi_benchmark_rule, energy_kpi_benchmark_rule, cost_kpi_benchmark_rule],# method 7
#      ]
#
# data_=np.array(data_)
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset_=np.zeros([8,3])  #row: how many methods, col: how many indexes
# dataset=[]
# for i in range (0, 3): # 6--> index
#
#   dataset= scaler.fit_transform(data_[:,i].reshape(-1, 1))
#   dataset=np.reshape(dataset, [len(dataset)])
#   dataset_[:,i]=np.array(dataset)
#   # if(i==0 or i==1):
#   dataset_[:, i]= 1-dataset_[:,i]
#   #print (i)
# fig = go.Figure()
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[0],
#       marker=dict(color='red', ),
#       theta=metrics,
#       name=name[0]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[1],
#       marker=dict(color='blue', ),
#       theta=metrics,
#       name=name[1]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[2],
#       marker=dict(color='grey', ),
#       theta=metrics,
#       name=name[2]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[3],
#       marker=dict(color='silver', ),
#       theta=metrics,
#       name=name[3]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[4],
#       marker=dict(color='mediumpurple' ),
#       theta=metrics,
#       name=name[4]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[5],
#       marker=dict(color='cyan', ),
#       theta=metrics,
#       name=name[5]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[6],
#       marker=dict(color='orange', ),
#       theta=metrics,
#       name=name[6]),
# )
# fig.add_trace(go.Scatterpolar(
#       r= dataset_[7],
#       marker=dict(color='yellow', ),
#       theta=metrics,
#       name=name[7]),
# )
#
#
# fig.update_traces(fill='toself')
# fig.update_layout(
#       #title="Classification performance",
#       polar=dict( radialaxis=dict(visible=True,  range=[0, 1] )),
#       font=dict(
#       family="Arial, monospace",
#       size=44,
#       color="Black"),
#       showlegend=True
# )
#
# if (debug_model == 0):
#     fig.show()
#
#
#
#
#
#


c_0,c_1,c_2,c_3,c_4, c_5,c_6, c_our=0,0,0,0,0,0,0,0
outputs="C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\Comparsion-March o4th\\Flexibility_.csv"

# f_1 = open(outputs, "w+")
# record='FI(t), Price(t),C_1(t), C_0(t)\n'
# f_1.write(record)
# f_1.close()
def sig(x):
 return (1 - np.exp(-x))/(1 + np.exp(-x))

def thermal(t,coo,hea):
    result= max((max((t-coo),0)+max((hea-t),0)),1)
    return result

c_our,c_0=0,0
for x in range(len(all_conumsption_benchmark_rule)-1):
    c_our += price[x] * all_conumsption_our_approach[x]
    # print('c_our = ',c_our)
    c_0   += price[x] * all_conumsption_benchmark_rule[x]
    # c_1 += price[x] * all_conumsption_benchmark_rl[x]
    # c_2 += price[x] * all_conumsption_benchmark_rl_2[x]
    # c_3 += price[x] * all_conumsption_benchmark_rl_3[x]
    # c_4 += price[x] * all_conumsption_benchmark_rl_4[x]
    # c_5 += price[x] * all_conumsption_benchmark_rl_5[x]
    # c_6 += price[x] * all_conumsption_benchmark_rl_6[x]
m,n=c_our,c_0
FI_our =1-c_our/c_0
# FI_1 =1-c_1/c_0
# FI_2 =1-c_2/c_0
# FI_3 =1-c_3/c_0
# FI_4 =1-c_4/c_0
# FI_5 =1-c_5/c_0
# FI_6 =1-c_6/c_0
# print(FI_our, FI_1, FI_2,FI_3,FI_4,FI_5,FI_6)
print(FI_our)

c_0_NEW,c_1_NEW,c_2_NEW,c_3_NEW,c_4_NEW, c_5_NEW,c_6_NEW, c_our_NEW=0,0,0,0,0,0,0,0
for x in range(len(all_conumsption_benchmark_rule)-1):
    c_our_NEW += price[x] * all_conumsption_our_approach[x] *  thermal(indoor_our_approach[x], up_boundary[x], low_boundary[x])
    c_0_NEW   += price[x] * all_conumsption_benchmark_rule[x] * thermal(indoor_benchmark_rule[x],up_boundary[x],low_boundary[x] )
    # c_1_NEW += price[x] * all_conumsption_benchmark_rl[x] * thermal(indoor_benchmark_rl[x],up_boundary[x],low_boundary[x] )
    # c_2_NEW += price[x] * all_conumsption_benchmark_rl_2[x] * thermal(indoor_benchmark_rl_2[x],up_boundary[x],low_boundary[x] )
    # c_3_NEW += price[x] * all_conumsption_benchmark_rl_3[x] * thermal(indoor_benchmark_rl_3[x],up_boundary[x],low_boundary[x] )
    # c_4_NEW += price[x] * all_conumsption_benchmark_rl_4[x] * thermal(indoor_benchmark_rl_4[x],up_boundary[x],low_boundary[x] )
    # c_5_NEW += price[x] * all_conumsption_benchmark_rl_5[x] * thermal(indoor_benchmark_rl_5[x],up_boundary[x],low_boundary[x] )
    # c_6_NEW += price[x] * all_conumsption_benchmark_rl_6[x] * thermal(indoor_benchmark_rl_6[x],up_boundary[x],low_boundary[x] )
a=c_our_NEW
b=c_0_NEW
FI_our_NEW =1-c_our_NEW/c_0_NEW
# FI_1_NEW =1-c_1_NEW/c_0_NEW
# FI_2_NEW =1-c_2_NEW/c_0_NEW
# FI_3_NEW =1-c_3_NEW/c_0_NEW
# FI_4_NEW =1-c_4_NEW/c_0_NEW
# FI_5_NEW =1-c_5_NEW/c_0_NEW
# FI_6_NEW =1-c_6_NEW/c_0_NEW
# print(FI_our_NEW, FI_1_NEW, FI_2_NEW,FI_3_NEW,FI_4_NEW,FI_5_NEW,FI_6_NEW)
print(FI_our_NEW)


#
# FIs = [FI_our, FI_1, FI_2, FI_3, FI_4, FI_5, FI_6]
# FIs_new = [FI_our_NEW, FI_1_NEW, FI_2_NEW, FI_3_NEW, FI_4_NEW, FI_5_NEW, FI_6_NEW]
# methods = ['Our approach', 'b:rl-1', 'b:rl-2', 'b:rl-3', 'b:rl-4', 'b:rl-5', 'b:rl-6']
# bar_color = ['red', 'blue', 'dimgrey', 'silver', 'm', 'c', 'orange']
#
# # Function to add value labels above each bar
# def add_value_labels(ax):
#     for rect in ax.patches:
#         height = rect.get_height()
#         ax.annotate(f'{height:.3f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 5),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# # Creating the bar chart
# plt.figure(figsize=(10, 6))
# ax = plt.gca()  # Get current axis
# ax.bar(methods, FIs, color=bar_color)
# ax.set_ylabel('FI')
# ax.set_title('Original Flexibility Index')
# add_value_labels(ax)
#
# plt.xticks(rotation=45)  # Rotate method names for better readability
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()
#
#
# plt.figure(figsize=(10, 6))
# ax = plt.gca()  # Get current axis
# ax.bar(methods, FIs_new, color=bar_color)
# ax.set_ylabel('FI')
# ax.set_title('New Flexibility Index')
# add_value_labels(ax)
#
# plt.xticks(rotation=45)  # Rotate method names for better readability
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()

# fig = plt.figure(constrained_layout=True, figsize=(10, 15))
# gs = gridspec.GridSpec(2, 1, figure=fig)
# ax1 = fig.add_subplot(gs[0, 0]) # ax1 takes the entire first row
# ax2 = fig.add_subplot(gs[1, 0]) # ax2 takes the entire second row
#
# def addlabels(x,y):
#     for i in range(len(x)):
#         plt.text(i,y[i],y[i])
#
# ax1.bar(methods, FIs, color=bar_color)
# addlabels (methods, FIs)
# ax1.set_ylabel('Orginal FI')
# ax1.set_title('Orginal Flexibility Index(FI)')
# # add_value_labels(ax1)
#
#
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
#
# # Second plot - outdoor
# ax2.bar(methods, FIs_new, color=bar_color)
# ax2.set_ylabel('New FI')
# ax2.set_title('New FI')
# addlabels (methods, FIs_new)
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()

