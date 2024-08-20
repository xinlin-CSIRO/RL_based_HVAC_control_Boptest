import os
import requests
import numpy as np
import random
import pandas as pd

target = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\step_no_control_50_typical_heat_day_.csv"
data=pd.read_csv(target)
indoor_air_temp=data['return']
indoor_temp_1st=np.diff(np.array(indoor_air_temp))
outdoor_air_temp=data['outside']
outdoor_temp_1st=np.diff(np.array(outdoor_air_temp))
outdoor_RH=data['RH']
outdoor_RH_1st=np.diff(np.array(outdoor_RH))
wind_speed=data['wend_speed']
wind_speed_1st=np.diff(np.array(wind_speed))
#
# radiation_1st=np.diff(np.array(radiation))

a=np.corrcoef(indoor_temp_1st,outdoor_temp_1st)[0,1]
a1=np.corrcoef(indoor_air_temp,outdoor_air_temp)[0,1]

b=np.corrcoef(indoor_temp_1st,outdoor_RH_1st)[0,1]
b1=np.corrcoef(indoor_air_temp,outdoor_RH)[0,1]

c=np.corrcoef(indoor_temp_1st,wind_speed_1st)[0,1]
c1=np.corrcoef(indoor_air_temp,wind_speed)[0,1]

# d=np.corrcoef(indoor_temp_1st,radiation_1st)[0,1]
# d1=np.corrcoef(indoor_air_temp,radiation)[0,1]

e=np.corrcoef(outdoor_air_temp,wind_speed)[0,1]
f=np.corrcoef(outdoor_air_temp,outdoor_RH)[0,1]

daytime_indoor_temp, daytime_outdoor_temp, daytime_wind_speed, daytime_radition =[],[],[], []
radiation=data['radiation']
for x in range(0, len(radiation)):
  if(radiation[x] !=0):
      daytime_indoor_temp.append(indoor_air_temp[x])
      daytime_outdoor_temp.append(outdoor_air_temp[x])
      daytime_wind_speed.append(wind_speed[x])
      daytime_radition.append(radiation[x])
daytime_indoor_temp= np.array(daytime_indoor_temp)
daytime_outdoor_temp=np.array(daytime_outdoor_temp)
daytime_wind_speed=np.array(daytime_wind_speed)
daytime_radition=np.array(daytime_radition)

x3= np.corrcoef(daytime_indoor_temp,daytime_radition)[0,1]
x1= np.corrcoef(daytime_outdoor_temp,daytime_radition)[0,1]
x2= np.corrcoef(outdoor_air_temp,outdoor_RH)[0,1]

print(1)

# g=np.corrcoef(outdoor_air_temp,radiation)[0,1]
record='return,heat,outside,radiation,RH,wend_speed\n'