import os
import requests
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

scaler = MinMaxScaler(feature_range=(0, 1))
result_location = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\critical_point_exploration_50_typical_heat_day_.csv"
data = pd.read_csv(result_location)
critical_points=[]
for x in range (1, len(data)):
    if(data['action'][x]==1) and (data['action'][x-1]==0):
        critical_points.append([data['indoor'][x-1],data['outdoor'][x-1],data['wind_speed'][x-1]])
# critical_points=
critical_points = pd.DataFrame(critical_points, columns=['indoor','outdoor','wind_speed'])
# critical_points['outdoor'] = (critical_points['outdoor'] - critical_points['outdoor'].min()) / (critical_points['outdoor'].max() - critical_points['outdoor'].min())
# critical_points['wind_speed'] = (critical_points['wind_speed'] - critical_points['wind_speed'].min()) / (critical_points['wind_speed'].max() - critical_points['wind_speed'].min())
Scores=[]

########sc#############
target=np.array(critical_points['wind_speed']).reshape(-1, 1)
for k in range(2, 9):
    estimator = KMeans(n_clusters=k)
    estimator.fit(target)
    Scores.append(silhouette_score(target, estimator.labels_, metric='euclidean'))
print(Scores)
a = Scores.index(max(Scores))+2
########sc#############
estimator = KMeans(n_clusters=a)
clustering_results= pd.Series(estimator.fit(target).labels_)
    # model = KMeans(n_clusters=int(a))
    # model.fit(data)
    # all_predictions = model.predict(data)
    # print(all_predictions)
    # for i in range(len(all_predictions)):
    #     f.write(str(all_predictions[i]) + '\n')
    # f.close()
critical_points['clusterings']=clustering_results

results = "C:\\Users\\wan397\\OneDrive - CSIRO\\Desktop\\RL_WORK_SUMMARY\\critical_point_finding_.csv"
critical_points.to_csv(results)
print(1)