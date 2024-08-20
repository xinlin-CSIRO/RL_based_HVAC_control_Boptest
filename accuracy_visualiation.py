import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
metrics = ['Thermal comfort', 'Energy efficiency', 'Cost efficiency']
name=[
'our method',
'benchmark_rl',
'benchmark_rule',
]

data_=[
          [0.4638540480713352,	4.824478123774525,	0.0539367796590688],#method 1
          [0.351720324881607,    5.957372099064368,    0.0641434829590199],#method 2
          [207.16414959756204, 3.621955786698536, 0.052512127810284796],#method 3
          ]

data_=np.array(data_)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_=np.zeros([3,3])  #row: how many methods, col: how many indexes
dataset=[]
for i in range (0, 3): # 6--> index

  dataset= scaler.fit_transform(data_[:,i].reshape(-1, 1))
  dataset=np.reshape(dataset, [len(dataset)])
  dataset_[:,i]=np.array(dataset)
  # if(i==0 or i==1):
  dataset_[:, i]= 1-dataset_[:,i]
  #print (i)

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
      r= dataset_[0],
      marker=dict(color='red', ),
      theta=metrics,
      name=name[0]),
)
fig.add_trace(go.Scatterpolar(
      r= dataset_[1],
      marker=dict(color='  blue', ),
      theta=metrics,
      name=name[1]),
)
fig.add_trace(go.Scatterpolar(
      r= dataset_[2],
      marker=dict(color=' yellowgreen', ),
      theta=metrics,
      name=name[2]),
)




fig.update_traces(fill='toself')
fig.update_layout(
      #title="Classification performance",
      polar=dict( radialaxis=dict(visible=True,  range=[0, 1] )),
      font=dict(
      family="Arial, monospace",
      size=44,
      color="Black") ,
      showlegend=True
)
#fig.update_layout(height=500, width=500)
# Path_sorce_="C:\\Users\\Xinlin\\Desktop\\3rd_predictior\\prediction_results.jpeg"
fig.show()
# fig.write_image(Path_sorce_)