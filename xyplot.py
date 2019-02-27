import pandas as pd
import numpy as np
from io import StringIO
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression 
frame=pd.read_csv("clothdata.csv")

x1=[]
y1=[]
 
for i in frame['Year']:
    x1.append(int(i))
for i in frame['Residential & commercial (tonnes)']:
    y1.append(float(i))
x2=np.array(x1)
y2=np.array(y1)
N=100
random_x = np.random.randn(N)
random_y = np.random.randn(N)

# Create a trace
trace = go.Scatter(
    x = x2,
    y = y2,
    mode = 'markers+lines'
)

data = [trace]

# Plot and embed in ipython notebook!


# plot
data=[trace]
plot(data)




    

