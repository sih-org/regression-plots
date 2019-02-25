import pandas as pd
import numpy as np
from io import StringIO
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import numpy
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression 
frame=pd.read_csv("xysample.csv",dtype={0:'int',1:'int'})

x1=[]
y1=[]

def estimate_coef(x, y): 
    # number of observations/points 
    n = np.size(x) 
  
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
  
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 
  
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x 
  
    return(b_0, b_1) 
for i in frame['x']:
    x1.append(i)
for i in frame['y']:
    y1.append(i)
x2=np.array(x1)
y2=np.array(y1)
b = estimate_coef(x2, y2) 
print(b[0],b[1])


# plot
trace = go.catter(
    x=x1,
    y=y1,
    mode='markers')
data=[trace]
plot(data)




    

