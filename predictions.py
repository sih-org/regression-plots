import operator
import numpy as np 
#import matplotlib.pyplot as plt 
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from operator import itemgetter

np.random.seed(0)
class cloth:
    def predictions(self,data):
        clothdata= pd.read_csv('cloth$data.csv')
        clothdata= pd.read_csv('cloth$data.csv')
        clothdata1= pd.read_csv('cloth$data1.csv')
        self.x0=clothdata['Cloth Waste'][:].values
        self.y1=clothdata['commercial(co2 in tonnes)'][:].values
        self.y2=clothdata['methane_due_to_cloth'][:].values
        self.y3=clothdata['N2o_due_to_cloth'][:].values
        self.y4=clothdata['so2_due_to_cloth'][:].values
        self.x1=clothdata['Year'][:].values
        self.x0 = self.x0[:, np.newaxis]
        self.y1 = self.y1[:, np.newaxis]
        self.y2 = self.y2[:, np.newaxis]
        self.y3 = self.y3[:, np.newaxis]
        self.y4 = self.y4[:, np.newaxis]
        self.x1 = self.x1[:, np.newaxis]

        self.py=[0,0,0,0,0,0,0,0]
        self.y=[0,0,0,0,0,0,0,0]
        k=0
        if(data[0]<=2017):
            self.k=data[0]-2017
            self.py[0]=data[1]
            self.py[1]=data[2]
            self.py[2]=data[3]
            self.py[3]=data[4]
            self.py[4]=0.79*(self.x0[k]+data[5])-0.79*self.x0[k]
            self.py[5]=0.095*(self.x0[k]+data[5])-0.095*self.x0[k]
            self.py[6]=0.095*(self.x0[k]+data[5])-0.095*self.x0[k]
            self.py[7]=0.03*(self.x0[k]+data[5])-0.03*self.x0[k]
            self.y[0]=self.y1[k]
            self.y[1]=self.y2[k]
            self.y[2]=self.y3[k]
            self.y[3]=self.y4[k]
            self.y[4]=0.79*self.x0[k]
            self.y[5]=0.095*self.x0[k]
            self.y[6]=0.095*self.x0[k]
            self.y[7]=0.03*self.x0[k]
            return self.py
        polynomial_features= PolynomialFeatures(degree=1)
        x1_poly = polynomial_features.fit_transform(self.x1)
        model0=LinearRegression()       
        model0.fit(x1_poly, self.x0)
        x0p=model0.predict(polynomial_features.fit_transform([[data[0]]]))
        model1 = LinearRegression()
        model2 = LinearRegression()
        model3 = LinearRegression()
        model4 = LinearRegression()
        x0_poly = polynomial_features.fit_transform(self.x0)        
        model1.fit(x0_poly, self.y1)
        model2.fit(x0_poly, self.y2)
        model3.fit(x0_poly, self.y3)
        model4.fit(x0_poly, self.y4)
        self.py[0]=data[1] 
        self.py[1]=data[2]  
        self.py[2]=data[3]  
        self.py[3]=data[4]
        i=model1.predict(polynomial_features.fit_transform(x0p)) 
        j=model2.predict(polynomial_features.fit_transform(x0p))  
        k=model3.predict(polynomial_features.fit_transform(x0p))  
        l=model4.predict(polynomial_features.fit_transform(x0p))
        self.y[0]=i[0][0]
        self.y[1]=j[0][0]
        self.y[2]=k[0][0]
        self.y[3]=l[0][0]

        self.py[4]=0.79*(x0p[0][0]+data[5])-0.79*x0p[0][0]
        self.py[5]=0.095*(x0p[0][0]+data[5])-0.095*x0p[0][0]
        self.py[6]=0.095*(x0p[0][0]+data[5])-0.095*x0p[0][0]
        self.py[7]=0.03*(x0p[0][0]+data[5])-0.03*x0p[0][0]
        self.y[7]=0.03*x0p[0][0]
        self.y[4]=0.79*x0p[0][0]
        self.y[5]=0.095*x0p[0][0]
        self.y[6]=0.095*x0p[0][0]
        return self.py
    def visualize(self):
        print(self.y,self.py)
        trace1 = go.Bar(
            x=['co2', 'methane', 'so2','n2o','Recycled','Incinerated','Discarded','ocean'],
            y=self.y,
            name='actual in M tons'
        )
        trace2 = go.Bar(
            x=['co2', 'methane', 'so2','n2o','Recycled','Incinerated','Discarded','ocean'],
            y=self.py,
            name='impact unit in kgs'
        )

        data = [trace1, trace2]
        layout = go.Layout(
            barmode='stack'
        )

        fig = go.Figure(data=data, layout=layout)
        plot(fig,filename='cloth.html')
def Cloth_Preprocessing(arr):
    b=[]
    if(arr[0][2]=='Kilograms'):
        a=5
        arr[0][2]='Tonnes'
    if(arr[1][1]=='Cotton'):
        b.append([arr[0][0],arr[0][1]*1.778,arr[0][1]*0.578,arr[0][1]*0.55,arr[0][1]*1.38,arr[0][1]])
    if(arr[1][1]=='Polyster'):
        b.append([arr[0][0],arr[0][1]*2.1193,arr[0][1]*0.778,arr[0][1]*0.42,arr[0][1]*1.78,arr[0][1]])
    if(arr[1][1]=='Nylon' or arr[1][1]=='Acroylic'):
        b.append([arr[0][0],arr[0][1]*0.38,arr[0][1]*0.35,arr[0][1]*0.19,arr[0][1]*0.28,arr[0][1]])
    return b[0]
k=[[2018,200,'Kilograms'],['cloth','Cotton']]
a=Cloth_Preprocessing(k)
c=cloth()
k=c.predictions(a)
print(k)
c.visualize()

