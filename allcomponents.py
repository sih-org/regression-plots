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
        x0p=model0.predict(x1_poly)
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
        self.y[0]=model1.predict(polynomial_features.fit_transform(x0p)) 
        self.y[1]=model2.predict(polynomial_features.fit_transform(x0p))  
        self.y[2]=model3.predict(polynomial_features.fit_transform(x0p))  
        self.y[3]=model4.predict(polynomial_features.fit_transform(x0p))
        self.py[4]=0.79*(x0p+data[5])-0.79*x0p
        self.py[5]=0.095*(x0p+data[5])-0.095*x0p
        self.py[6]=0.095*(x0p+data[5])-0.095*x0p
        
        self.y[4]=0.79*x0p
        self.y[5]=0.095*x0p
        self.y[6]=0.095*x0p
        return self.py
def Cloth_Preprocessing(arr):
    b=[]
    if(arr[0][2]=='Kilograms'):
        arr[0][1]=arr[0][1]/1000
        arr[0][2]='Tonnes'
    if(arr[1][1]=='Cotton'):
        b.append([arr[0][0],arr[0][1]*1.778,arr[0][1]*0.578,arr[0][1]*0.55,arr[0][1]*1.38,arr[0][1]])
    if(arr[1][1]=='Polyster'):
        b.append([arr[0][0],arr[0][1]*2.1193,arr[0][1]*0.778,arr[0][1]*0.42,arr[0][1]*1.78,arr[0][1]])
    if(arr[1][1]=='Nylon' or arr[1][1]=='Acroylic'):
        b.append([arr[0][0],arr[0][1]*0.38,arr[0][1]*0.35,arr[0][1]*0.19,arr[0][1]*0.28,arr[0][1]])
    return b[0]
k=[[2015,200,'Kilograms'],['cloth','Cotton']]
a=Cloth_Preprocessing(k)
c=cloth()
k=c.predictions(a)
print(k)      

        
        
"""        polynomial_features= PolynomialFeatures(degree=1)
        x0_poly = polynomial_features.fit_transform(self.x0)
        predx=[]
        model1 = LinearRegression()
        model2 = LinearRegression()
        model3 = LinearRegression()
        model4 = LinearRegression()
        model5 = LinearRegression()
        model6 = LinearRegression()
        model7 = LinearRegression()
        model1.fit(x0_poly, self.y1)
        model2.fit(x0_poly, self.y2)
        model3.fit(x0_poly, self.y3)
        model4.fit(x0_poly, self.y4)
        model5.fit(x0_poly, self.y5)
        model6.fit(x0_poly, self.y6)
        model7.fit(x0_poly, self.y7)
        diffsum=[0,0,0,0,0,0,0]
        y1p=model1.predict(x0_poly)
        y2p=model2.predict(x0_poly)
        y3p=model3.predict(x0_poly)
        y4p=model4.predict(x0_poly)
        y5p=model5.predict(x0_poly)
        y6p=model6.predict(x0_poly)
        y7p=model7.predict(x0_poly)
        error=[abs(y1p-self.y1),abs(y2p-self.y2),abs(y3p-self.y3),abs(y4p-self.y4),abs(y5p-self.y5),abs(y6p-self.y6),abs(y7p-self.y7)]
#       j =datas[len(datas)-1][0]
        for data in datas:
            k=int(data[0])-1990
            dump=data[1]+clothdata['Cloth Waste'][k]
            predx=[[dump]]
            self.py1[k]=model1.predict(polynomial_features.fit_transform(predx))
            self.py2[k]=model2.predict(polynomial_features.fit_transform(predx))
            self.py3[k]=model3.predict(polynomial_features.fit_transform(predx))
            self.py4[k]=model4.predict(polynomial_features.fit_transform(predx))
            self.py5[k]=model5.predict(polynomial_features.fit_transform(predx))
            self.py6[k]=model6.predict(polynomial_features.fit_transform(predx))
            self.py7[k]=model7.predict(polynomial_features.fit_transform(predx))
            diffsum[0]=diffsum[0]+self.py1[k]-self.y1[k]+error[0][k]
            diffsum[1]=diffsum[1]+self.py2[k]-self.y2[k]+error[1][k]
            diffsum[2]=diffsum[2]+self.py3[k]-self.y3[k]+error[2][k]
            diffsum[3]=diffsum[3]+self.py4[k]-self.y4[k]+error[3][k]
            diffsum[4]=diffsum[4]+self.py5[k]-self.y5[k]+error[4][k]
            diffsum[5]=diffsum[5]+self.py6[k]-self.y6[k]+error[5][k]
            diffsum[6]=diffsum[6]+self.py7[k]-self.y7[k]+error[6][k]
        return diffsum

    def visualize(self):
        diff=[]
        for i in range(len(self.py)):
            diff.append(self.py[i]-self.y[0])
        trace1 = go.Bar(
            x=['co2', 'methane', 'so2','n2o','Recycled','Incinerated','Discarded'],
            y=self.y,
            name='actual'
        )
        trace2 = go.Bar(
            x=['giraffes', 'orangutans', 'monkeys'],
            y=self.py,
            name='impact'
        )

        data = [trace1, trace2]
        layout = go.Layout(
            barmode='stack'
        )

        fig = go.Figure(data=data, layout=layout)
        plot(fig,filename='cloth.html')
            
"       clothdata= pd.read_csv('cloth$data.csv')
        years=clothdata['Year']
        x1=[]
        for i in years:
            x1.append(int(i))
        y4=[]
        py4=[]
        for i in self.y4:
            y4.append(int(i))
        for i in self.py4:
            py4.append(int(i))
        x1=np.array(x1)
        y4=np.array(y4)
        py4=np.array(py4)
        trace1 = go.Scatter(
            x = x1,
            y = y4,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py4,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Recycled cloth',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Recycle',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='clothrecycle.html',auto_open=False)
        y5=[]
        py5=[]
        for i in self.y5:
            y5.append(int(i))
        for i in self.py5:
            py5.append(int(i))
        y5=np.array(y5)
        py5=np.array(py5)
        trace1 = go.Scatter(
            x = x1,
            y = y5,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py5,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Discarded cloth',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Discarded',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='clothdiscarded.html',auto_open=False)
        y6=[]
        py6=[]
        for i in self.y6:
            y6.append(int(i))
        for i in self.py6:
            py6.append(int(i))
        y6=np.array(y6)
        py6=np.array(py6)
        trace1 = go.Scatter(
            x = x1,
            y = y6,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py6,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Incenerated cloth',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Incenerated',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='clothincenerated.html',auto_open=False)
        y7=[]
        py7=[]
        for i in self.y7:
            y7.append(int(i))
        for i in self.py7:
            py7.append(int(i))
        y7=np.array(y7)
        py7=np.array(py7)
        trace1 = go.Scatter(
            x = x1,
            y = y7,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py7,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'cloth waste in ocean',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Ocean waste',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='clothocean.html',auto_open=False)
"""
def preProcess(arr):
    a=[]
    b=arr[0]
    arr=arr[1:]
    final=[]
    for i in range(0,len(arr)):
       if(arr[i] is not None):
            if(arr[i][1] is not None and arr[i][0] is not None):
                a.append(arr[i])
            else:
                continue
    a=sorted(a, key=itemgetter(0))
    for i in range(0,len(a)):
        if(a[i][2]=='Kilograms'):
            a[i][1]=a[i][1]/1000
            a[i][2]='Tonnes'
        if(i>0):
            a[i][1]+=a[i-1][1]
    
    for i in range(0,len(a)-1):
        if((a[i][0]+1)!=a[i+1][0]):
            final.append(a[i])
            for j in range(1,(a[i+1][0]-a[i][0])):
                final.append([a[i][0]+j,a[i][1],a[i][2]])
        else:
            final.append(a[i])
    final.append(a[len(a)-1])
    return final





class toy:
    def predictions(self,datas):
        tooldata= pd.read_csv('Schools_and_toys.csv')
        tooldata1= pd.read_csv('Schools_and_toys.csv')
        self.x0=tooldata['plastic_waste'][:].values
        self.y1=tooldata['Co2_plastic_material'][:].values
        self.py1=tooldata1['Co2_plastic_material'][:].values
        self.y2=tooldata['methane_plastic'][:].values
        self.py2=tooldata1['methane_plastic'][:].values
        self.y3=tooldata['N2o_plastic'][:].values
        self.py3=tooldata1['N2o_plastic'][:].values
        self.py4=tooldata1['Recycle'][:].values
        self.py5=tooldata1['Discarded'][:].values
        self.py6=tooldata1['Incinerated'][:].values
        self.py7=tooldata1['Plastic_in_ocean'][:].values
        self.y4=tooldata['Recycle'][:].values
        self.y5=tooldata['Discarded'][:].values
        self.y6=tooldata['Incinerated'][:].values
        self.y7=tooldata['Plastic_in_ocean'][:].values
        self.x0 = self.x0[:, np.newaxis]
        self.y1 = self.y1[:, np.newaxis]
        self.y2 = self.y2[:, np.newaxis]
        self.y3 = self.y3[:, np.newaxis]
        self.y4 = self.y4[:, np.newaxis]
        self.y5 = self.y5[:, np.newaxis]
        self.y6 = self.y6[:, np.newaxis]
        self.y7 = self.y7[:, np.newaxis]
        self.py1= self.py1[:,np.newaxis]
        self.py2= self.py2[:,np.newaxis]
        self.py3= self.py3[:,np.newaxis]
        self.py4= self.py4[:,np.newaxis]
        self.py5= self.py5[:,np.newaxis]
        self.py6= self.py6[:,np.newaxis]
        self.py7= self.py7[:,np.newaxis]
        polynomial_features= PolynomialFeatures(degree=2)
        x0_poly = polynomial_features.fit_transform(self.x0)
        predx=[]
        model1 = LinearRegression()
        model2 = LinearRegression()
        model3 = LinearRegression()
        model4 = LinearRegression()
        model5 = LinearRegression()
        model6 = LinearRegression()
        model7 = LinearRegression()
        model1.fit(x0_poly, self.y1)
        model2.fit(x0_poly, self.y2)
        model3.fit(x0_poly, self.y3)
        model4.fit(x0_poly, self.y4)
        model5.fit(x0_poly, self.y5)
        model6.fit(x0_poly, self.y6)
        model7.fit(x0_poly, self.y7)
        diffsum=[0,0,0,0,0,0,0]
        y1p=model1.predict(x0_poly)
        y2p=model2.predict(x0_poly)
        y3p=model3.predict(x0_poly)
        y4p=model4.predict(x0_poly)
        y5p=model5.predict(x0_poly)
        y6p=model6.predict(x0_poly)
        y7p=model7.predict(x0_poly)
        error=[abs(y1p-self.y1),abs(y2p-self.y2),abs(y3p-self.y3),abs(y4p-self.y4),abs(y5p-self.y5),abs(y6p-self.y6),abs(y7p-self.y7)]
        j=datas[len(datas)-1][0]
        for data in datas:
            k=data[0]-1990
            dump=data[1]+tooldata['plastic_waste'][k]
            predx=[[dump]]
            self.py1[k]=model1.predict(polynomial_features.fit_transform(predx))
            self.py2[k]=model2.predict(polynomial_features.fit_transform(predx))
            self.py3[k]=model3.predict(polynomial_features.fit_transform(predx))
            self.py4[k]=model4.predict(polynomial_features.fit_transform(predx))
            self.py5[k]=model5.predict(polynomial_features.fit_transform(predx))
            self.py6[k]=model6.predict(polynomial_features.fit_transform(predx))
            self.py7[k]=model7.predict(polynomial_features.fit_transform(predx))
            diffsum[0]=diffsum[0]+self.py1[k]-self.y1[k]+error[0][k]
            diffsum[1]=diffsum[1]+self.py2[k]-self.y2[k]+error[1][k]
            diffsum[2]=diffsum[2]+self.py3[k]-self.y3[k]+error[2][k]
            diffsum[3]=diffsum[3]+self.py4[k]-self.y4[k]+error[3][k]
            diffsum[4]=diffsum[4]+self.py5[k]-self.y5[k]+error[4][k]
            diffsum[5]=diffsum[5]+self.py6[k]-self.y6[k]+error[5][k]
            diffsum[6]=diffsum[6]+self.py7[k]-self.y7[k]+error[6][k]
        return diffsum
    def visualize(self):
        tooldata= pd.read_csv('Schools_and_toys.csv')
        years=tooldata['Year']
        x1=[]
        for i in years:
            x1.append(int(i))
        y4=[]
        py4=[]
        for i in self.y4:
            y4.append(int(i))
        for i in self.py4:
            py4.append(int(i))
        x1=np.array(x1)
        y4=np.array(y4)
        py4=np.array(py4)
        trace1 = go.Scatter(
            x = x1,
            y = y4,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py4,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Recyled Toys and School materials',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Recycle',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='t&srecycle.html',auto_open=False)
        y5=[]
        py5=[]
        for i in self.y5:
            y5.append(int(i))
        for i in self.py5:
            py5.append(int(i))
        y5=np.array(y5)
        py5=np.array(py5)
        trace1 = go.Scatter(
            x = x1,
            y = y5,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py5,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Discarded Toys and School materials',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Discarded',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='t&sdiscarded.html',auto_open=False)
        y6=[]
        py6=[]
        for i in self.y6:
            y6.append(int(i))
        for i in self.py6:
            py6.append(int(i))
        y6=np.array(y6)
        py6=np.array(py6)
        trace1 = go.Scatter(
            x = x1,
            y = y6,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py6,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Incenerated Toy And School material',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Incenerated',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='t&sincenerated.html',auto_open=False)
        y7=[]
        py7=[]
        for i in self.y7:
            y7.append(int(i))
        for i in self.py7:
            py7.append(int(i))
        y7=np.array(y7)
        py7=np.array(py7)
        trace1 = go.Scatter(
            x = x1,
            y = y7,
            mode = 'lines+markers',
            name = 'actual values',
            
        )
        trace2 = go.Scatter(
            x = x1,
            y = py7,
            mode = 'lines+markers',
            name = 'impact values'
        )
        layout= go.Layout(
            title= 'Toy And School waste in ocean',
            hovermode= 'closest',
            xaxis= dict(
                title= 'Year',
                ticklen= 1,
                zeroline= False,
                gridwidth= 1,
            ),
            yaxis=dict( 
                title= 'Ocean waste',
                ticklen= 5,
                gridwidth= 1,
            ),
            showlegend= True
        )
        data=[trace1,trace2]
        fig= go.Figure(data,layout=layout)
        plot(fig,filename='t&socean.html',auto_open=False)
class paper:
    def predictions(self,arr):
        self.b=[]
        for i in range(0,len(arr)):
            self.b.append([arr[i][0],(arr[i][1]*1.46)+(arr[i][1]*0.213),(arr[i][1]*0.26+arr[i][1]*0.213),(arr[i][1]*1.066+arr[i][1]*1.066),(arr[i][1]*0.24+arr[i][1]*0.29)])
        return self.b[len(b)-1][1:]
    def visualize(self):
        data = [go.Bar(
                    x=['carban dioxide', 'methane', 'sulfur dioxide','nitrogen oxide'],
                    y=self.b[len(b)-1][1:]
            )]
        plot(data,filename='paperplot.html')
        

        
        

ls=[['Cloth'],[2015,2000,'Kilograms'],[2012,16,'Kilograms'],[2014,34,'Tonnes'],[2005,555,'Kilograms']]
b=preProcess(ls)
c=paper()
k=c.predictions(b)
print(k)
c.visualize()

"""        


        

            
 
        
