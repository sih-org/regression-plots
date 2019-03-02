import operator
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(0)
class cloth:
    def pedictions(self,datas):
        clothdata= pd.read_csv('cloth$data.csv')
        clothdata1= pd.read_csv('cloth$data1.csv')
        x0=clothdata['Cloth Waste'][:].values
        y1=clothdata['commercial(co2 in tonnes)'][:].values
        py1=clothdata1['commercial(co2 in tonnes)'][:].values
        y2=clothdata['commercial methane'][:].values
        py2=clothdata1['commercial methane'][:].values
        y3=clothdata['commercial(n2o)'][:].values
        py3=clothdata1['commercial(n2o)'][:].values
        py4=clothdata1['Recycled'][:].values
        py5=clothdata1['discarded'][:].values
        py6=clothdata1['incenerated'][:].values
        py7=clothdata1['waste in ocean'][:].values
        y4=clothdata['Recycled'][:].values
        y5=clothdata['discarded'][:].values
        y6=clothdata['incenerated'][:].values
        y7=clothdata['waste in ocean'][:].values
        x0 = x0[:, np.newaxis]
        y1 = y1[:, np.newaxis]
        y2 = y2[:, np.newaxis]
        y3 = y3[:, np.newaxis]
        y4 = y4[:, np.newaxis]
        y5 = y5[:, np.newaxis]
        y6 = y6[:, np.newaxis]
        y7 = y7[:, np.newaxis]
        py1= py1[:,np.newaxis]
        py2= py2[:,np.newaxis]
        py3= py3[:,np.newaxis]
        py4= py4[:,np.newaxis]
        py5= py5[:,np.newaxis]
        py6= py6[:,np.newaxis]
        py7= py7[:,np.newaxis]
        polynomial_features= PolynomialFeatures(degree=1)
        x0_poly = polynomial_features.fit_transform(x0)
        predx=[]
        model1 = LinearRegression()
        model2 = LinearRegression()
        model3 = LinearRegression()
        model4 = LinearRegression()
        model5 = LinearRegression()
        model6 = LinearRegression()
        model7 = LinearRegression()
        model1.fit(x0_poly, y1)
        model2.fit(x0_poly, y2)
        model3.fit(x0_poly, y3)
        model4.fit(x0_poly, y4)
        model5.fit(x0_poly, y5)
        model6.fit(x0_poly, y6)
        model7.fit(x0_poly, y7)
        for data in datas:
            dump=data[1]+clothdata['Cloth Waste'][data[0]-1990]
            predx.append([dump])
            py1[data[0]-1990]=model1.predict(polynomial_features.fit_transform(predx))
            py2[data[0]-1990]=model2.predict(polynomial_features.fit_transform(predx))
            py3[data[0]-1990]=model3.predict(polynomial_features.fit_transform(predx))
            py4[data[0]-1990]=model4.predict(polynomial_features.fit_transform(predx))
            py5[data[0]-1990]=model5.predict(polynomial_features.fit_transform(predx))
            py6[data[0]-1990]=model6.predict(polynomial_features.fit_transform(predx))
            py7[data[0]-1990]=model7.predict(polynomial_features.fit_transform(predx))
        
obj= cloth()
print(a,b)
obj.pedictions([[1990,500]])

        


        
            
        
            
            
            
