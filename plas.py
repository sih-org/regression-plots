class school:
    def predictions(self,datas):
        schooldata= pd.read_csv('Schools_and_toys.csv')
        schooldata1= pd.read_csv('Schools_and_toys1.csv')
        self.x0=clothdata['Cloth Waste'][:].values
        self.y1=clothdata['commercial(co2 in tonnes)'][:].values
        self.py1=clothdata1['commercial(co2 in tonnes)'][:].values
        self.y2=clothdata['methane_due_to_cloth'][:].values
        self.py2=clothdata1['methane_due_to_cloth'][:].values
        self.y3=clothdata['N2o_due_to_cloth'][:].values
        self.py3=clothdata1['N2o_due_to_cloth'][:].values
        self.py4=clothdata1['Recycled'][:].values
        self.py5=clothdata1['discarded'][:].values
        self.py6=clothdata1['incenerated'][:].values
        self.py7=clothdata1['waste in ocean'][:].values
        self.y4=clothdata['Recycled'][:].values
        self.y5=clothdata['discarded'][:].values
        self.y6=clothdata['incenerated'][:].values
        self.y7=clothdata['waste in ocean'][:].values
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
        diffsum=[0,0,0,0,0,0,0]
        y1p=model1.predict(x0_poly)
        y2p=model2.predict(x0_poly)
        y3p=model3.predict(x0_poly)
        y4p=model4.predict(x0_poly)
        y5p=model5.predict(x0_poly)
        y6p=model6.predict(x0_poly)
        y7p=model7.predict(x0_poly)
        error=[abs(y1p-y1),abs(y2p-y2),abs(y3p-y3),abs(y4p-y4),abs(y5p-y5),abs(y6p-y6),abs(y7p-y7)]
        for data in datas:
            dump=data[1]+clothdata['Cloth Waste'][data[0]-1990]
            predx=[[dump]]
            k=data[0]-1990
            py1[k]=model1.predict(polynomial_features.fit_transform(predx))
            py2[k]=model2.predict(polynomial_features.fit_transform(predx))
            py3[k]=model3.predict(polynomial_features.fit_transform(predx))
            py4[k]=model4.predict(polynomial_features.fit_transform(predx))
            py5[k]=model5.predict(polynomial_features.fit_transform(predx))
            py6[k]=model6.predict(polynomial_features.fit_transform(predx))
            py7[k]=model7.predict(polynomial_features.fit_transform(predx))
            diffsum[0]=diffsum[0]+py1[k]-y1[k]+error[0][k]
            diffsum[1]=diffsum[1]+py2[k]-y2[k]+error[1][k]
            diffsum[2]=diffsum[2]+py3[k]-y3[k]+error[2][k]
            diffsum[3]=diffsum[3]+py4[k]-y4[k]+error[3][k]
            diffsum[4]=diffsum[4]+py5[k]-y5[k]+error[4][k]
            diffsum[5]=diffsum[5]+py6[k]-y6[k]+error[5][k]
            diffsum[6]=diffsum[6]+py7[k]-y7[k]+error[6][k]
        return diffsum
        
