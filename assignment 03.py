


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import sklearn.metrics as sm

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import MinMaxScaler

from scipy.optimize import curve_fit
from numpy import arange
import scipy.optimize as opt

import matplotlib.ticker as mticker



font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

def worldbankdatafile(f):
    '''this function will call the worldbank data file in its original format
    and will return the transpose of the file'''
    worldData = f
    trans = worldData.T
    return worldData,trans


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

methane = pd.read_csv('C://Users//samkh//Desktop//methane//methaneEmission.csv')
nit = pd.read_csv('C://Users//samkh//Desktop//methane//nitrous.csv')

meth = worldbankdatafile(methane)#calling the worldbank data file
nitrous = worldbankdatafile(nit)
meth # viewing original and transposed file

methane_df = worldbankdatafile(methane)[0] #calling only the original format not transposed
nit_df = worldbankdatafile(nit)[0]

methane_df

Countries_meth = methane_df[(methane_df['Indicator Code'] == 'EN.ATM.METH.KT.CE')]
c_meth = Countries_meth.drop(['2021'], axis=1)
c_meth = c_meth.drop(['2020'], axis=1)
c_meth.rename(columns = {'Unnamed: 4':'1989'}, inplace = True)
c_meth = c_meth.drop(['1989'], axis=1)

nit_df
Countries_nit = nit_df[(nit_df['Indicator Code'] == 'EN.ATM.NOXE.KT.CE')]
c_nit = Countries_nit.drop(['2021'], axis=1)
c_nit = c_nit.drop(['2020'], axis=1)
c_nit.rename(columns = {'Unnamed: 4':'1989'}, inplace = True)
c_nit = c_nit.drop(['1989'], axis=1)


c_meth = c_meth.round(2)
c_meth.dropna()


c_nit = c_nit.round(2)
c_nit.dropna()
print(c_nit)

meth_nit1990 = c_meth[['Country Name','1990']]
meth_nit1990.rename(columns = {'1990':'MethaneEmission1990'}, inplace = True)
meth_nit1990['Nitrous_in1990'] = c_nit[['1990']]
meth_nit1990 = meth_nit1990.dropna(axis=0)
print(meth_nit1990)




##methane emission of all the countries of worldbank in the year 2019 
meth_nit2019 = c_meth[['Country Name','2019']]
meth_nit2019.rename(columns = {'2019':'MethaneEmission2019'}, inplace = True)
meth_nit2019['Nitrous_in2019'] = c_nit[['2019']]
meth_nit2019 = meth_nit2019.dropna(axis=0)
print(meth_nit2019)










fontsize = 20
plt.scatter(meth_nit1990['MethaneEmission1990'],meth_nit1990['Nitrous_in1990'],color = 'blue',s=30, label='Countries')
plt.title('Countries Methane v/s nitrous Emission(1990)',fontdict={'fontsize': fontsize})
plt.xlabel('MethaneEmission')
plt.xticks(rotation=60)
plt.ylabel('NitrousOxide Emission')
plt.legend(loc='upper left')
plt.show()


plt.scatter(meth_nit2019['MethaneEmission2019'],meth_nit2019['Nitrous_in2019'],color = 'blue',s=30, label='Countries')
plt.title('Countries Methane v/s nitrous Emission(2019)',fontdict={'fontsize': fontsize})
plt.xlabel('Methane Emission')
plt.xticks(rotation=60)
plt.ylabel('NitrousOxide Emission')
plt.legend(loc='upper left')
plt.show()




s = MinMaxScaler()
s.fit(meth_nit1990[['MethaneEmission1990']])
meth_nit1990['MethaneEmission1990'] = s.transform(meth_nit1990[['MethaneEmission1990']])
s.fit(meth_nit1990[['Nitrous_in1990']])
meth_nit1990['Nitrous_in1990'] = s.transform(meth_nit1990[['Nitrous_in1990']])
print(meth_nit1990)

s.fit(meth_nit2019[['MethaneEmission2019']])
meth_nit2019['MethaneEmission2019'] = s.transform(meth_nit2019[['MethaneEmission2019']])
s.fit(meth_nit2019[['Nitrous_in2019']])
meth_nit2019['Nitrous_in2019'] = s.transform(meth_nit2019[['Nitrous_in2019']])
print(meth_nit2019)


x = meth_nit1990['MethaneEmission1990']
y = meth_nit1990['Nitrous_in1990']



data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method for clustering')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig('elbow1990.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()

year1990 = meth_nit1990.drop('Country Name',axis='columns')

kmeans = KMeans(n_clusters= 3)
 
#predict the labels of clusters.
pred = kmeans.fit_predict(year1990)
 
print(pred)
center = kmeans.cluster_centers_
print(center)

clusters1990 = meth_nit1990.iloc[:,:]
clusters1990 = pd.concat([meth_nit1990, pd.DataFrame(pred, columns=['cluster'])], axis = 1)
clusters1990


year2019 = meth_nit2019.drop('Country Name',axis='columns')

kmeans = KMeans(n_clusters= 3)
 
#predict the labels of clusters.
pred_val = kmeans.fit_predict(year2019)
 
print(pred_val)
cent = kmeans.cluster_centers_
print(cent)

clusters2019 = meth_nit2019.iloc[:,:]
clusters2019 = pd.concat([meth_nit2019, pd.DataFrame(pred_val, columns=['cluster'])], axis = 1)
clusters2019

df0 = clusters1990[clusters1990.cluster==0]
df1 = clusters1990[clusters1990.cluster==1]
df2 = clusters1990[clusters1990.cluster==2]


fig,ax = plt.subplots(figsize=(10,6))



 


plt.scatter(df0.MethaneEmission1990,df0['Nitrous_in1990'],color='yellow',s=50,label='cluster0')
plt.scatter(df1.MethaneEmission1990,df1['Nitrous_in1990'],color='pink',s=50,label='cluster1')
plt.scatter(df2.MethaneEmission1990,df2['Nitrous_in1990'],color='cyan',s=50,label='cluster2')
plt.scatter(center[:,0] , center[:,1] , s = 100, color = 'k',label='centriods')


plt.title('Countries Clusters1990')
plt.xlabel('Methane emssion')
plt.xticks(rotation=60)
plt.ylabel('Nitrous Oxide Emission')
plt.legend()
plt.savefig('cluster1990.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()


df3 = clusters2019[clusters2019.cluster==0]
df4 = clusters2019[clusters2019.cluster==1]
df5 = clusters2019[clusters2019.cluster==2]
fig,ax = plt.subplots(figsize=(10,6))



 


plt.scatter(df3.MethaneEmission2019,df3['Nitrous_in2019'],color='blue',s=50,label='cluster0')
plt.scatter(df4.MethaneEmission2019,df4['Nitrous_in2019'],color='red',s=50,label='cluster1')
plt.scatter(df5.MethaneEmission2019,df5['Nitrous_in2019'],color='orange',s=50,label='cluster2')
plt.scatter(cent[:,0] , cent[:,1] , s = 100, color = 'k',label='centriods')


plt.title('Countries Clusters2019')
plt.xlabel('Methane emssion')
plt.xticks(rotation=60)
plt.ylabel('Nitrous Oxide Emission')
plt.legend()
plt.savefig('cluster2019.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()

methaneofAus = methane[methane['Country Name']=='Australia']
del methaneofAus['Country Name']
del methaneofAus['Indicator Name']
del methaneofAus['Country Code']
del methaneofAus['Indicator Code']
del methaneofAus['2020']
del methaneofAus['2021']
#del methaneofAus['1960']
print(type(methaneofAus))
gdparr = methaneofAus.values.tolist()
print(gdparr)


year = []
for i in range(31):
    year.append(1989+i)


x = []
for i in range(31):
    x.append(i)

dfaus = pd.DataFrame(columns = ['years','methane emission'],
                    index = x )
print(dfaus.loc[0][0])
for i in range(31):
    dfaus.loc[i] = [year[i],gdparr[0][i]]
    dfaus = dfaus.dropna(axis=0)
    dfaus['years'] = dfaus['years'].astype(float)
print(dfaus)


#plt.scatter(dfusa['years'],dfusa['gdp'],color = 'red',s=20,label = 'gdpvalues of USA')
dfaus.plot('years','methane emission',color='red')
plt.title('methane emission Australia(past 29 years)')
plt.xlabel('Year from 1990 to 2019')
plt.ylabel('methane emission')
plt.legend()
plt.savefig('methane.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()





param, covar = opt.curve_fit(logistic, dfaus["years"], dfaus["methane emission"],
p0=(3e12, 0.03, 2000.0))


sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
dfaus["fit"] = logistic(dfaus["years"], *param)
dfaus.plot("years", ["methane emission", "fit"])
plt.title('curve fit(logistic)')
plt.xlabel('years')
plt.ylabel('methane emisision')
plt.savefig('curvefit.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()




year = np.arange(1990, 2031)
print(year)
forecast = logistic(year, *param)

plt.figure()
plt.plot(dfaus["years"], dfaus["methane emission"], label="methane")
plt.plot(year, forecast, label="forecast")
plt.title('forecast of10 years using curve fit')
plt.xlabel("years")
plt.ylabel("methane emission")
plt.legend()
plt.savefig('forecast.png', format='png',dpi=600,
            bbox_inches='tight')
plt.show()



