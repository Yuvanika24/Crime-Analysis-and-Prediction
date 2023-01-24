#!/usr/bin/env python
# coding: utf-8

# # <center>CRIME ANALYSIS OF WOMEN IN INDIA

# ## <center> COMAPRISION OF ARIMA AND SARIMA (SARIMAX) MODEL

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


# In[17]:


df  = pd.read_excel('crime.xlsx')


# In[18]:


c=0
for i in range(len(df)):
    if df["STATE/UT"][i]=='All India':
        c=c+1
print("All India crimes : ",c)


# In[19]:


df.set_index("STATE/UT", inplace = True)


# In[20]:


df=df.drop("All India")
df.tail()


# In[21]:


crime=[]
head=df["CRIME HEAD"].unique()
#print("Crime heads",head)
k=0
for i in head:
    crime.append(i)
    print(k,i)
    k+=1
#print(crime)
cr=int(input("\nEnter crime name from given menu(0-7) : "))
print("\nSelected Crime : ",crime[cr])


# In[22]:


cr1=[]
b=0
a=0
for z in range(2001,2013):
    for i in range(len(df)):
        if df["CRIME HEAD"][i]==crime[cr]:
            a=df[z][i]
            b+=a
    cr1.append(b)
    b=0
yr=[feat for feat in df]
yr.remove("CRIME HEAD")
print(f'List of Total " {crime[cr]} " case registered:\n')
print(yr)
print(cr1)


# In[23]:


yeardf =pd.DataFrame(yr)
crimedf =pd.DataFrame(cr1)


# In[24]:


CrimeTable = pd.concat([yeardf,crimedf],axis=1)
CrimeTable.columns=["Year","Total Crime Recorded"]
CrimeTable


# In[25]:


plt.style.use('fivethirtyeight')
plt.plot(yr, cr1, color='#e70631')
plt.xlabel('YEAR')
plt.ylabel('CRIME CASES RECORDED')
plt.title(crime[cr])
plt.show()


# In[26]:


df=df.reset_index()


# In[27]:


states=df["STATE/UT"].unique()


# In[28]:


states=list(states)


# In[29]:


stateG=df.groupby(["STATE/UT"])
states1 =pd.DataFrame(states)
states1


# In[30]:


totalC=[]

year=int(input("Enter year: "))
s1=stateG.agg({year:"sum"})
for i in range(len(s1)):
        val=s1[year][i]
        totalC.append(val)


# In[31]:


totalC1 =pd.DataFrame(totalC)


# In[32]:


tc = pd.concat([states1,totalC1],axis=1)


# In[33]:


tc.columns=["States","Total Crime"]
tc


# In[34]:


tc=tc.set_index("States")


# In[35]:


#Visualisation of TOTAL COMMITTED CRIME STATE-WISE
plt.style.use('fivethirtyeight')
plt.title('TOTAL CRIME STATE-WISE')
plt.xlabel('STATE')
plt.ylabel('CRIME INCIDENTS')
tc["Total Crime"].plot(kind='bar',figsize=(20,10),fontsize=15, color='#ec9104')


# In[36]:


#Analysis of increase in crime from 2001 to 2012
#CRH=['ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
 #'CRUELTY BY HUSBAND OR RELATIVES',
 #'DOWRY DEATH',
 #'IMMORAL TRAFFIC(PREVENTION)ACT',
 #'INDECENT REPRESENTATION OF WOMEN(PREVENTION)ACT',
 #'INSULT TO THE MODESTY OF WOMEN',
 #'KIDNAPPING & ABDUCTION',
 #'RAPE']
#Cgroup=df.groupby("CRIME HEAD")
#Cgroup.sum()
crime=list(crime)
crime


# In[37]:


Cgroup=df.groupby("CRIME HEAD")
Cgroup.sum()


# In[38]:


Cgroup1=Cgroup.agg({2001:'sum',2012:'sum'})
Cgroup1


# In[39]:


#Visualization of graph comparing crimes in years 2001 and 2012
plt.style.use('fivethirtyeight')
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = []
bars2 = []
for i in range(len(Cgroup1)):
        bars1.append(Cgroup1[2001][i])
for i in range(len(Cgroup1)):
        bars2.append(Cgroup1[2012][i])
        
        
plt.figure(figsize=(15,8),)

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

plt.title('COMPARISON OF CRIME BETWEEN 2001 AND 2012 ',fontweight='bold')

# Make the plot
plt.bar(r1, bars1, color='#134402', width=barWidth, edgecolor='white', label='2001')
plt.bar(r2, bars2, color='#42c714', width=barWidth, edgecolor='white', label='2012')


plt.ylabel('CRIME INCIDENTS',fontweight='bold')
plt.yticks(fontsize=12)
# Add xticks on the middle of the group bars
plt.xlabel('CRIME HEADS', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))],crime ,rotation="vertical",fontsize=12)


# Create legend & Show graphic
plt.legend(fontsize=12)
plt.show()


# In[40]:


df2  = pd.read_excel('crime.xlsx',sheet_name="Sheet2")
df2.head()


# In[ ]:





# In[26]:


#Analysis of Crime cases in a specified state and year
#Cr=["Assault on women with intent to outrage her modesty","Cruelty by Husband or his Relatives","Dowry Deaths","Importation of Girls","Insult to modesty of Women","Kidnapping and Abduction","Rape"]
crime


# In[27]:


states=df2["STATE/UT"].unique()
states


# In[28]:


Cr1=pd.DataFrame(crime)
Cr1


# In[29]:


states=list(states)
#The input of the state name and year is taken from the user
k=0
for i in states:
    print(f'{k}.',i)
    k+=1
s=int(input("Enter State/UT no. as indicated in the list: "))
print(states[s])
year=int(input("Enter year: "))
print("Year:",year)


# In[30]:


mask1=df2['STATE/UT']==states[s]
mask2=df2['Year']==year
dd=df2[mask1 & mask2]
dd.head(5)


# In[31]:


dd=dd.set_index('DISTRICT')
dd=dd.drop("TOTAL")
dd=dd.reset_index('DISTRICT')
dd


# In[32]:


#SUM of all crimes
dd=dd.agg({"Assault on women with intent to outrage her modesty":"sum","Cruelty by Husband or his Relatives":"sum","Dowry Deaths":"sum","Importation of Girls":"sum","Insult to modesty of Women":"sum","Kidnapping and Abduction":"sum","Rape":"sum"})
dd


# In[33]:


dd=list(dd)
dd1=pd.Series(dd)
dd1=pd.DataFrame(dd1)
#dd1
DaTa = pd.concat([Cr1,dd1],axis=1)
#DaTa


# In[34]:


DaTa.columns=["Crime Heads","Total Crime"]
DaTa=DaTa.set_index("Crime Heads")


# In[35]:


#Analysis of crime in a given state and year District-wise
StatesList=df2["STATE/UT"].unique()
StatesList=list(StatesList)


# In[36]:


#The state name in taken as an input from the user
k=0
for i in StatesList:
    print(f'{k}.',i)
    k+=1
s=int(input("Enter State/UT no. as indicated in the list: "))
print(StatesList[s])


# In[37]:


somedata=df2['STATE/UT']==states[s]
somedata=df2[somedata]
somedata.head(6)


# In[38]:


data=somedata.groupby('DISTRICT')
#data


# In[39]:


data1=data.agg({"Assault on women with intent to outrage her modesty":"sum",
                "Cruelty by Husband or his Relatives":"sum",
                "Dowry Deaths":"sum",
                "Importation of Girls":"sum",
                "Insult to modesty of Women":"sum",
                "Kidnapping and Abduction":"sum","Rape":"sum"})
data1


# In[40]:


data2=data1.drop('TOTAL')
data2


# In[41]:


indx=data2.index
#indx


# In[42]:


indx=list(indx)
#indx


# In[43]:


indx_df=pd.DataFrame(indx)
#indx_df


# In[44]:


CrimeHead=["Assault on women with intent to outrage her modesty","Cruelty by Husband or his Relatives","Dowry Deaths","Importation of Girls","Insult to modesty of Women","Kidnapping and Abduction","Rape"]
#CrimeHead


# In[45]:


#The Crime Head input is taken from the user
Crime=[]
k=0
for i in CrimeHead:
    Crime.append(i)
    print(f'{k}.',i)
    k+=1
#print(Crime)
c=int(input("Enter crime name from given menu: "))
print(Crime[c])


# In[46]:


CrimeCases=[]
for i in range(len(data2)):
    CrimeCases.append(data2[Crime[c]][i])


# In[47]:


CrimeCases_df=pd.DataFrame(CrimeCases)
#CrimeCases_df


# In[48]:


DataTable = pd.concat([indx_df,CrimeCases_df],axis=1)
DataTable.columns=["Districts","Crime Cases"]
DataTable=DataTable.set_index("Districts")
#DataTable


# In[49]:


#Webscrapping to fetch hex color codes from a website
import bs4
import requests
colorp=[]
c=0
url="http://www.color-hex.com/"
data=requests.get(url)
soup1=bs4.BeautifulSoup(data.text,"html.parser")
for z in soup1.find_all("a"):
    a1=str(z.get("title"))
    if a1[:1]=="#":
        #print(a1[:7])
        colorp.append(a1[:7])
        c=c+1
print(c)


# In[50]:


colorp1=colorp[:34]
len(colorp1)


# In[51]:


#Visualization of graph on CRIME CASES DISTRICT-WISE
plt.style.use('fivethirtyeight')
plt.title(StatesList[s])
plt.xlabel('DISTRICTS')
plt.ylabel('CRIME INCIDENTS')
DataTable["Crime Cases"].plot(kind='bar',color=colorp1,fontsize=15,legend=False,figsize=(20,10))


# In[52]:


import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


# In[53]:


from statsmodels.tsa.arima.model import ARIMA
names = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008'
         , '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016','2017','2018','2019','2020','2021','2022']
def state_case(state, case):
    for i in range(0, len(df)):
        if df.iloc[i,0] == state and df.iloc[i,1]==case:
            temp = df.iloc[i, 2:]
            train = np.array(temp)
            train = train.astype(np.int64)
            train = np.reshape(train, (-1, 1))
            
    temp = pd.DataFrame(train)
    sm.graphics.tsa.plot_acf(temp.values.squeeze())
    sm.graphics.tsa.plot_pacf(temp.values.squeeze(),lags=5)
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()
    pred = model_fit.predict(start=13, end=22)
    new_data = np.append(train, pred)
    plt.figure(figsize=(16,5))
    plt.bar(names, new_data)
    year = [2013, 2014, 2015, 2016,2017,2018,2019,2020,2021,2022]
    for w in range(0, 10):
        print(year[w]," " ,pred[w].round(0))
    return pred


# In[54]:


print("Enter the state:")
s=input()
print("Enter the crime")
c=input()
pred = state_case(s, c)


# In[57]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
names = ['2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008'
         , '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016','2017','2018','2019','2020','2021','2022']
def state_case(state, case):
    for i in range(0, len(df)):
        if df.iloc[i,0] == state and df.iloc[i,1]==case:
            temp = df.iloc[i, 2:]
            train = np.array(temp)
            train = train.astype(np.int64)
            train = np.reshape(train, (-1, 1))

    temp = pd.DataFrame(train)
    sm.graphics.tsa.plot_acf(temp.values.squeeze(),lags=10)
    sm.graphics.tsa.plot_pacf(temp.values.squeeze(),lags=5)
    model = SARIMAX(train, order=(12,1,1))
    model_fit = model.fit()
    pred = model_fit.predict(start=13, end=22)
    model_fit.summary()
    new_data = np.append(train, pred)
    plt.figure(figsize=(16,5))
    plt.bar(names, new_data)
    year = [2013, 2014, 2015, 2016,2017,2018,2019,2020,2021,2022]
    for w in range(0, 10):
        print(year[w]," " ,pred[w].round(0))
    return pred


# In[58]:


print("Enter the state:")
s=input()
print("Enter the crime")
c=input()
pred = state_case(s, c)


#  
