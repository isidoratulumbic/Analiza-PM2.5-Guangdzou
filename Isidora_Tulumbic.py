# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:51:05 2022

@author: Isidora
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
#%%
# I DEO: ANALIZA PODATAKA
#%%
#učitavanje u dataframe format

df = pd.read_csv("C:/Users/isido/Desktop/PO-domaći/domaći/GuangzhouPM20100101_20151231.csv")
df.head()
#%%
print('Broj obeležja je: ', df.shape[1])
print('Broj uzoraka je: ', df.shape[0])
print('Obeležja su:\n', df.dtypes)

#kategorička obeležja
print('Godine: ', df['year'].unique())
print('Meseci: ', df['month'].unique())
print('Dani: ', df['day'].unique())
print('Sati: ', df['hour'].unique())
print('Sezone: ', df['season'].unique())
print('Vrednosti pravca vetra: ', df['cbwd'].unique())

#provera nedostajućih vrednosti
print(df.isnull().sum())
#procentualni uvid u nedostajuće vrednosti
print(df.isnull().sum()/df.shape[0]*100)

#%%
df.drop(['PM_City Station', 'PM_5th Middle School'], axis = 1, inplace = True)
df.head()

#%%
#provera nedostajućih vrednosti
print(df.isnull().sum())
#procentualni uvid u nedostajuće vrednosti
print(df.isnull().sum()/df.shape[0]*100)
#%%
#prvo se rešavamo nedostajućih vrednosti u koloni sezona
season_null = df[df['season'].isnull()] #tražimo koji uzorak ima null vrednost za obeležje sezona
print(season_null)
#to je uzorak 52584
#%%
#u rezultatu se vidi da u ovom uzorku postoje null vrednosti i za preostala obeležja, pa ćemo ovaj uzorak izbrisati
df.drop(52583, axis = 0, inplace = True)
df.tail()
#%%
#preostaje da se rešimo još nedostajućih vrednosti obeležja PM_US Post
#grupisaću podatke po godinama i proveriti da li postoji neka godina kojoj nedostaje vis od 80% podataka za PM_US Post obeležje
gbdf = df.groupby(by='year').agg('count')
gbdf['PM_US Post']/gbdf['month'] < 0.2
#2010 i 2011 godina imaju više od 80% nedost. pod. za obeležje PMUS Post, izbrisaću ovu godinu iz baze jer nam je PM obeležje jako bitno 
#za dalju analizu pa nam uzorci te godine bez ovog obeležja nisu od nekog značaja
#%%
del_year = gbdf.index[gbdf['PM_US Post']/gbdf['month'] < 0.2]
del_year

print('Broj uzoraka i obelezja pre brisanja: ', df.shape)
print('Broj jedinstvenih godina pre brisanja: ', len(df['year'].unique()))
df = df[~df['year'].isin(del_year)]

print('Broj uzoraka i obelezja posle brisanja: ', df.shape)
print('Broj jedinstvenih godina posle brisanja: ', len(df['year'].unique()))

print(df)

print(df.isnull().sum()/df.shape[0]*100)
#%%
#ostalo je da se uradi dopuna preostalih nedostajućih vrednosti
#nedostajuće menjam sa prosekom vrednosti za prethodni ili naredni sat
df.isna().sum()

df['PM_US Post'].fillna(method='ffill',inplace=True)

#provera da li još uvek postoji nedostajućih vrednosti
print(df.shape)
df.isna().sum()   
df.head()
#%%
#sada treba proveriti da li medju podacima postoje neke nevalidne vrednosti
#to radimo pomoću describe fje
df.describe()
df['DEWP'].describe()
df['HUMI'].describe()
#%%
#obeležja dewp i humi imaju min vrednost -9999
#nevalidne vrednosti menjam null vrednostima
df.loc[df['DEWP']< -20, 'DEWP'] = np.nan
print(df['DEWP'].isna().sum()/df.shape[0]*100)
#uklanjam ove podatke jer ih ima manje od 1% pa njihovo brisanje neće puno uticati na skup podataka
df.dropna(inplace = True, axis = 0)
df['DEWP'].isna().sum()  
#%%
df.isna().sum() 
print('Broj obeležja nakon sređivanja podataka je: ', df.shape[1])
print('Broj uzoraka je nakon sređivanja podataka je: ', df.shape[0])
#%%
#resetovanje indeksa zbog lakšeg pretraživanja
df.reset_index(inplace=True)
df.drop('index',axis=1, inplace=True)
df
#%%
#raspodele

#temp. rose
plt.hist(df['DEWP'], density = True, bins=20, label='Temperatura rose/kondenzacije(°C)')
plt.legend()
plt.show()
#najčešća temperature rose je između 23 i 25 stepena.
#raspodela koja je dosta iskrivljena na desno tj. ka 
#većim temperaturama što znači da je nešto manja verovatnoća temperatura ispod  23 stepena,
#a izuzetno mala preko 26 i manje od 0 stepeni

#temp
plt.hist(df['TEMP'], density = True, bins=20, label='Temperatura(°C)')
plt.legend()
plt.show()
#najčešća temperatura je između 24 i 27 stepena.
#raspodela je takva da je nešto manja verovatnoća temperatura ispod  23 stepena i iznad 28,
#a izuzetno mala preko 35 i manje od 5 stepeni

#vl.vazduha
plt.hist(df['HUMI'], density = True, bins=20, label='Vlažnost vazduha(%)')
plt.legend()
plt.show()
#najčešća vlažnost vazduha je između 90 i 95%
#raspodela koja je dosta iskrivljena na desno tj. ka 
#većim vl što znači da je nešto manja verovatnoća  vlažnosti ispod 88%,
#a izuzetno mala ispod 30%

#vaz. prit.
plt.hist(df['PRES'], density = True, bins=20, label='Vazdušni pritisak (hPa)')
plt.legend()
plt.show()
# najčešći oko 1000hPa, izuzetno mala ver. ispod 990hPa i 1021hPa

#Kum. brz. vetra
plt.hist(df['Iws'],  density = True, bins=20, label='Kumulativna brzina vetra (m/s)')
plt.legend()
plt.show()
#najčešća između 0 i 10m/s, nešto manja ver. 10 i 40, a izuzetno mala pojava brz vetra od 50m/s
#padavine
plt.hist(df['precipitation'],  density = True, bins=20, label='Padavine na sat (mm)')
plt.legend()
plt.show()
#kum. pad.
plt.hist(df['Iprec'], density = True, bins=20, label='Kumulativne padavine (mm)')
plt.legend()
plt.show()

#raspodela temperature 2012.i 2013.god.
df_year = df.set_index('year')
df_year.head()

plt.hist(df_year.loc['2012','TEMP'], density=True, alpha=0.5, bins=50, label = '2012')
plt.hist(df_year.loc['2013','TEMP'], density=True, alpha=0.5, bins=50, label='2013')

plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
plt.legend()
#raspodele temperatura za 2012 i 2013 godinu su vrlo slične tako da je veoma teško međusobno razlikovati temp. za ove dve godine

plt.hist(df_year.loc['2014','TEMP'], density=True, alpha=0.5, bins=50, label = '2014')
plt.hist(df_year.loc['2015','TEMP'], density=True, alpha=0.5, bins=50, label='2015')

plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
plt.legend()
#isti slučaj i sa ove dve godine

#koeficijent asimetrije i spljoštenosti temperature za 2012. godinu
print('koef.asimetrije:  %.2f' % skew(df_year.loc['2012','TEMP']))
print('koef.spljoštenosti:  %.2f' % kurtosis(df_year.loc['2012','TEMP']))
#s obzirom da je koef.asim. manji od 0 to znači da imamo desnu asimetričnu raspodelu(modusa, med. i sr. vr.)

temp_2012 = df_year.loc['2012','TEMP']
sb.distplot(temp_2012, fit=norm)
plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
#%%analiza pm2.5
#raspodela
plt.hist(df['PM_US Post'], density = True, bins=30, label='Koncentracija PM2.5 čestica(µg/m3)')
plt.legend()
plt.show()
#najčešća ver. konc. pm2.5 čestica između 20 i 40µg/m3, nešto manja ver. od 0-20 i od 40 do 100, dosta manja iznad 100µg/m3
#blaga iskrivljenost  na levo tj ka manjim koncentracijama
#%%
#prebacivanje kategoričkih u numerička obeležja
#cbwd
#pristup formiranje dummy varijabli
x1 = df.drop(columns=['cbwd'])
x_temp = pd.get_dummies(df['cbwd'],prefix='cbwd')
x1=pd.concat([x1, x_temp.iloc[:,:-1]],axis =1)
dfd = x1
print(dfd)
#%%
#promena PM2.5 čestica kroz godine
#postavljam godine za indeks
df_year = df.set_index('year')
df_year.head()

f = plt.figure(figsize=(12, 9))
sb.boxplot(data=df, x='year', y='PM_US Post')
plt.xlabel('Godina')
plt.ylabel('Koncentracija PM2.5 čestica(µg/m3)')
print(df['PM_US Post'].unique())
plt.show()
#2014. godine se uočavaju najveće vrednosti PM2.5 čestica dok je u 2015. konc. naglo opala što može ukazati 
#na to da se povelo računa o koncentraciji ovih čestica
#%%
#promena PM2.5 čestica kroz mesece za svaku godinu
f = plt.figure(figsize=(12, 9))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2012 = gb.loc[2012]['PM_US Post']
T_2013 = gb.loc[2013]['PM_US Post']
T_2014 = gb.loc[2014]['PM_US Post']
T_2015 = gb.loc[2015]['PM_US Post']
plt.plot(np.arange(1, 13, 1), T_2012, 'b', label='2012') 
plt.plot(np.arange(1, 13, 1), T_2013, 'r', label='2013')
plt.plot(np.arange(1, 13, 1), T_2014, 'g', label='2014')
plt.plot(np.arange(1, 13, 1), T_2015, 'y', label='2015')
plt.ylabel('Koncentracija PM2.5 čestica(µg/m3)')
plt.xlabel('Mesec')
plt.legend()
#može se uočiti da je u letnjim mesecima koncentracija PM2.5 čestica 
#dosta manja nego u zimskim
#%%
#promena PM2.5 čestica kroz sezone
#postavljam sezone za indeks
df_season = df.set_index('season')
df_season.head()

plt.hist(df_season.loc[2,'PM_US Post'], bins=15, density=True, alpha=0.3, label='leto')
plt.hist(df_season.loc[4,'PM_US Post'], bins=15, density=True, alpha=0.3, label='zima')

plt.xlabel('Koncentracija PM2.5 čestica (µg/m3)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper right')
#kao i na gornjem grafiku uočavamo veću koncentraciju čestica zimi nego leti
#%%
plt.hist(df_season.loc[1,'PM_US Post'], bins=15, density=True, alpha=0.3, label='proleće')
plt.hist(df_season.loc[3,'PM_US Post'], bins=15, density=True, alpha=0.3, label='jesen')
plt.xlabel('Koncentracija PM2.5 čestica (µg/m3)')
plt.ylabel('Verovatnoća')
plt.legend(loc='upper right')
#%%
#promena PM2.5 čestica u odnosu na promenu temerature rose/kondenzacije(DEWP), kao i promenu temperature za 2013. i 2014. godinu
f = plt.figure(figsize=(12, 9))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2013pm = gb.loc[2013]['PM_US Post']
T_2014pm = gb.loc[2014]['PM_US Post'] 
plt.plot(np.arange(1, 13, 1), T_2013pm, 'r', label='2013')
plt.plot(np.arange(1, 13, 1), T_2014pm, 'g', label='2014')
plt.ylabel('Koncentracija PM2.5 čestica(µg/m3)')
plt.xlabel('Mesec')
plt.legend()

f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2013dewp = gb.loc[2013]['DEWP']
T_2014dewp = gb.loc[2014]['DEWP'] 
plt.plot(np.arange(1, 13, 1), T_2013dewp, 'r', label='2013')
plt.plot(np.arange(1, 13, 1), T_2014dewp, 'g', label='2014')
plt.ylabel('Temperatura rose/kondenzacije(°C)')
plt.xlabel('Mesec')
plt.legend()

f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2013temp = gb.loc[2013]['TEMP']
T_2014temp = gb.loc[2014]['TEMP'] 
plt.plot(np.arange(1, 13, 1), T_2013temp, 'r', label='2013')
plt.plot(np.arange(1, 13, 1), T_2014temp, 'g', label='2014')
plt.ylabel('Temperatura(°C)')
plt.xlabel('Mesec')
plt.legend()
#temperatura i temperatura rose simetricne promene, a suprotne u odnosu na PM2.5
#%%
#PM2.5 i brzina vetra za 2013.

f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2013pm = gb.loc[2013]['PM_US Post']
plt.plot(np.arange(1, 13, 1), T_2013pm, 'purple', label='2013')
plt.ylabel('Koncentracija PM2.5 čestica(µg/m3)')
plt.xlabel('Mesec')
plt.legend()

f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2013Iws = gb.loc[2013]['Iws']
plt.plot(np.arange(1, 13, 1), T_2013Iws, 'purple', label='2013')
plt.ylabel('Kumulativna brzina vetra(m/s)')
plt.xlabel('Mesec')
plt.legend()

#%%
#promena PM2.5 čestica u odnosu na padavine tokom meseci 2013. i 2014. godine

gb_year_month=df.groupby(by=['year','month']).mean()
T_2013precipitation = gb_year_month.loc[2013]['precipitation']
T_2014precipitation = gb_year_month.loc[2014]['precipitation']
T_2013pm = gb_year_month.loc[2013]['PM_US Post']
T_2014pm = gb_year_month.loc[2014]['PM_US Post']

plt.figure(figsize=(7,4))
plt.subplot(2,2,1)
plt.plot(np.arange(1,13,1),T_2013precipitation, 'pink',label="padavine")
plt.xlabel('Meseci tokom 2013. godine')
plt.legend()
plt.subplot(2,2,2)
plt.plot(np.arange(1,13,1),T_2014precipitation, 'pink',label="padavine")
plt.xlabel('Meseci tokom 2014. godine')
plt.legend()
plt.subplot(2,2,3)
plt.plot(np.arange(1,13,1),T_2013pm,'y',label="Koncentracija PM2.5 čestica(µg/m3)")
plt.xlabel('Meseci tokom 2013. godine')
plt.legend()
plt.subplot(2,2,4)
plt.plot(np.arange(1,13,1),T_2014pm, 'y',label="Koncentracija PM2.5 čestica(µg/m3)")
plt.xlabel('Meseci tokom 2014. godine')
plt.legend()
#%%
#promena vlažnosti vazduha tokom meseci 2015. godina
f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2015 = gb.loc[2012]['HUMI']
plt.plot(np.arange(1, 13, 1), T_2015, 'b', label='2015')
plt.ylabel('Vlažnost vazduha(%)')
plt.xlabel('Mesec')
plt.legend()

#PM tokom meseci 2015. godine

f = plt.figure(figsize=(5, 4))
gb = df.groupby(by=['year', 'month']).mean()
print(gb)
T_2015pm = gb.loc[2012]['PM_US Post']
plt.plot(np.arange(1, 13, 1), T_2015pm, 'b', label='2015')
plt.ylabel('Koncentracija PM2.5 čestica(µg/m3)')
plt.xlabel('Mesec')
plt.legend()


#%%
#količina padavina kroz godine
#postavljam godine za indeks
df_year = df.set_index('year')
df_year.head()

f = plt.figure(figsize=(12, 9))
sb.boxplot(data=df, x='year', y='precipitation')
plt.xlabel('Godina')
plt.ylabel('Koncentracija PM2.4 čestica(µg/m3)')

plt.show()
print(df)
#%%
#korelacija
#koristimo dfd jer su za utvrdjivanje kor. potrebna numerička obeležja
df_kor = dfd.drop(['No','year', 'month', 'day','hour', 'season'], axis=1)
matr_kor = df_kor.corr()
print(matr_kor)
f = plt.figure(figsize=(12, 9))
sb.heatmap(matr_kor, annot = True)
plt.show()
#velika negativna korelacija se uočava izmedju dewp i pres
#velika pozitivna korelacija se uočava izmedju dewp i temp
#velika negativna korelacija se uočava izmedju temp i pres
#%%
##############SREDITI
#funkcija corr
post=df['PM_US Post']
humi=df['HUMI']
dewp=df['DEWP']
pres=df['PRES']
temp=df['TEMP']
#cbwd=df['cbwd']
iws=df['Iws']
prec=df['precipitation']
iprec=df['Iprec']
sea=df['season']


c1=post.corr(humi)
print("korelacija post humi: %.3f" % c1)
c2=post.corr(dewp)
print("korelacija post dewp: %.3f" % c2)
c3=post.corr(sea)
print("korelacija post season: %.3f" % c3)
c4=post.corr(pres)
print("korelacija post pres: %.3f" % c4)
c5=post.corr(temp)
print("korelacija post temp: %.3f" % c5)
#c6=post.corr(cbwd)
#print("korelacija post cwbd: %.3f" % c6)
c7=post.corr(iws)
print("korelacija post iws: %.3f" % c7)
c8=post.corr(prec)
print("korelacija post prec: %.3f" % c8)
c9=post.corr(iprec)
print("korelacija post iprec: %.3f" % c9)

c10=humi.corr(dewp)
print("korelacija humi dewp: %.3f" % c10)
c11=humi.corr(sea)
print("korelacija humi season: %.3f" % c11)
c12=humi.corr(pres)
print("korelacija humi pres: %.3f" % c12)
c13=humi.corr(iws)
print("korelacija humi iws: %.3f" % c13)
c14=humi.corr(prec)
print("korelacija humi prec: %.3f" % c14)
c15=humi.corr(iprec)
print("korelacija humi dewp: %.3f" % c15)
c15b=humi.corr(temp)
print("korelacija humi temp: %.3f" % c15b)

c16=pres.corr(sea)
print("korelacija pres season: %.3f" % c16)
c17=pres.corr(iws)
print("korelacija pres iws: %.3f" % c17)
c18=pres.corr(prec)
print("korelacija pres prec: %.3f" % c18)
c19=pres.corr(iprec)
print("korelacija pres iprec: %.3f" % c19)
c20=pres.corr(dewp)
print("korelacija pres dewp: %.3f" % c20)
c21=pres.corr(temp)
print("korelacija pres temp: %.3f" % c21)

c22=temp.corr(iws)
print("korelacija temp dewp: %.3f" % c22)
c23=temp.corr(prec)
print("korelacija temp prec: %.3f" % c23)
c24=temp.corr(iprec)
print("korelacija temp iprec: %.3f" % c24)
c25=temp.corr(sea)
print("korelacija temp season: %.3f" % c25)
c26=temp.corr(dewp)
print("korelacija temp season: %.3f" % c26)

c27=iws.corr(prec)
c28=iws.corr(iprec)
c29=iws.corr(sea)
c30=iws.corr(dewp)

c31=prec.corr(iprec)
c32=prec.corr(sea)
c33=prec.corr(dewp)

c34=iprec.corr(sea)
c35=iprec.corr(dewp)

c36=sea.corr(dewp)

###############################
#%%
#1.11 
#prikazujem piecharom intervale koncentracije PM2.5 čestica u odnosu na to koliko su opasni po životnu sredinu

labels = ['Good','Moderate','Unhelalthy for Sensitive Groups', 'Unhelalthy','Very Unhealthy','Hazardous']
df['grp'] = pd.cut(df['PM_US Post'],bins=
                   (0,50,100,150,200,300,400) , 
                   labels=labels)
f = plt.figure(figsize=(12, 9))
plt.pie(df.groupby(['grp'])['PM_US Post'].sum(), labels=labels,autopct='%1.1f%%')
# show plot
plt.legend(loc='upper left',title="Levels of health concern", labels =['0-50: Good','51-100: Moderate','101-150: Unhealthy for Sensitive Groups', '151-200: Unhelalthy','201-300: Very Unhealthy','301+ :Hazardous'])
plt.show()

#%%
#Pri analizi kategoričkih obeležja, korisno je prikazati tzv. pivot tabele, odnosno tabele koje sumarno prikazuju broj uzoraka koji imaju određenu kombinaciju vrednosti dva obeležja
#Ovde vidimo u kojim sezonama se koji pravci vetrova se najčešće pojavljuju
pd.crosstab(df['season'],df['cbwd'])
sb.countplot(x='season',hue='cbwd',data=df)
plt.show()
#vidimo da je severoistočni pravac najzastupljeniji tokom jeseni i zime, 
#severozapadni tokom jeseni, jugozapadni tokom leta i jugoistočni tokom proleća

#%%
#U kojim satima u toku dana se javljaju koji vetrovi
pd.crosstab(df['hour'],df['cbwd'])
plt.figure(figsize=(10,5))
sb.countplot(x='hour',hue='cbwd',data=df)
plt.show()
#vidimo da se vetar kreće severoistočno najviše oko 9-10h, severozapadno oko 4h,  
#jugozapadno 13-14h i jugoistočno oko 17h

#%% II DEO: LINEARNA REGRESIJA

#izbacujemo ono obeležje koje predviđamo
#dfd-DataFrame sa dummy varijablama
x = dfd. drop(['PM_US Post'], axis = 1).copy() #vektor x sadrzi podatke o godini, danu, satu, sezoni, teperaturi kada je neka konc. cestice
#x- matrica uzorci puta obelezja/sva obel
#y-za te uzorke zeljeni izlazi/vrednosti pm
y = dfd['PM_US Post'].copy() #sadrzi pmcestice za sve uzorke
#zad lr regresije se svodi na pod.param teta pa kad nam dodje novi x mi ubacimo u jnu i vidimo izlaz
#biramo teta sve dok je zlaz i z fje
#malo po malo menjamo teta u pravcu opadanja gradijenta i na taj nacin dolazimo do idealnih vrednosti teta da smanji razl izmedju predvidjenog i izbacenog rez.
#%%

#pravimo test skup i trening skup
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, train_size=0.7, random_state=42)#prosl x i y
#u x train stavlja obel trening uzoraka u x test obelezja test uzoraka u y train izl za tr uzorke i z test izl za tr uzorke
#random_state - posto kakav god random u progr nisu stvarno slucajni brojevi nego jako duge sekvence 
#naizgled slucajne koje se ponavljaju, ako odabremo samo deo te sekv delovace kao da su nasumicni, kada stavimo neki broj 
#on ce uvek na isti nacin podeliti trening i test, kad promenimo stavlja druge uzorke u test i trening
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15, random_state=42)
#%%
#funkcija koja računa različite mere uspešnosti regresora, a koja će biti korišćena nakon svake obuke i testiranja da bi se utvrdilo koji je model najbolji
def model_evaluation(y, y_predicted, N, d): 
    mse = mean_squared_error(y_test, y_predicted) #prosek kvadrata odstupanje predvidjene od prave vrednosti
    #mse je nezgodna za interp. pr reska modela 25km na kv. nista nam ne znaci
    mae = mean_absolute_error(y_test, y_predicted) #apsolutno odsutpanje -žž-
    rmse = np.sqrt(mse) #koren iz mse, laksi za interpretaciju
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)#koristi se kad hocemo da uporedimo model koji su trenirani na
    #razl brojem uzoraka/obel 

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    # Uporedni prikaz nekoliko pravih i predvidjenih vrednosti
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
#%%
#Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)  
#fit true znaci postoji teta 0 u obuci ako je false hip je bez teta 0
# Obuka
first_regression_model.fit(x_train, y_train) #posto je nadgledano prosl se i y train

# Testiranje
y_predicted = first_regression_model.predict(x_test)
#obucen model iskoriscavamo na novim podacima, tj predvidjamo na osnovu x testakoji su izlazni

# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1]) #koliko uzoraka i obel zbog ajd r2

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
#iz obucenog modela  trazim da vidim koji su nauceni koef. i iscrta
#ako su koef blizu nuli znaci da to obelezje ne doprinosi puno ovom sto se prevdidja
#ako su vrednosti dosta vece a negativne to je manja koncentr. i obrnuto, ako je poz. onda je slicno
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

#po y i y_pred vidimo da model nije baš najbolje istreniran i da loše predviđa
#apsolutna greska od 24.98
#R2 score daleko od 1, što znaci da se pravi velika greška tj imamo jako veliku varijansu, i da lr nije mozda najbolje resenje za nas problem
#za MSE ako je npr 3 to znaci da ako moj model previdi neku vrednost to je uglavnom plus/minus tri u odnosu na tacnu vrednost

#%%
#SELEKCIJA UNAZAD
#krecem tako sto napr jedan model koji ima sva obel i proveravam onda koje je obel najmanje bitno
#to radimo tako sto za svaki param tetai se utvrdi interval poverenja- taj interval kaze da se tacna vred teta nalazi u tom opsegu 95%
#ako mi je p manje od 0.01 to znaci da je ver da je testirana hipoteza ispravna manja od 1%
#tj. da je tetai 0 tj da vrv tetai ne bi trebalo da bude 0, bolji rez bi bio da nije
#0.2 znaci 20 posto sansa da je hip ispravna, to je velika ver pa je pametno da to obelezje uz tetai bude izbaceno

import statsmodels.api as sm
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()  #ordinary list square- najobicnija lr
#u sc nema impl. ovo pa ovo koristimo iz druge biblioteke
model.summary()
#%%
#gpostavljena ver. će biti 1%
#obeležja cbwd_NE, cbwd_SE imaju verovatnoće veće od 0.01 i njih možemo izbaciti

X = sm.add_constant(x_train.drop(['cbwd_NE','cbwd_SE'], axis=1))

model = sm.OLS(y_train, X.astype('float')).fit()
model.summary()
#%%
#pravimo test skup i trening skup
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, train_size=0.7, random_state=42)

#Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model = LinearRegression(fit_intercept=True)  

# Obuka
first_regression_model.fit(x_train, y_train)

# Testiranje
y_predicted = first_regression_model.predict(x_test)


# Evaluacija
model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1]) 

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

#i izbacivanjem obeležja model i dalje ne radi najbolje, tako da ova hipoteza ne radi najbolje
#%%
#preporuka je da se obel normalizuju pre upotrebe lr
#to ne dovodi do poboljsanja lr i modela,vec da ubrza konvergenciju metode opadanja gradijenta
# Standardizacija obelezja 
#svodjenje sr vr obel na 0 i stand devijacije na 1
scaler = StandardScaler() #inic.klasu standard scaler, prima par da l cemo oba ili jedno ili drugo
scaler.fit(x_train) #radi se samo na treningg skupu, racuna sr vr i sand dev od svakog obel
#na tr pod jer ne zelimo da tedt uzorci na bilo koji nacin uticu na nas model

x_train_std = scaler.transform(x_train) #kada izr param koji su sacuvani u scaler, onda vrsimo
#transf, tj preskaliramo sve uzrorke
x_test_std = scaler.transform(x_test) 

x_train_std = pd.DataFrame(x_train_std) #kada se ovo gore izvrsi dobijemo nesto sto nije dataframe pa namestamo dataframe
x_test_std = pd.DataFrame(x_test_std)

x_train_std.columns = list(x.columns) #i nazive
x_test_std.columns = list(x.columns)

x_train_std.head()
#%%
#sada sa standardizovanim obeležjima ponavljamo obuku modela

#Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
first_regression_model_std = LinearRegression(fit_intercept=True)  

# Obuka
first_regression_model_std.fit(x_train_std, y_train)

# Testiranje
y_predicted = first_regression_model_std.predict(x_test_std)


# Evaluacija
model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1]) 

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model_std.coef_)),first_regression_model_std.coef_)
plt.show()
print("koeficijenti: ", first_regression_model_std.coef_)

#standardizacija nije dovela do poboljšanja linearne regresije, što je i očekivano jer 
#ona generalno ne poboljšava linearnu regresiju već samo utiče na brzinu konvergenije algoritma opadanja gradijenta

#gledamo da li možda treba da upotrebimo interakciju između obeležja(ovo se radi ukoliko
#su neka obeležja korelisana, a u delu analize smo utvrdili da jesu)
#%%
#Lin.regresija sa drugačijom hipotezom
#ako dosadasnji rez daju lose rez, onda je ono sto jos mozemo da uradimo jeste da pretp da veza izmedju obelezja nije linearna
#onda radimo uopstenje linearnog modela, npr kvadriramo, mnozimo (pravimo nelin.transf) i mozemo
#i interakcije x1*x2, x1*x3 itd, to se koristi kad  korelacija postoji 
poly = PolynomialFeatures(interaction_only=True, include_bias=False) 
#pomocu polfeat pravimo nova obelezja, inter only ako je false pored stepena uzima i interakcije, ako je true uzima samo interakcije u obzir bez nelin ytrans.
#degree do kog stepena da uzima obel, po def je 2
#incl bias false da se ne bi odnosilo na teta 0
x_inter_train = poly.fit_transform(x_train_std) #fit definise oper, a trasnf pravi vred novih obelezja
x_inter_test = poly.transform(x_test_std)
pd.DataFrame(x_inter_train).head()
#sada imamo sve moguće kombinacije sva po dva od svih standardnih obeležja
print(poly.get_feature_names()) 
#%%
# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...

# Inicijalizacija
regression_model_inter = LinearRegression()

# Obuka modela
regression_model_inter.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_inter.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)
#svi parameti su malo bolji od prethonih tkd je
#ovaj model malo bolji
#%%
#pokušaćemo još da unapredimo dodavanjem kvadrata obeležja 
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)
#%%
# Linearna regresija sa hipotezom y=b0+b1x1+b2x2+...+bnxn+c1x1x2+c2x1x3+...+d1x1^2+d2x2^2+...+dnxn^2

# Inicijalizacija
regression_model_degree = LinearRegression()

# Obuka modela
regression_model_degree.fit(x_inter_train, y_train)

# Testiranje
y_predicted = regression_model_degree.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
plt.show()
print("koeficijenti: ", regression_model_degree.coef_)

#još malo smo poboljšali 
#ovde vidimo da imamo neke velike koeficijente, a velike vrednosti uvek treba sprečiti da bismo sprečili
#model da se nadprilagodi
#%%
#Ridge regresija

# Inicijalizacija
ridge_model = Ridge(alpha=5) #alpha-regulariz parametar, sto je veci ima veci uticaj

# Obuka modela 
ridge_model.fit(x_inter_train, y_train)#prosl. poslednja dobijena obelezja

# Testiranje
y_predicted = ridge_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)
#ovde vidimo da su vrednosti koeficijenta dosta ravnomernije, ali nismo puno poboljšali prosečne mere
#svejedno model je najbolji jer nemamo velike koeficijente
#ako koeficijenti nisu ravnomerni, tj ako imamo neki prevelik koef nas model se previse prilagodio trening uzorcima tj gleda samo to jedno obel
# a otala zanemaruje, pa to moramo spreciti
#%%
#Lasso regresija

# Inicijalizacija
lasso_model = Lasso(alpha=0.01) #kod lasso regresije za alfu se obično postavljaju manji brojevi 

# Obuka modela 
lasso_model.fit(x_inter_train, y_train)

# Testiranje
y_predicted = lasso_model.predict(x_inter_test)

# Evaluacija
model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

#dešavanja sa koef-grafički

plt.figure(figsize=(10,5))
plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') 
plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='best')
plt.show()

#%% III DEO: KNN KLASIFIKATOR :(

#%%
df.drop(['grp'], axis =1)
#%%
#dodela labeli
labels = ['bezbedno', 'nebezbedno', 'opasno']
df['Nivo bezbednosti'] = pd.cut(df['PM_US Post'],bins=
                   (0,55.4,150.4,550) , 
                   labels=labels)
df.head()
#%%
#Ovo je definisana fja koja racuna mere uspesnosti klasifikatora
def evaluation_classifier(conf_mat):
    
    TP = conf_mat[1, 1]
    TN = conf_mat[0, 0]
    FP = conf_mat[0, 1]
    FN = conf_mat[1, 0]
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    F_score = 2*precision*sensitivity/(precision+sensitivity)
    
    print('precision: ', precision)
    print('accuracy: ', accuracy)
    print('sensitivity/recall: ', sensitivity)
    print('specificity: ', specificity)
    print('F score: ', F_score)
#%%podela skupa podataka
X = df.iloc[:, -1:].copy() # obelezja
y = df.iloc[:, -1].copy() # labele
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 5, stratify = y)
#%%
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#%%

