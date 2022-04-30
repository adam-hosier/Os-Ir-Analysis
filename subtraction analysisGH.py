import copy
from os import execve
import numpy as np
import csv 
import math as m 
import pandas as pd 
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
from lmfit import minimize, Parameters, report_fit, Model, Parameter
from lmfit.models import GaussianModel
from lmfit.models import SplitLorentzianModel 
import matplotlib.ticker as tck
import matplotlib.pylab as pl
from matplotlib.offsetbox import AnchoredText
import sys
import glob

#plotpath = str('C:\\Users\\ahosi\\Desktop\\FileTrans4\\NewMethodPlots\\')
#plotpath = str('C:\\Users\\ahosi\\OneDrive\\Desktop\\TestPlots\\')
#style = 'diff'
style = 'reg'
r=7                               #number of data points used in analysis
rcen = 7
cs =  299792458                     #speed of light m/s
hp = 6.62607015*(10)**(-34)         #plancks constant 
JeV = 1/(1.602176634*10**(-19))     #Joules to eV
crit = 4                          #criteria for outlier detection (number of standard deviation)
bins = 10                           #number of bins used for histograms
offset1 = 1                     #offset in photon counts to avoid zeros for weighted fitting 
siglev1 = 1                         #number of std err used in final calc 
binrange = 0.002

JGOsNa = 7.4491
JGOsNae = 0.0013 
JGIrNa = 7.2948
JGIrNae = 0.0013

SapIrNa = 7.29662

DiptiIrMg = 7.507248
DiptiOsMg = 7.669836

SteveIrMg = 7.49448
SteveOsMg = 7.65694



theoendiff = 170.028145 - 166.5075315
theoendiffe = np.sqrt((0.0013)**2 + (0.0013)**2)

econv = 8065.543937            #conversion from inverse cm to eV 
ftranloc = str('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4')
#Desktop Files 
wavefile = str(ftranloc + '\\OsIrWavelengthCal5.csv')          ###Adams calibration, but seperated for each individual spectra
wavefile2 = str(ftranloc + '\\OsIrWavelengthCal6b.csv')     ###Adam's calibration (cal set basis)
#wavefile2 = str(ftranloc + '\\OsIrWavelengthCalSam.csv')    ###Sam's calibration (cal set basis)
sysfile = str(ftranloc + '\\Calibration6bSystematic_Uncertainty.csv')
visualbadfiles = str(ftranloc + '\\badshit.csv')
redDatatab = str(ftranloc + '\\OsIr_NucChargeRadius_ReducedDataTable3.csv')         #was 2 
#specNumtab = str('ftranloc + '\\OsIr_NucChargeRadius_SpectrumNo02.csv')
specNumtab = str(ftranloc + '\\OsIr_NucChargeRadius_SpectrumNo.csv')
dropTab = str(ftranloc + '\\OsIr_NucChargeRadius_SpectrumNo2.csv')


dfw = pd.read_csv(r""+wavefile)
dfvis = pd.read_csv(r""+visualbadfiles)
df = pd.read_csv(r""+redDatatab)
df2 = pd.read_csv(r""+specNumtab)
dfdrop = pd.read_csv(r""+dropTab)
df2copy = copy.deepcopy(df2)
dfwa = pd.read_csv(r""+wavefile2)
dfsys = pd.read_csv(r""+sysfile)

calibrationuncert = []

for i in range(len(dfdrop.columns)): 
    name = dfdrop.columns[i]
    dfwa.drop(dfwa.columns[dfwa.columns.str.contains(name, case=False)], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains(name, case=False)], axis=1, inplace=True)
    df2.drop(df2.columns[df2.columns.str.contains(name, case=False)], axis=1, inplace=True)


badOs = dfvis['Os'] 


for i in range(len(badOs)):
    name = str(badOs[i]) 
    dfwa.drop(dfwa.columns[dfwa.columns.str.contains(name, case=False)], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains(name, case=False)], axis=1, inplace=True)
    df2.drop(df2.columns[df2.columns.str.contains(name, case=False)], axis=1, inplace=True)

dfa = copy.deepcopy(df)
df2a = copy.deepcopy(df2)
dfwab = copy.deepcopy(dfwa)


#df = Ir
#dfa = Os
#dfb = Bg
#dfc =Ne

df.drop(df.columns[df.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
df.drop(df.columns[df.columns.str.contains('Os', case=False)], axis=1, inplace=True)    #CHANGE
df.drop(df.columns[df.columns.str.contains('BG', case=False)], axis=1, inplace=True)
dfw.drop(dfw.columns[dfw.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
dfw.drop(dfw.columns[dfw.columns.str.contains('Os', case=False)], axis=1, inplace=True)    #CHANGE
dfw.drop(dfw.columns[dfw.columns.str.contains('BG', case=False)], axis=1, inplace=True)
df2.drop(df2.columns[df2.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
df2.drop(df2.columns[df2.columns.str.contains('Os', case=False)], axis=1, inplace=True) #CHANGE
df2.drop(df2.columns[df2.columns.str.contains('BG', case=False)], axis=1, inplace=True)
df.head()
df2.head()
dfw.head()



dfa.drop(dfa.columns[dfa.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
dfa.drop(dfa.columns[dfa.columns.str.contains('Ir', case=False)], axis=1, inplace=True)    #CHANGE
dfa.drop(dfa.columns[dfa.columns.str.contains('BG', case=False)], axis=1, inplace=True)
dfwab.drop(dfwab.columns[dfwab.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
dfwab.drop(dfwab.columns[dfwab.columns.str.contains('Ir', case=False)], axis=1, inplace=True)    #CHANGE
dfwab.drop(dfwab.columns[dfwab.columns.str.contains('BG', case=False)], axis=1, inplace=True)
df2a.drop(df2a.columns[df2a.columns.str.contains('Ne', case=False)], axis=1, inplace=True)
df2a.drop(df2a.columns[df2a.columns.str.contains('Ir', case=False)], axis=1, inplace=True) #CHANGE
df2a.drop(df2a.columns[df2a.columns.str.contains('BG', case=False)], axis=1, inplace=True)
dfa.head()
df2a.head()
dfwab.head()


##########################All data used for fitting 


xosna = []
yosna = []
xirna = []
yirna = []
xosmg = []
yosmg = []
xirmg = []
yirmg = []



calibset = []

#Lists for average of wavelength per calibration set 
OsNa = []
OsNae = []
OsMg = []
OsMge = []
IrNa = []
IrNae = []
IrMg = []
IrMge = []



sig = np.zeros(np.shape(df)[0],)
sig2 = np.zeros(np.shape(dfa)[0],)

sigIr = np.zeros(np.shape(df)[0],)
sigOs = np.zeros(np.shape(dfa)[0],)

#first order gauss
irna = []
osna = []
irnae = []
osnae = []
irnatime = []
osnatime = []

irmg = []
osmg = []
irmge = []
osmge = []
irmgtime = []
osmgtime = []

#first order centroid
osnacen = []
osmgcen = []
irnacen = []
irmgcen = []
osnacenu = []
osmgcenu = []
irnacenu = []
irmgcenu = []
osnacent = []
osmgcent = []
irnacent = []
irmgcent = []

#2nd order gauss
irna2 = []
osna2 = []
irnae2 = []
osnae2 = []
irnatime2 = []
osnatime2 = []

#centroid 2nd order
irna2c = []
osna2c = []
irnae2c = []
osnae2c = []
irnatime2c = []
osnatime2c = []

adsysunc = dfsys['after removal']

################################### Revised abs wavelength w/ set based calibration uncertainty 
##Gaussian fitting 
osnaset = []
osnasete = []
irnaset = []
irnasete = []
osmgset = []
osmgsete = []
irmgset = []
irmgsete = []

osnasetstat = []
irnasetstat = []
osmgsetstat = []
irmgsetstat = []

irnasetcal = []
irmgsetcal = []
osnasetcal = []
osmgsetcal =  []

##
osnaset2 = []
osnasete2 = []
irnaset2 = []
irnasete2 = []
osnasetstat2 = []
irnasetstat2 = []
osnasetcal2 = []
irnasetcal2 = []

#####
##Centroid 
cosnaset = []
cosnasete = []
cirnaset = []
cirnasete = []
cosmgset = []
cosmgsete = []
cirmgset = []
cirmgsete = []

cosnasetstat = []
cirnasetstat = []
cosmgsetstat = []
cirmgsetstat = []

cirnasetcal = []
cirmgsetcal = []
cosnasetcal = []
cosmgsetcal =  []

##
cosnaset2 = []
cosnasete2 = []
cirnaset2 = []
cirnasete2 = []
cosnasetstat2 = []
cirnasetstat2 = []
cosnasetcal2 = []
cirnasetcal2 = []

#####################################
#Quad gaussian fitting  

qirnastat = []
qosnastat = []
qirmgstat = []
qosmgstat = []

qosnaset = []
qosnasete = []
qirnaset = []
qirnasete = []
qosmgset = []
qosmgsete = []
qirmgset = []
qirmgsete = []

qosnasetstat = []
qirnasetstat = []
qosmgsetstat = []
qirmgsetstat = []

qirnasetcal = []
qirmgsetcal = []
qosnasetcal = []
qosmgsetcal =  []

########
#cal set based centroid measurements (summing spectra in single calibration set)
csetosna = []
csetosnae = []
csetirna = []
csetirnae = []
csetosmg = []
csetosmge = []
csetirmg = []
csetirmge = []

def gauss(x, A, mu, sig, B):
    return A * np.exp(-(x-mu)**2 / (2 * sig**2)) + B

def gauss2(x, A, mu, A2, mu2, sig, B):
    return B + (A*np.exp(-(x-mu)**2 / (2 * sig**2)) + A2*np.exp(-(x-mu2)**2 / (2*sig**2))) 

#def gauss4(x, A, mu, sig, A2, mu2, A3, mu3, A4, mu4):
 #   return A*np.exp(-(x-mu)**2 / (2*sig**2)) + A2*np.exp(-(x-mu2)**2 / (2*sig**2)) + A3*np.exp(-(x-mu3)**2 / (2*sig**2)) + A4*np.exp(-(x-mu4)**2 / (2*sig**2))

def gauss4(x, A, mu, sig, A2, mu2, A3, mu3, A4, mu4, K):
    return K + A*np.exp(-(x-mu)**2 / (2*sig**2)) + A2*np.exp(-(x-mu2)**2 / (2*sig**2)) + A3*np.exp(-(x-mu3)**2 / (2*sig**2)) + A4*np.exp(-(x-mu4)**2 / (2*sig**2))

def centroid(x,y, yerr):
    x = np.array(x)
    if np.min(y) < 0:
        y = np.array(y+np.abs(np.min(y)))
    else:
        y = np.array(y)

    if yerr is None:
        ye =  np.sqrt(y)
    else: 
        ye = yerr
    cnum = 0
    cdem = 0
    o = 0
    res = {}
    for i in range(len(y)):

        o = y[i] * x[i]            
        cnum += o
        cdem += y[i]

    centroid = cnum / cdem
    res['centroid'] = centroid
    a1 = 0
    b1 = 0
    for i in range(len(y)):
        b1 += y[i]
        a1 += ((ye[i])**2)*((x[i] - centroid)**2)
    
    dcentroid = m.sqrt(a1 / (b1**2))
    res['cent_unc'] = dcentroid

    return res

def parts(x):

    arr = np.zeros([4,1])

    arr[0,0] = 1
    arr[1,0] = x 
    arr[2,0] = x**2 
    arr[3,0] = x**3


    return arr 

def dataResizer2(cenos, cenerros, cenir, cenerrir,  specir, specos):
    
    limit = 200
    resDat = dict()
    counter = 0

    

    try:
        cenos = np.reshape(cenos, (1, len(cenos)))
    except:
        cenos = cenos

    try:
        cenir = np.reshape(cenir, (1, len(cenir)))
    except:
        cenir = cenir



    osdatalen = np.shape(cenos)[1]
    irdatalen = np.shape(cenir)[1]



    diff = osdatalen - irdatalen

    while counter < diff:
        cenir = np.append(cenir, 'nan')
        
        cenerrir = np.append(cenerrir, 'nan')
        
        #specir = np.append(specir, 300+counter)
        specir = np.append(specir, 'nan')
        
        if counter == limit:
            break
        counter+=1
    
    if len(cenir) > 5:
        cenir = np.reshape(cenir, (1,len(cenir)))
        cenerrir = np.reshape(cenerrir, (1,len(cenerrir)))
        specir = np.reshape(specir, (1,len(specir)))
    
    cenir = cenir.astype(np.float64)
    cenerrir = cenerrir.astype(np.float64)
    specir = specir.astype(np.float64)
    
    resDat['cenir_rs'] = cenir
    resDat['cenerrir_rs'] = cenerrir
    resDat['specir_rs'] = specir 

    counter = 0
    diff = -osdatalen + irdatalen

    while counter < diff:
        cenos = np.append(cenos, 'nan')
        cenerros = np.append(cenerros, 'nan')
        #specos = np.append(specos, 300+counter)
        specos = np.append(specos, 'nan')
        counter+=1
        if counter == limit:
            break
    
    if len(cenos) > 5:
        cenos = np.reshape(cenos, (1,len(cenos)))
        cenerros = np.reshape(cenerros, (1,len(cenerros)))
        specos = np.reshape(specos, (1, len(specos)))

    cenos = cenos.astype(np.float64)
    cenerros = cenerros.astype(np.float64)
    specos = specos.astype(np.float64)
    
    resDat['cenos_rs'] = cenos
    resDat['cenerros_rs'] = cenerros
    resDat['specos_rs'] = specos 


    return resDat


def nankill0(arr):                          #checks if values in array == 'nan'
    indexlist = np.zeros([1,0])
    
    try:
        arrlen = np.shape(arr)[1]
    
    

        for i in range(arrlen):
            if arr[0,i] == 'nan' or arr[0,i] == np.nan:
                indexlist = np.append(indexlist,i)
            else:
                continue
        indexlist = np.around(indexlist)
        indexlist = indexlist.astype(int)
        
        arr = np.delete(arr, indexlist)
        newlen = len(arr)
        arr = np.reshape(arr, (newlen,))

    except:
        arrlen = len(arr)
        for i in range(arrlen):
            if arr[i] == 'nan' or arr[i] == np.nan:
                indexlist = np.append(indexlist,i)
            else:
                continue
        indexlist = np.around(indexlist)
        indexlist = indexlist.astype(int)
        
        arr = np.delete(arr, indexlist)
        newlen = len(arr)
        arr = np.reshape(arr, (newlen,))



    return arr


def nankill1(arr):                          #checks if values in array == 'nan'
    indexlist = np.zeros([1,0])
    
    try:
        arrlen = np.shape(arr)[1]
    
    

        for i in range(arrlen):
            if arr[0,i] == 'nan' or arr[0,i] == np.nan:
                indexlist = np.append(indexlist,i)
            else:
                continue
        indexlist = np.around(indexlist)
        indexlist = indexlist.astype(int)
        
        arr = np.delete(arr, indexlist)
        newlen = len(arr)
        arr = np.reshape(arr, (1,newlen))

    except:
        arrlen = len(arr)
        for i in range(arrlen):
            if arr[i] == 'nan' or arr[i] == np.nan:
                indexlist = np.append(indexlist,i)
            else:
                continue
        indexlist = np.around(indexlist)
        indexlist = indexlist.astype(int)
        
        arr = np.delete(arr, indexlist)
        newlen = len(arr)
        arr = np.reshape(arr, (1,newlen))



    return arr


def Histogram(res, bins, bin_range, num):

    binneddata = np.histogram(res, bins=bins, range=[-bin_range, bin_range])         #binned data for centroid
    
    Res = dict()

    yuncert = np.zeros([1, len(binneddata[0][:])])

    uncerterr = np.zeros([1, len(binneddata[0][:])])
    freq = np.zeros([1, len(binneddata[0][:])])
    totalobs = 0

    for i in range(len(binneddata[1][:])-1):
        
        yuncert[0,i] = (1/2) * (binneddata[1][i] + binneddata[1][i+1])

        if binneddata[0][i] == 0:
            uncerterr[0,i] = np.nan
        else:
            uncerterr[0,i] = 1 / np.sqrt(binneddata[0][i])

        
        freq[0,i] = (binneddata[0][i])

        totalobs += binneddata[0][i] 


    Res['totalobs'] = totalobs 


    def gaussfun(x, A, mu, sig):
        return A * np.exp((-(x-mu)**2)/ (2. * sig**2))

    mod1 = Model(gaussfun)
    params = Parameters() 
    params.add('A', value=1, min=0)
    params.add('mu', value = 0)
    params.add('sig', value=0.01, min=0)


    errfit = mod1.fit(freq, params, x=yuncert, nan_policy='omit', weights = uncerterr)  #weights = uncerterr
    params.update(errfit.params)


    xplot = np.linspace(np.min(yuncert), np.max(yuncert), num=500)
    yplot = mod1.eval(params=params, x=xplot)
    Res['xplot'] = xplot
    Res['yplot'] = yplot
    Res['xdata'] = yuncert 
    Res['ydata'] = freq 


    xgraph = np.linspace(-np.max(yuncert), np.max(yuncert), num=1000)

    values1 = params.valuesdict()

    cen1 = values1['mu']

    ###################################Graphing
    
    resideval = errfit.eval(params, x=xgraph)

    ####################################
    calcparams = params.valuesdict()

    lineposerroros = calcparams['sig'] / np.sqrt(num)
    Res['Line Position Error'] = lineposerroros
    Res['sigma']= calcparams['sig']
    Res['ABS Line Position Error'] = calcparams['sig'] / np.sqrt(totalobs)
    Res['calc params'] = params
    Res['mu stderr'] = params['mu'].stderr
    Res['mu val'] = params['mu'].value

    return Res


def poly(x, a, b, c, d):

    return (a*x**3 + b*x**2 + c*x + d)



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def OutlierDet(osna, Oserr, osnatime, irna, Irerr, irnatime, poly, crit):       #single point removal

    osna = osna.astype(np.float64)
    Oserr = Oserr.astype(np.float64)
    osnatime = osnatime.astype(np.float64)
    irna = irna.astype(np.float64)
    Irerr = Irerr.astype(np.float64)
    irnatime = irnatime.astype(np.float64)


    Osnewres = np.zeros([1,])
    Irnewres = np.zeros([1,])
    sigO = np.zeros([1,])
    sigI = np.zeros([1,])
    #Ircomp = np.zeros([1,])
    #Oscomp = np.zeros([1,])
    Oscomp = []
    Ircomp = []

    for i in range(np.shape(osna)[0]):

        arr = copy.deepcopy(osna)            #making copy of arrays to avoid altering the original arrays
        arrerr = copy.deepcopy(Oserr)
        Ostim = copy.deepcopy(osnatime)

        arr2 = np.delete(arr, i)             #deleting ith row in the data (rotating through all data for single removal)
        arrerr2 = np.delete(arrerr, i)
        tim2 = np.delete(Ostim, i)

        arr2 = np.append(arr2, np.nan)
        arrerr2 = np.append(arrerr2, np.nan)
        tim2 = np.append(tim2, np.nan)


        newfit = poly(tim2, arr2, arrerr2, irnatime, irna, Irerr)
        #print('Ostim: ', np.shape(Ostim))
        #try:
        val = newfit['d_1'] + newfit['c_1']*(Ostim[i]) + newfit['b_1']*(Ostim[i])**2 + newfit['a_1']*(Ostim[i])**3 
        #except:
            #val = newfit['d_1'] + newfit['c_1']*(Ostim[0,i]) + newfit['b_1']*(Ostim[0,i])**2 + newfit['a_1']*(Ostim[0,i])**3 

        #arr2 = np.reshape(arr2, (1,len(arr2)))
        #tim2 = np.reshape(tim2, (1,len(tim2)))
        #print(np.shape(arr2))
        arr2 = nankill0(arr2)
        tim2 = nankill0(tim2) 
        #print(np.shape(arr2))
        resiO = arr - (newfit['d_1'] + newfit['c_1']*Ostim + newfit['b_1']*Ostim**2 + newfit['a_1']*Ostim**3)

        OHis = Histogram(resiO, bins=bins, bin_range=(np.average(np.abs(resiO))*1.5), num=newfit['num'])

        sigO = np.append(sigO, OHis['sigma'])
        Osnewres = np.append(Osnewres, np.abs(arr[i]-val))
        #Oscomp.append((np.abs(arr[0,i] - val) / OHis['sigma']))
        newval = np.abs(arr[i]-val)/ OHis['sigma']
        #Oscomp = np.concatenate((Oscomp, newval))
        Oscomp.append(newval)
        #Oscomp2.append((np.abs(arr[0,i] - val) / OHis['sigma']))


    for i in range(np.shape(irna)[0]):
        arr = copy.deepcopy(irna)            #making copy of arrays to avoid altering the original arrays
        arrerr = copy.deepcopy(Irerr)
        Irtim = copy.deepcopy(irnatime)

        arr2 = np.delete(arr, i)             #deleting ith row in the data (rotating through all data for single removal)
        arrerr2 = np.delete(arrerr, i)
        tim2 = np.delete(Irtim, i)

        arr2 = np.append(arr2, np.nan)
        arrerr2 = np.append(arrerr2, np.nan)
        tim2 = np.append(tim2, np.nan)
        
        newfit = poly(osnatime, osna, Oserr, tim2, arr2, arrerr2)
        #arr2 = np.reshape(arr2, (1,len(arr2)))
        #tim2 = np.reshape(tim2, (1,len(tim2)))
        arr2 = nankill0(arr2)
        tim2 = nankill0(tim2)

        val = newfit['d_2'] + newfit['c_2']*(Irtim[i]) + newfit['b_2']*(Irtim[i])**2 + newfit['a_2']*(Irtim[i])**3 
        resiI = arr - (newfit['d_2'] + newfit['c_1']*Irtim + newfit['b_1']*Irtim**2 + newfit['a_1']*Irtim**3)

        IHis = Histogram(resiI, bins=bins, bin_range=(np.average(np.abs(resiI))*1.5), num=newfit['num'])

        sigI = np.append(sigI, IHis['sigma'])
        Irnewres = np.append(Irnewres, np.abs(arr[i]-val))
        newval = [np.abs(arr[i] - val) / IHis['sigma']]
        #Ircomp = np.concatenate((Ircomp, newval))
        Ircomp.append((np.abs(arr[i] - val) / IHis['sigma']))


    Osmaxres = np.max(Oscomp)
    Osmaxin = np.argmax(Oscomp)
    Irmaxres = np.max(Ircomp)
    Irmaxin = np.argmax(Ircomp) 


    if Osmaxres > Irmaxres:
        if Osmaxres >= crit:            #rejection criteria based on 
            #try:
            badlines = df2copy.columns[df2copy.eq(str(Ostim[Osmaxin])).any()][0] 
            #except:
            #    badlines = df2copy.columns[df2copy.eq(str(Ostim[0,Osmaxin])).any()][0]           
            newosna = np.delete(osna, Osmaxin)
            newOserr = np.delete(Oserr, Osmaxin)
            newosnatime = np.delete(osnatime, Osmaxin)
            newirna = irna 
            newIrerr = Irerr
            newirnatime = irnatime 

        else: #(Osmaxres/(newfit['Osconfband'][0,np.around(Ostim[0,Osmaxin]).astype(int)])) < crit:
            badlines = None
            newosna = osna
            newOserr = Oserr
            newosnatime = osnatime
            newirna = irna
            newIrerr = Irerr 
            newirnatime = irnatime 


            newfit = poly(osnatime, osna, Oserr, irnatime, irna, Irerr) 
    
        

    if Irmaxres > Osmaxres:
        if Irmaxres >= crit:            #rejection criteria based on 
            
            #try:
            badlines = df2copy.columns[df2copy.eq(str(Irtim[Irmaxin])).any()][0]   #only the distance from the confidenceband
            
            #except:
            #    badlines = df2copy.columns[df2copy.eq(str(Irtim[0,Irmaxin])).any()][0]
            newirna = np.delete(irna, Irmaxin)
            newIrerr = np.delete(Irerr, Irmaxin)
            newirnatime = np.delete(irnatime, Irmaxin)
            newosna = osna 
            newOserr = Oserr 
            newosnatime = osnatime 

        else: #(Irmaxres/(newfit['Irconfband'][0,np.around(Irtim[0,Irmaxin]).astype(int)])) < crit:

            badlines = None
            newirna = irna
            newIrerr = Irerr
            newirnatime = irnatime
            newosna = osna 
            newOserr = Oserr 
            newosnatime = osnatime 
            newfit = poly(osnatime, osna, Oserr, irnatime, irna, Irerr)   
        

    #print(badlines)

    ODres = {}
    ODres['osna'] = newosna
    ODres['Oserr'] = newOserr
    ODres['osnatime'] = newosnatime
    ODres['irna'] = newirna 
    ODres['Irerr'] = newIrerr 
    ODres['irnatime'] = newirnatime 

    ODres['badlines'] = badlines
    ODres['syspoly'] = poly(newosnatime, newosna, newOserr, newirnatime, newirna, newIrerr)
    
    ODres['a_1'] = newfit['a_1']
    ODres['b_1'] = newfit['b_1']
    ODres['c_1'] = newfit['c_1'] 
    ODres['d_1'] = newfit['d_1']
    ODres['d_2'] = newfit['d_2'] 
    ODres['chi-sq'] = newfit['chi-sq']
    ODres['red-chi-sq'] = newfit['red-chi-sq']
    
    return ODres



def quadgfit(x, y, r, c, num):

    resdic = dict()
    wave = []
    sig = []
    for i in range(1, 2*r):
        wave.append(x[c-r+i])
        sig.append(y[c-r+i])

    mod = Model(gauss4)
    params = Parameters() 
    params.add('A', value = np.max(sig)-np.average(sig), min=0)
    params.add('mu', value = 7.29, min=7.26, max=7.32)
    params.add('sig', value = 0.01, min=0)

    params.add('A2', value = np.min(sig)-np.average(sig), max=0)
    params.add('mu2', value = 7.44, min=7.43, max=7.46)

    params.add('A3', value = np.max(sig)-np.average(sig), min=0)
    params.add('mu3', value = 7.49, min=7.46, max=7.51)

    params.add('A4', value = np.min(sig)-np.average(sig), max=0)
    params.add('mu4', value = 7.63, min=7.61, max=7.67)

    params.add('K', value = np.average(sig))

    res = mod.fit(sig, params=params, x=wave, weights = np.sqrt(np.abs(sig)), nan_policy='omit')
    params.update(res.params)

    xplot = np.linspace(np.min(wave), np.max(wave), 1000)
    yplot = res.eval(params=params, x=xplot)

    sumphot = np.sum(np.abs(sig))



    resdic['res'] = res
    resdic['params'] = params
    resdic['xplot'] = xplot
    resdic['yplot'] = yplot 
    resdic['xdat'] = wave
    resdic['ydat'] = sig 
    resdic['sumphot'] = sumphot

    if num==1: 
        resdic['MU'] = params['mu'].value
        resdic['MU_err'] = params['mu'].stderr
    elif num==2:
        resdic['MU'] = params['mu2'].value
        resdic['MU_err'] = params['mu2'].stderr
    elif num==3:
        resdic['MU'] = params['mu3'].value
        resdic['MU_err'] = params['mu3'].stderr
    elif num==4:
        resdic['MU'] = params['mu4'].value
        resdic['MU_err'] = params['mu4'].stderr

    return resdic 


def datcoll(y, x, c, r):  #in single spectra loop
    res = dict()

    ydat = []   
    ydate = []                   
    wavedat = []
    pix = []
    bg = 0                  #background outside of radius of data 
    photc = []                  #photon count

    #Collecting data around a point of interest defined by 'c' and 'r' above, along with centroid calcs
    for i in range(1,2*r):
        ydat.append(y[c - r + i])
        wavedat.append(x[c-r+i])
        pix.append(c-r+i)
   
    ydat = np.array(ydat)
    ydate = np.sqrt(ydat)
    wavedat = np.array(wavedat)
    pix = np.array(pix) 

    res['y'] = ydat
    res['ye'] = ydate 
    res['x'] = pix 
    res['wave'] = wavedat

    return res 

    


#pixel = np.linspace(0, 2047, num=2048)
pixel = np.linspace(1, 2048, num=2048)
####uncertainty breakdown 

osna1cal = []
osna1sys = []
osna1stat = []

osna2cal = []
osna2sys = []
osna2stat = []

irna1cal = []
irna1stat = []
irna1sys = []

irna2cal = []
irna2stat = []
irna2sys = []

osmg1cal = []
osmg1stat = []
osmg1sys = []


irmg1cal = []
irmg1stat = []
irmg1sys = []


osnastat1ori = []


########

tsum1 = []
ttime1 = []

tsum1os = []
ttime1os = []

tsumosna = []
tsumosmg = []
tsumirna = []
tsumirmg = []

ttimeosna = []
ttimeosmg = []
ttimeirna = []
ttimeirmg = []




for v in range(1, 11):
    v = 2*v
    
    #v = 2*v-1
    tval2 = 2.201
    if v==15 or v==16:
        continue
    cIr = 0
    fram = 0
    fram2 = 0
    framIr = 0
    framOs = 0

    ############################## 4-gauss fit over 1 normalized calibration set. 


    for k in range(np.shape(df)[1]):        #Ir

        if int(df2.iloc[2,k][3:]) == int(v):

            wave = dfwa[df2.iloc[2,k]]
            sig += df.iloc[:,k] #/ int(df2.iloc[4,k]) 
            fram += int(df2.iloc[4,k])
            calstring = df2.iloc[2,k]

            cIr +=1

    sigtestIr = sig
    energy = JeV*hp*cs / (wave * 10**(-9))
    photc = ((3 * sig))/ (energy / 3.6) 
    photc = photc /(cIr * fram)   
    sig = sig/fram      #normalized average Ir signal for a given calibration set 
    
    cOs = 0
    for b in range(np.shape(dfa)[1]):           #Os

        if int(df2a.iloc[2,b][3:]) == int(v):

            wave2 = dfwab[df2a.iloc[2,b]]
            sig2 += dfa.iloc[:,b] #/ 
            fram2 += int(df2a.iloc[4,b])
            calstring2 = df2a.iloc[2,b]
            cOs +=1

    sigtestOs = sig2
    energy2 = JeV*hp*cs / (wave2 * 10**(-9))
    photc2 = ((3 * sig2))/ (energy2 / 3.6) 
    photc2 = photc2 / (cOs * fram2)

    sig2 = sig2 / fram2             #normalized average Os spectra for a given calibration set 
    difference = photc - photc2        ###  Ir minus Os in photon counts
    difference2 = sig/(cIr) - sig2/(cOs)            ### Ir minus Os in ADU signal 
    difference3 = sigtestIr/cIr - sigtestOs/cOs
    phottest = ((3* difference3))/(energy/3.6)
    
    for i in range(len(phottest)):
        if phottest[i]<0:
            phottest[i] = phottest[i]*cOs
        else: 
            phottest[i] = phottest[i]*cIr

    phottestos = ((3*sigtestOs))/(energy/3.6)
    phottestir = ((3*sigtestIr))/(energy/3.6)
    # plt.figure() 
    # # plt.plot(pixel, sigtestOs, label='Os',c='r')
    # # plt.plot(pixel, sigtestIr, label='Ir',c='b')
    # # plt.xlabel('Pixel')
    # # plt.xlim(np.min(pixel), np.max(pixel))
    # # plt.ylabel('ADU')
    # #plt.plot(wave, difference3, label='Ir-Os')
    # plt.plot(wave, phottest, label='Ir-Os')
    # plt.xlabel('wavelength (nm)')
    # plt.ylabel('Photon counts')
    # plt.legend()
    # plt.show() 
    # plt.close() 


    #cal set centroid calculations for subtracted spectra
    osnadat = datcoll(np.abs(phottest), wave, c=566, r=rcen)
    osmgdat = datcoll(np.abs(phottest), wave, c=597, r=rcen)
    irnadat = datcoll(np.abs(phottest), wave, c=544, r=rcen)
    irmgdat = datcoll(np.abs(phottest), wave, c=574, r=rcen)

    osnacen1 = centroid(osnadat['wave'], osnadat['y'], yerr=None)
    osmgcen1 = centroid(osmgdat['wave'], osmgdat['y'], yerr=None)
    irnacen1 = centroid(irnadat['wave'], irnadat['y'], yerr=None)
    irmgcen1 = centroid(irmgdat['wave'], irmgdat['y'], yerr=None)

    osnadat2 = datcoll(np.abs(phottestos), wave, c=566, r=rcen)
    osmgdat2 = datcoll(np.abs(phottestos), wave, c=597, r=rcen)
    irnadat2 = datcoll(np.abs(phottestir), wave, c=544, r=rcen)
    irmgdat2 = datcoll(np.abs(phottestir), wave, c=574, r=rcen)

    osnacen2 = centroid(osnadat2['wave'], osnadat2['y'], yerr=None)
    osmgcen2 = centroid(osmgdat2['wave'], osmgdat2['y'], yerr=None)
    irnacen2 = centroid(irnadat2['wave'], irnadat2['y'], yerr=None)
    irmgcen2 = centroid(irmgdat2['wave'], irmgdat2['y'], yerr=None)


    ####################### calibration uncertainty determination from dispersion relation

    #Na-like centers 
    # centerIr = 544
    # centerOs = 566

    #Na-like 2nd order centers
    centerIr = 1480
    centerOs = 1515
    #Mg-like centers 
    #centerIr = 574
    #centerOs = 597
    testIr = []
    testOs = []
    pOs = []
    pIr = []

    for i in range(1, 2*r):
        testIr.append(sigtestIr[centerIr-r+i])
        testOs.append(sigtestOs[centerOs-r+i])
        pOs.append(pixel[centerOs-r+i])
        pIr.append(pixel[centerIr-r+i])

    modtestIr = Model(gauss)
    paramstestIr = Parameters()
    paramstestIr.add('A', value = np.max(testIr)-np.min(testIr), min=0)
    paramstestIr.add('mu', value=np.mean(pIr))
    paramstestIr.add('sig', value = 0.001, min=0)
    paramstestIr.add('B', value = np.min(testIr))
    resIrtest = modtestIr.fit(testIr, params=paramstestIr, x=pIr, max_nfev=None, nan_policy='omit')
    paramstestIr.update(resIrtest.params)

    
    modtestOs = Model(gauss)
    paramstestOs = Parameters() 
    paramstestOs.add('A', value = np.max(testOs)-np.min(testOs), min=0)
    paramstestOs.add('mu', value=np.mean(pOs))
    paramstestOs.add('sig', value = 0.001, min=0)
    paramstestOs.add('B', value = np.min(testOs))
    resOstest = modtestOs.fit(testOs, params=paramstestOs, x=pOs, max_nfev=None, nan_policy='omit')
    paramstestOs.update(resOstest.params)


    ostestx = np.rint(paramstestOs['mu'].value)
    irtestx = np.rint(paramstestIr['mu'].value)

    testrang = np.linspace(irtestx, ostestx, num=(int(ostestx)-int(irtestx)+int(1)))

    calteststr = 'Cal'+str(v)

    for name in glob.glob(ftranloc+'\\Calibration6b\\'+calstring+'\\*PolyParameters.csv'):
        
        nam = name

    testdf = pd.read_csv(r""+nam)

    bet1 = testdf['Uncertainty'][1]
    gam1 = testdf['Uncertainty'][2]
    omeg1 = testdf['Uncertainty'][3]

    def Funcert(x, beta, gamma, omega):
        val = np.sqrt(beta**2 + (2*gamma*x)**2 + (3*omega*x**2)**2)
        sumfun = np.sum(val)
        return sumfun

    
    #estcalunc = (-1)*Funcert(testrang, bet1, gam1, omeg1)
    estcalunc = 0
    #########################################################

    #### New calibration uncertainty determination for absolute wavelength on set basis
    #Na-like centers 
    nacenterIr = 544
    nacenterOs = 566
    nacenterIr2 = 1480
    nacenterOs2 = 1515

    #Mg-like centers 
    mgcenterIr = 574
    mgcenterOs = 597

    natestIr = []
    natestOs = []
    mgtestIr = []
    mgtestOs = []
    natestIr2 = []
    natestOs2 = []
    napOs = []
    napIr = []
    mgpOs = []
    mgpIr = []
    napOs2 = []
    napIr2 = []


    for i in range(1, 2*r):
        natestIr.append(sigtestIr[nacenterIr-r+i])
        natestOs.append(sigtestOs[nacenterOs-r+i])
        napOs.append(pixel[nacenterOs-r+i])
        napIr.append(pixel[nacenterIr-r+i])

        mgtestIr.append(sigtestIr[mgcenterIr-r+i])
        mgtestOs.append(sigtestOs[mgcenterOs-r+i])
        mgpOs.append(pixel[mgcenterOs-r+i])
        mgpIr.append(pixel[mgcenterIr-r+i])

        natestIr2.append(sigtestIr[nacenterIr2-r+i])
        natestOs2.append(sigtestOs[nacenterOs-r+i])
        napIr2.append(pixel[nacenterIr2-r+i])
        napOs2.append(pixel[nacenterOs2-r+i])


    namodtestIr = Model(gauss)
    naparamstestIr = Parameters()
    naparamstestIr.add('A', value = np.max(natestIr)-np.min(natestIr), min=0)
    naparamstestIr.add('mu', value=np.mean(napIr))
    naparamstestIr.add('sig', value = 0.001, min=0)
    naparamstestIr.add('B', value = np.min(natestIr))
    naresIrtest = namodtestIr.fit(natestIr, params=naparamstestIr, x=napIr, max_nfev=None, nan_policy='omit')
    naparamstestIr.update(naresIrtest.params)

    
    namodtestOs = Model(gauss)
    naparamstestOs = Parameters() 
    naparamstestOs.add('A', value = np.max(natestOs)-np.min(natestOs), min=0)
    naparamstestOs.add('mu', value=np.mean(napOs))
    naparamstestOs.add('sig', value = 0.001, min=0)
    naparamstestOs.add('B', value = np.min(natestOs))
    naresOstest = namodtestOs.fit(natestOs, params=naparamstestOs, x=napOs, max_nfev=None, nan_policy='omit')
    naparamstestOs.update(naresOstest.params)

    mgmodtestIr = Model(gauss)
    mgparamstestIr = Parameters()
    mgparamstestIr.add('A', value = np.max(mgtestIr)-np.min(mgtestIr), min=0)
    mgparamstestIr.add('mu', value=np.mean(mgpIr))
    mgparamstestIr.add('sig', value = 0.001, min=0)
    mgparamstestIr.add('B', value = np.min(mgtestIr))
    mgresIrtest = mgmodtestIr.fit(mgtestIr, params=mgparamstestIr, x=mgpIr, max_nfev=None, nan_policy='omit')
    mgparamstestIr.update(mgresIrtest.params)

    
    mgmodtestOs = Model(gauss)
    mgparamstestOs = Parameters() 
    mgparamstestOs.add('A', value = np.max(mgtestOs)-np.min(mgtestOs), min=0)
    mgparamstestOs.add('mu', value=np.mean(mgpOs))
    mgparamstestOs.add('sig', value = 0.001, min=0)
    mgparamstestOs.add('B', value = np.min(mgtestOs))
    mgresOstest = mgmodtestOs.fit(mgtestOs, params=mgparamstestOs, x=mgpOs, max_nfev=None, nan_policy='omit')
    mgparamstestOs.update(mgresOstest.params)


    ###
    namodtestIr2 = Model(gauss)
    naparamstestIr2 = Parameters()
    naparamstestIr2.add('A', value = np.max(natestIr2)-np.min(natestIr2), min=0)
    naparamstestIr2.add('mu', value=np.mean(napIr2))
    naparamstestIr2.add('sig', value = 0.001, min=0)
    naparamstestIr2.add('B', value = np.min(natestIr2))
    naresIrtest2 = namodtestIr.fit(natestIr2, params=naparamstestIr2, x=napIr2, max_nfev=None, nan_policy='omit')
    naparamstestIr2.update(naresIrtest2.params)

    
    namodtestOs2 = Model(gauss)
    naparamstestOs2 = Parameters() 
    naparamstestOs2.add('A', value = np.max(natestOs2)-np.min(natestOs2), min=0)
    naparamstestOs2.add('mu', value=np.mean(napOs2))
    naparamstestOs2.add('sig', value = 0.001, min=0)
    naparamstestOs2.add('B', value = np.min(natestOs2))
    naresOstest2 = namodtestOs.fit(natestOs2, params=naparamstestOs2, x=napOs2, max_nfev=None, nan_policy='omit')
    naparamstestOs2.update(naresOstest2.params)

    ###


    calteststr = 'Cal'+str(v)

    for name in glob.glob(ftranloc+'\\Calibration6b\\'+calstring+'\\*CovarianceMatrix.csv'):
        
        nam = name

    tempdf = pd.read_csv(r""+nam)
    coma = np.array(tempdf)

    ###first order 
    colvecnaos = parts(naresOstest.params['mu'].value)
    temp1naos = np.matmul(coma, colvecnaos)
    tempnaos = tval2*np.sqrt(np.matmul(colvecnaos.T, temp1naos))
    caluncnaos = tempnaos[0,0]

    colvecnair = parts(naresIrtest.params['mu'].value)
    temp1nair = np.matmul(coma, colvecnair)
    tempnair = tval2*np.sqrt(np.matmul(colvecnair.T, temp1nair))
    caluncnair = tempnair[0,0]

    colvecmgos = parts(mgresOstest.params['mu'].value)
    temp1mgos = np.matmul(coma, colvecmgos)
    tempmgos = tval2*np.sqrt(np.matmul(colvecmgos.T, temp1mgos))
    caluncmgos = tempmgos[0,0]

    colvecmgir = parts(mgresIrtest.params['mu'].value)
    temp1mgir = np.matmul(coma, colvecmgir)
    tempmgir = tval2*np.sqrt(np.matmul(colvecmgir.T, temp1mgir))
    caluncmgir = tempmgir[0,0]

    ###2nd order 
    colvecnaos2 = parts(naresOstest2.params['mu'].value)
    temp1naos2 = np.matmul(coma, colvecnaos2)
    tempnaos2 = tval2*np.sqrt(np.matmul(colvecnaos2.T, temp1naos2))
    caluncnaos2 = tempnaos2[0,0]

    colvecnair2 = parts(naresIrtest2.params['mu'].value)
    temp1nair2 = np.matmul(coma, colvecnair2)
    tempnair2 = tval2*np.sqrt(np.matmul(colvecnair2.T, temp1nair2))
    caluncnair2 = tempnair2[0,0]
    ###

    ###Appending calibration set centroid calculations 
    # print(osnacen1)
    # print(irnacen1)
    # print(osmgcen1)
    # print(irmgcen1)
    csetosna.append(osnacen1['centroid'])
    csetosnae.append(np.sqrt(osnacen2['cent_unc']**2 + caluncnaos**2))
    csetirna.append(irnacen2['centroid'])
    csetirnae.append(np.sqrt(irnacen2['cent_unc']**2 + caluncnair**2))
    csetosmg.append(osmgcen2['centroid'])
    csetosmge.append(np.sqrt(osmgcen2['cent_unc']**2 + caluncmgos**2))
    csetirmg.append(irmgcen2['centroid'])
    csetirmge.append(np.sqrt(irmgcen2['cent_unc']**2 + caluncmgir**2))
    
    ######################### Individual spectra analysis (single gauss fit / centroid)
    tempval = []
    tempvale = []

    tempvalc = []
    tempvalec = []

    qtempval = []
    qtempvale = []


    for k in range(np.shape(df)[1]):        #Ir individual spectra(Na-like) 

        if int(df2.iloc[2,k][3:]) == int(v):

            wave = dfwa[df2.iloc[2,k]]
            sigIr = df.iloc[:,k] 
            framIr = int(df2.iloc[4,k])
            calstring = df2.iloc[2,k]

            sigIr = sigIr / framIr 
            diffIr = sigIr - sig2


            for i in range(len(diffIr)):
                if diffIr[i] < 0:
                    diffIr[i] = diffIr[i]*framIr
                else: 
                    diffIr[i] = diffIr[i]*fram2/cOs

            etest = hp*cs*JeV/(10**(-9)*wave)
            ptest = (3*diffIr)/(etest/3.6)

            ptest2 = ptest


            # plt.figure() 
            # plt.plot(wave, ptest)
            # plt.show()
            # plt.close() 

            irfit = []
            irfit2 = []
            irwave = []
            irpix = []

            irfitc = []
            irfitc2 = []
            irwavec = []
            irpixc = []

            pweight = []
            pweightc = []
            c=544


            if style == 'diff':
                for i in range(1, 2*r):
                    irfit.append(diffIr.iloc[c-r+i])
                    irfit2.append(ptest.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)
                
                for i in range(1, 2*rcen):
                    irfitc.append(diffIr.iloc[c-rcen+i])
                    irfitc2.append(ptest.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)
            elif style == 'reg':
                for i in range(1, 2*r):
                    irfit.append(framIr*sigIr.iloc[c-r+i])
                    pweight.append(ptest.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)  

                for i in range(1, 2*rcen):
                    irfitc.append(framIr*sigIr.iloc[c-rcen+i])
                    pweightc.append(ptest.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)  

                irfit2 = irfit
                irfitc2 = irfitc


            xirna += irwave

            irfit = np.array(irfit)
            irwave = np.array(irwave)
            irenergy = JeV*hp*cs / (irwave * 10**(-9))
            irphot = (3*irfit)/(irenergy / 3.6)
            irphot = framIr*(irphot+np.abs(np.min(irphot)))+offset1
            
            irfitc = np.array(irfitc)
            irwavec = np.array(irwavec)
            irenergyc = JeV*hp*cs / (irwavec * 10**(-9))
            irphotc = (3*irfitc)/(irenergyc/3.6)
            irphotc = framIr*(irphotc+np.abs(np.min(irphotc)))+offset1


            ################
            testwave = dfwa[df2.iloc[2,k]]
            testfit = df.iloc[:,k]
            testen = JeV*hp*cs/ (testwave * 10**(-9))
            testphot = (3*testfit)/(testen / 3.6)


            ###calibration uncertainty determination 
            modcalIr = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(irfit2)-np.min(irfit2), min=0)
            params.add('B', value = np.min(irfit2))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalir = modcalIr.fit(irfit2, params=params, x=irpix, weights=1/np.sqrt(np.abs(irfit2)), nan_policy='omit', max_nfev=None)
            params.update(resultcalir.params)

            #print(params['A'].value)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)

            colvec = parts(resultcalir.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)

            if style=='diff':

                test1 = centroid(irwavec, irfitc2, yerr=None)
            else:
                test1 = centroid(irwavec, irfitc2, yerr=np.sqrt(np.abs(pweightc)))

            

            #irfitc2 = irfitc2 + offset1
            irnacen.append(test1['centroid'])
            tempvalc.append(test1['centroid'])
            #irnacenu.append(np.sqrt(centroid(irwavec,irphotc)['cent_unc']**2 + calunc**2))
            irnacenu.append(np.sqrt(test1['cent_unc']**2 ))
            tempvalec.append(test1['cent_unc'])
            irnacent.append(np.float64(df2.iloc[1,k]))
            ####single gauss fit
            modIr = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(irfit2)-np.min(irfit2), min=0)
            params.add('B', value= np.min(irfit2))
            params.add('mu', value = np.mean(irwave))
            params.add('sig', value = 0.01, min=0)

            resultir = modIr.fit(irfit2, params=params, x=irwave, max_nfev = None, nan_policy='omit', weights=np.sqrt(np.abs(irfit2)))
            #resultir = modIr.fit(irfit2, params=params, x=irwave, max_nfev = None, nan_policy='omit', weights=np.sqrt(np.abs(pweight)))
            
            params.update(resultir.params)
            xplot = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot = modIr.eval(params=params, x=xplot)

            plt.figure() 
            plt.minorticks_on() 
            plt.xlabel('wavelength (nm)')
            plt.ylabel('Photon counts (arb)')
            plt.plot(irwave, irfit2, c='r', label='data')
            # plt.plot(testwave, testphot, c='g', label='single spec')
            plt.axvline(x=test1['centroid'], c='k', label='centroid calc')
            plt.axvline(x=test1['centroid']+test1['cent_unc'], c='k', ls='--')
            plt.axvline(x=test1['centroid']-test1['cent_unc'], c='k', ls='--')
            plt.plot(xplot, yplot, c='g', label='gaussian fit')
            plt.axvline(x=params['mu'].value, c='g')
            plt.axvline(x=params['mu'].value+params['mu'].stderr, c='g', ls='--')
            plt.axvline(x=params['mu'].value-params['mu'].stderr, c='g', ls='--')
            plt.title(df.columns[k])
            plt.legend() 
            plt.xlim(np.min(irwave), np.max(irwave))
            #plt.show() 
            plt.close()




            if str(resultir.params['mu'].stderr) == 'None':
                pass
            elif resultir.params['mu'].stderr < 0.05:
                irna.append(resultir.params['mu'].value)
                #irnae.append(np.sqrt(siglev1*resultir.params['mu'].stderr**2 + calunc**2))
                irnae.append(np.sqrt(siglev1*resultir.params['mu'].stderr**2))
                irnatime.append(np.float64(df2.iloc[1,k]))
                #print(resultir.params['mu'].value, resultir.params['mu'].stderr)
                irna1cal.append(calunc)
                irna1stat.append(siglev1*resultir.params['mu'].stderr)
                irna1sys.append(adsysunc[v])

                tempval.append(resultir.params['mu'].value)
                tempvale.append(siglev1*resultir.params['mu'].stderr)


                
            else:
                pass

            ###Double gauss fit 
            modIr2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(irfit2)-np.min(irfit2), min=0)
            params2.add('A2', value = np.max(irfit2)/2, min=0)
            params2.add('B', value= np.min(irfit2))
            params2.add('mu', value = np.mean(irwave))
            params2.add('mu2', value = np.mean(irwave))
            params2.add('sig', value = 0.01, min=0)

            resultir2 = modIr2.fit(irfit2, params=params2, x=irwave, max_nfev = None, nan_policy='omit', weights=1/np.sqrt(np.abs(irfit2)))
            params2.update(resultir2.params)
            xplot2 = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot2 = modIr2.eval(params=params2, x=xplot2)



            # plt.figure() 
            # plt.plot(irwave, irphot, c='b')
            # plt.plot(xplot, yplot, c='r')
            # #plt.xlim(7.28, 7.51)
            # #plt.axhline(y=0)
            # #plt.show() 
            # plt.close()


            ##Quad gauss fit 
            irnaq = quadgfit(wave, ptest, r=40, c=567, num=1)

            qirnastat.append(siglev1*irnaq['MU_err'])
            qtempval.append(irnaq['MU'])
            qtempvale.append(siglev1*irnaq['MU_err'])

            plt.figure() 
            plt.title(df.columns[k])
            plt.plot(irnaq['xdat'], irnaq['ydat'], c='b', label='data')
            plt.plot(irnaq['xplot'], irnaq['yplot'], c='r', label='4 gaussian fit')
            plt.plot(xplot, yplot, c='g', label='single gaussian fit')
            plt.axvline(x=test1['centroid'], c='k', label='centroid calc')
            plt.axvline(x=test1['centroid']+test1['cent_unc'], c='k', ls='--')
            plt.axvline(x=test1['centroid']-test1['cent_unc'], c='k', ls='--')
            plt.ylabel('Photons (Ir - Os)')
            plt.xlabel('wavelength (nm)')
            plt.minorticks_on()
            plt.legend() 
            #plt.show()
            plt.close() 

            tsum1.append(irnaq['sumphot'])
            ttime1.append(np.float64(df2.iloc[1,k]))

            tsumirna.append(np.sum(np.abs(irfit2)))
            ttimeirna.append(np.float64(df2.iloc[1,k]))


    qirnaset.append(np.average(qtempval, weights = 1/np.array(qtempvale)))
    qirnasete.append(np.sqrt((np.sqrt(np.sum(np.array(qtempvale)**2))/len(qtempvale))**2+caluncnair**2))
    qirnasetstat.append(np.sqrt(np.sum(np.array(qtempvale)**2))/len(qtempval))
    qirnasetcal.append(caluncnair)

    irnaset.append(np.average(tempval, weights = 1 / np.array(tempvale)))
    irnasete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2))/len(tempval))**2 + caluncnair**2))

    irnasetstat.append(np.sqrt(np.sum(np.array(tempvale)**2))/len(tempval))
    irnasetcal.append(caluncnair)

    cirnaset.append(np.average(tempvalc, weights = 1 / np.array(tempvale)))
    cirnasete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2))/len(tempvalc))**2 + caluncnair**2))
    cirnasetstat.append(np.sqrt(np.sum(np.array(tempvalec)**2))/len(tempvalc))
    cirnasetcal.append(caluncnair)


    qtempval = []
    qtempvale = []
    tempval = []
    tempvale = []
    tempvalc = []
    tempvalec = []
    for k in range(np.shape(df)[1]):        #Ir individual spectra(Mg-like) 

        if int(df2.iloc[2,k][3:]) == int(v):

            wave = dfwa[df2.iloc[2,k]]
            sigIr = df.iloc[:,k] 
            framIr = int(df2.iloc[4,k])
            calstring = df2.iloc[2,k]

            sigIr = sigIr / framIr 
            diffIr = sigIr - sig2

            irfit = []
            irwave = []

            irpix = []

            for i in range(len(diffIr)):
                if diffIr[i] < 0:
                    diffIr[i] = diffIr[i]*framIr
                else: 
                    diffIr[i] = diffIr[i]*fram2/cOs

            etest = hp*cs*JeV/(10**(-9)*wave)
            ptest = (3*diffIr)/(etest/3.6)
            
            c=574

            # if style == 'diff':
            #     for i in range(1, 2*r):
            #         irfit.append(diffIr.iloc[c-r+i])
            #         irwave.append(wave[c-r+i])
            #         irpix.append(c-r+i)
            # elif style == 'reg':
            #     for i in range(1, 2*r):
            #         irfit.append(framIr*sigIr.iloc[c-r+i])
            #         irwave.append(wave[c-r+i])
            #         irpix.append(c-r+i)  
            irfitc = []
            irwavec = []
            irpixc = []

            irfit2 = []
            irfitc2 = []

            pweight = []
            pweightc = []


            if style == 'diff':
                for i in range(1, 2*r):
                    irfit.append(diffIr.iloc[c-r+i])
                    irfit2.append(ptest.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)
                
                for i in range(1, 2*rcen):
                    irfitc.append(diffIr.iloc[c-rcen+i])
                    irfitc2.append(ptest.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)
            elif style == 'reg':
                for i in range(1, 2*r):
                    irfit.append(framIr*sigIr.iloc[c-r+i])
                    pweight.append(ptest.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)  

                for i in range(1, 2*rcen):
                    irfitc.append(framIr*sigIr.iloc[c-rcen+i])
                    pweightc.append(ptest.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)          

                irfit2 = irfit
                irfitc2 = irfitc 

            #if v==16:

            xirmg += irwave


            irfit = np.array(irfit)
            irwave = np.array(irwave)

            irenergy = JeV*hp*cs / (irwave * 10**(-9))
            irphot = (3*irfit)/(irenergy / 3.6)

            irfitc = np.array(irfitc)
            irwavec = np.array(irwavec)
            irenergyc = JeV*hp*cs / (irwavec * 10**(-9))
            irphotc = (3*irfitc)/(irenergyc/3.6)
            irphotc = framIr*(irphotc+np.abs(np.min(irphotc)))+offset1

            irphot = framIr*(irphot+np.abs(np.min(irphot)))+offset1
            #yirmg += irphot.tolist()
            ###calibration uncertainty determination 
            modcalIr = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(irfit2)-np.min(irfit2), min=0)
            params.add('B', value = np.min(irfit2))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalir = modcalIr.fit(irfit2, params=params, x=irpix, weights=None, nan_policy='omit', max_nfev=None)
            params.update(resultcalir.params)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)

            colvec = parts(resultcalir.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)
            #irfitc2 = irfitc2 + offset1

            test1 = centroid(irwavec,irfitc2, yerr=None)
            #test1 = centroid(irwavec,irfitc2, yerr=np.sqrt(np.abs(pweightc)))

            irmgcen.append(test1['centroid'])
            #irmgcenu.append(np.sqrt(centroid(irwavec,irphotc)['cent_unc']**2 + calunc**2))
            irmgcenu.append(np.sqrt(test1['cent_unc']**2))
            tempvalc.append(test1['centroid'])
            tempvalec.append(np.sqrt(test1['cent_unc']**2))
            irmgcent.append(np.float64(df2.iloc[1,k]))
            ####
            modIr = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(irfit2)-np.min(irfit2), min=0)
            params.add('B', value= np.min(irfit2))
            params.add('mu', value = np.mean(irwave))
            params.add('sig', value = 0.01, min=0)

            resultir = modIr.fit(irfit2, params=params, x=irwave, max_nfev = None, nan_policy='omit', weights=np.abs(irfit2)/np.sqrt(np.abs(irfit2)))
            #resultir = modIr.fit(irfit2, params=params, x=irwave, max_nfev = None, nan_policy='omit', weights=np.abs(pweight)/np.sqrt(np.abs(pweight)))
            
            params.update(resultir.params)
            #params.pretty_print()
            xplot = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot = modIr.eval(params=params, x=xplot)

            if resultir.params['mu'].stderr is not None:# or resultir.params['mu'].stderr < 0.05: 
            
                irmg.append(resultir.params['mu'].value)
                #irmge.append(np.sqrt(siglev1*resultir.params['mu'].stderr**2 + calunc**2))
                irmge.append(np.sqrt(siglev1*resultir.params['mu'].stderr**2))
                irmgtime.append(np.float64(df2.iloc[1,k]))

                irmg1cal.append(calunc)
                irmg1stat.append(siglev1*resultir.params['mu'].stderr)
                irmg1sys.append(adsysunc[v])

                tempval.append(resultir.params['mu'].value)
                tempvale.append(siglev1*resultir.params['mu'].stderr)

                
            else:
                pass
            
            ###Double gauss fit 
            modIr2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(irphot)-np.min(irphot), min=0)
            params2.add('A2', value = np.max(irphot)/2, min=0)
            params2.add('B', value= np.min(irphot))
            params2.add('mu', value = np.mean(irwave))
            params2.add('mu2', value = np.mean(irwave))
            params2.add('sig', value = 0.01, min=0)

            resultir2 = modIr2.fit(irphot, params=params2, x=irwave, max_nfev = None, nan_policy='omit', weights=None)
            params2.update(resultir2.params)
            xplot2 = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot2 = modIr2.eval(params=params2, x=xplot2)

            irmgq = quadgfit(wave, ptest, r=40, c=567, num=3)
            tsumirmg.append(np.sum(np.abs(irfit2)))
            ttimeirmg.append(np.float64(df2.iloc[1,k]))


            qtempval.append(irmgq['MU'])
            qtempvale.append(siglev1*irmgq['MU_err'])


    qirmgset.append(np.average(qtempval, weights=1/np.array(qtempvale)))
    qirmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))**2 + caluncmgir**2))
    qirmgsetcal.append(caluncmgir)
    qirmgsetstat.append(np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))

    irmgset.append(np.average(tempval, weights = 1 / np.array(tempvale)))
    irmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))**2 + caluncmgir**2))
    irmgsetcal.append(caluncmgir)
    irmgsetstat.append(np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))

    cirmgset.append(np.average(tempvalc, weights = 1 / np.array(tempvalec)))
    cirmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))**2 + caluncmgir**2))
    cirmgsetcal.append(caluncmgir)
    cirmgsetstat.append(np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))


    qtempval = []
    qtempvale = []
    tempval = []
    tempvale = []
    tempvalc = []
    tempvalec = []
    for b in range(np.shape(dfa)[1]):           #Os individual spectra Na-like

        if int(df2a.iloc[2,b][3:]) == int(v):

            wave2 = dfwab[df2a.iloc[2,b]]
            sigOs = dfa.iloc[:,b] #/ 
            framOs = int(df2a.iloc[4,b])
            calstring2 = df2a.iloc[2,b]
            
            sigOs = sigOs / framOs 
            diffOs = sigOs - sig 


            osfit = []
            oswave = []
            ospix = []
            osfit2 = []
            osfitc2 = []

            for i in range(len(diffOs)):
                if diffOs[i] < 0:
                    diffOs[i] = diffOs[i]*framOs
                else: 
                    diffOs[i] = diffOs[i]*fram/cIr

            etest = hp*cs*JeV/(10**(-9)*wave2)
            ptest = (3*diffOs)/(etest/3.6)



            c = 566 

            osfitc = []
            oswavec = []
            ospixc = []

            pweight = []
            pweightc = []


            if style == 'diff':
                for i in range(1, 2*r):
                    osfit.append(diffOs[c-r+i])
                    osfit2.append(ptest[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(diffOs[c-rcen+i])
                    osfitc2.append(ptest[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)
            if style == 'reg':
                for i in range(1, 2*r):
                    osfit.append(framOs*sigOs[c-r+i])
                    pweight.append(ptest[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(framOs*sigOs[c-rcen+i])
                    pweightc.append(ptest[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)
                
                osfit2 = osfit
                osfitc2 = osfitc

            osfit = np.array(osfit)
            oswave = np.array(oswave)

            osenergy = JeV*hp*cs/(oswave * 10**(-9))
            osphot = (3*osfit) / (osenergy /3.6)

            xosna += oswave.tolist()
            yosna += osphot.tolist()
            #print(np.min(osphot))
            osphot = framOs*(osphot+np.abs(np.min(osphot)))+offset1

            osfitc = np.array(osfitc)
            oswavec = np.array(oswavec)
            osenergyc = JeV*hp*cs / (oswavec * 10**(-9))
            osphotc = (3*osfitc)/(osenergyc/3.6)
            osphotc = framOs*(osphotc+np.abs(np.min(osphotc)))+offset1
            #print(np.min(osphot))
            ###calibration uncertainty determination 
            modcalOs = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(osfit2)-np.min(osfit2), min=0)
            params.add('B', value = np.min(osfit2))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalos = modcalOs.fit(osfit2, params=params, x=ospix, weights=np.sqrt(np.abs(osfit2)), nan_policy='omit', max_nfev=None)
            #resultcalos = modcalOs.fit(osfit2, params=params, x=ospix, weights=1/np.sqrt(np.abs(pweight)), nan_policy='omit', max_nfev=None)
            
            
            params.update(resultcalos.params)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring2+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,b]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)

            colvec = parts(resultcalos.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)

            test1 = centroid(oswavec, osfitc2, yerr=None)
            #test1 = centroid(oswavec, osfitc2, yerr=np.sqrt(np.abs(pweightc)))
            #osfitc2 = osfitc2 + offset1
            osnacen.append(test1['centroid'])
            #osnacenu.append(np.sqrt(centroid(oswavec,osphotc)['cent_unc']**2 + calunc**2))
            osnacenu.append(np.sqrt(test1['cent_unc']**2))
            tempvalc.append(test1['centroid'])
            tempvalec.append(np.sqrt(test1['cent_unc']**2))
            osnacent.append(np.float64(df2.iloc[1,k]))
            ###Single gauss fit
            modOs = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(osfit2)-np.min(osfit2), min=0)
            params.add('B', value =np.min(osfit2))
            params.add('mu', value = np.mean(oswave))
            params.add('sig', value = 0.01)
            
            resultos = modOs.fit(osfit2, params=params, x=oswave, weights=np.abs(osfit2)/np.sqrt(np.abs(osfit2)), nan_policy='omit', max_nfev=None)
            params.update(resultos.params)
            xplot = np.linspace(np.min(oswave), np.max(oswave), num=1000)
            yplot = resultos.eval(params=params, x=xplot)
            
            if resultos.params['mu'].stderr < 0.05:
                osna.append(resultos.params['mu'].value)
                #osnae.append(np.sqrt(siglev1*resultos.params['mu'].stderr**2 + calunc**2))
                osnae.append(np.sqrt(siglev1*resultos.params['mu'].stderr**2))
                osnatime.append(np.float64(df2a.iloc[1,b]))

                osna1cal.append(calunc) 
                osna1stat.append(siglev1*resultos.params['mu'].stderr)
                osna1sys.append(adsysunc[v])

                tempval.append(resultos.params['mu'].value)
                tempvale.append(siglev1*resultos.params['mu'].stderr)



                
            else:
                pass

            ###Double gauss fit
            modOs2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(osfit2)-np.min(osfit2), min=0)
            params2.add('A2', value = 0.5*np.max(osfit2), min=0)
            params2.add('B', value =np.min(osfit2))
            params2.add('mu', value = np.mean(oswave))
            params2.add('mu2', value = np.mean(oswave))
            params2.add('sig', value = 0.01)
        
            resultos2 = modOs2.fit(osfit2, params=params2, x=oswave, weights=None, nan_policy='omit', max_nfev=None)
            params2.update(resultos2.params)



            osnaq = quadgfit(wave, -ptest, r=40, c=567, num=2)
            qtempval.append(osnaq['MU'])
            qtempvale.append(osnaq['MU_err'])
            #print(osnaq)
            plt.figure() 
            plt.title(df.columns[b])
            plt.plot(irnaq['xdat'], osnaq['ydat'], c='b', label='data')
            plt.plot(irnaq['xplot'], osnaq['yplot'], c='r', label='4 gaussian fit')
            plt.plot(xplot, -yplot, c='g', label='single gaussian fit')
            plt.axvline(x=test1['centroid'], c='k', label='centroid calc')
            plt.axvline(x=test1['centroid']+test1['cent_unc'], c='k', ls='--')
            plt.axvline(x=test1['centroid']-test1['cent_unc'], c='k', ls='--')
            plt.ylabel('Photons (Ir - Os)')
            plt.xlabel('wavelength (nm)')
            plt.minorticks_on()
            plt.legend() 
            #plt.show()
            plt.close() 

            tsum1os.append(osnaq['sumphot'])
            ttime1os.append(np.float64(df2a.iloc[1,b]))

            tsumosna.append(np.sum(np.abs(osfit2)))
            ttimeosna.append(np.float64(df2a.iloc[1,b]))
            

    qosnaset.append(np.average(qtempval, weights = 1 / np.array(qtempvale)))
    qosnasete.append(np.sqrt((np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))**2 + caluncnaos**2))
    qosnasetcal.append(caluncnaos)
    qosnasetstat.append(np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))
    
    
    osnaset.append(np.average(tempval, weights = 1 / np.array(tempvale)))
    osnasete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))**2 + caluncnaos**2))
    osnasetcal.append(caluncnaos)
    osnasetstat.append(np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))

    cosnaset.append(np.average(tempvalc, weights = 1 / np.array(tempvalec)))
    cosnasete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))**2 + caluncnaos**2))
    cosnasetcal.append(caluncnaos)
    cosnasetstat.append(np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))


    qtempval = []
    qtempvale = []
    tempval = []
    tempvale = []
    tempvalc = []
    tempvalec = []

    for b in range(np.shape(dfa)[1]):           #Os individual spectra Mg-like

        if int(df2a.iloc[2,b][3:]) == int(v):

            wave2 = dfwab[df2a.iloc[2,b]]
            sigOs = dfa.iloc[:,b] #/ 
            framOs = int(df2a.iloc[4,b])
            calstring2 = df2a.iloc[2,b]
            
            sigOs = sigOs / framOs 
            diffOs = sigOs - sig 


            osfit = []
            oswave = []
            ospix = []
            c = 597 


            etest1 = hp*cs*JeV/(10**(-9)*wave2)
            ptest1 = (3*sigOs*framOs)/(etest1/3.6)
    
            # print(framOs)
            # plt.figure() 
            # plt.plot(wave2, ptest1, c='b', label='dat')
            # plt.ylabel('total photon counts')
            # plt.ylim(60, 95)
            # # plt.plot(wave2, sigOs*framOs, c='r')
            # # plt.ylabel('ADU')
            # #plt.ylim(1000, 1400)
            # plt.xlabel('wavelength [nm]')
            # plt.xlim(7.20, 7.70)
            # plt.minorticks_on()
            # plt.legend()
            # plt.show()
            # plt.close()



            osfitc = []
            oswavec = []
            ospixc = []

            osfit2 = []
            osfitc2 = []
            pweight = []
            pweightc = []

            for i in range(len(diffOs)):
                if diffOs[i] < 0:
                    diffOs[i] = diffOs[i]*framOs
                else: 
                    diffOs[i] = diffOs[i]*fram/cIr

            etest = hp*cs*JeV/(10**(-9)*wave2)
            ptest = (3*diffOs)/(etest/3.6)



            if style == 'diff':
                for i in range(1, 2*r):
                    osfit.append(diffOs[c-r+i])
                    osfit2.append(ptest[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(diffOs[c-rcen+i])
                    osfitc2.append(ptest[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)
            if style == 'reg':
                for i in range(1, 2*r):
                    osfit.append(framOs*sigOs[c-r+i])
                    pweight.append(ptest[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(framOs*sigOs[c-rcen+i])
                    pweightc.append(ptest[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)

            osfit2 = osfit
            osfitc2 = osfitc

            osfit = np.array(osfit)
            oswave = np.array(oswave)

            osenergy = JeV*hp*cs/(oswave * 10**(-9))
            osphot = (3*osfit) / (osenergy /3.6)

            xosmg += oswave.tolist()
            yosmg += osphot.tolist()

            osphot = framOs*(osphot+np.abs(np.min(osphot)))+offset1

            osfitc = np.array(osfitc)
            oswavec = np.array(oswavec)
            osenergyc = JeV*hp*cs / (oswavec * 10**(-9))
            osphotc = (3*osfitc)/(osenergyc/3.6)
            osphotc = framOs*(osphotc+np.abs(np.min(osphotc)))+offset1
            ###calibration uncertainty determination 
            modcalOs = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(osfit)-np.min(osfit), min=0)
            params.add('B', value = np.min(osfit))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalos = modcalOs.fit(osfit, params=params, x=ospix, weights=None, nan_policy='omit', max_nfev=None)
            
            params.update(resultcalos.params)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring2+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)


            colvec = parts(resultcalos.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)
            #osfitc2 = osfitc2 + offset1

            test1 = centroid(oswavec,osfitc2, yerr=None)
            #test1 = centroid(oswavec,osfitc2, yerr=np.sqrt(np.abs(pweightc)))
            osmgcen.append(test1['centroid'])
            #osmgcenu.append(np.sqrt(centroid(oswavec,osphotc)['cent_unc']**2 + calunc**2))
            osmgcenu.append(np.sqrt(test1['cent_unc']**2))
            
            tempvalc.append(test1['centroid'])
            tempvalec.append(np.sqrt(test1['cent_unc']**2))
            osmgcent.append(np.float64(df2.iloc[1,k]))

            ###line center determination 
            modOs = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(osfit2)-np.min(osfit2), min=0)
            params.add('B', value =np.min(osfit2))
            params.add('mu', value = np.mean(oswave))
            params.add('sig', value = 0.01)
            
            #resultos = modOs.fit(osfit2, params=params, x=oswave, weights=np.abs(pweight)/np.sqrt(np.abs(pweight)), nan_policy='omit', max_nfev=None)
            resultos = modOs.fit(osfit2, params=params, x=oswave, weights=np.abs(osfit2)/np.sqrt(np.abs(osfit2)), nan_policy='omit', max_nfev=None)
            
            
            params.update(resultos.params)
            if resultos.params['mu'].stderr < 0.05:
                osmg.append(resultos.params['mu'].value)
                #osmge.append(np.sqrt(siglev1*resultos.params['mu'].stderr**2 + calunc**2))
                osmge.append(np.sqrt(siglev1*resultos.params['mu'].stderr**2 ))
                osmgtime.append(np.float64(df2a.iloc[1,b]))

                osmg1cal.append(calunc)
                osmg1stat.append(siglev1*resultos.params['mu'].stderr)
                osmg1sys.append(adsysunc[v])

                tempval.append(resultos.params['mu'].value)
                tempvale.append(siglev1*resultos.params['mu'].stderr)
                

                
            else:
                pass

            ###Double gauss fit
            modOs2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(osfit2)-np.min(osfit2), min=0)
            params2.add('A2', value = 0.5*np.max(osfit2), min=0)
            params2.add('B', value =np.min(osfit2))
            params2.add('mu', value = np.mean(oswave))
            params2.add('mu2', value = np.mean(oswave))
            params2.add('sig', value = 0.01)
            
            resultos2 = modOs2.fit(osfit2, params=params2, x=oswave, weights=None, nan_policy='omit', max_nfev=None)
            params2.update(resultos2.params)


            tsumosmg.append(np.sum(np.abs(osfit2)))
            ttimeosmg.append(np.float64(df2a.iloc[1,b]))

            osmgq = quadgfit(wave2, -ptest, r=40, c=567, num=4)
            qtempval.append(osmgq['MU'])
            qtempvale.append(osmgq['MU_err'])
            #print(osmgq['MU'], osmgq['MU_err'])
            # plt.figure()
            # plt.plot(osmgq['xplot'], osmgq['yplot'], c='r')
            # plt.plot(osmgq['xdat'], osmgq['ydat'], c='b')
            # plt.show()
            # plt.close()


    qosmgset.append(np.average(qtempval, weights = 1 / np.array(qtempvale)))
    qosmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))**2 + caluncmgos**2))
    qosmgsetcal.append(caluncmgos)
    qosmgsetstat.append(np.sqrt(np.sum(np.array(qtempvale)**2)) / len(qtempval))




    osmgset.append(np.average(tempval, weights = 1 / np.array(tempvale)))
    osmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))**2 + caluncmgos**2))
    osmgsetcal.append(caluncmgos)
    osmgsetstat.append(np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))

    cosmgset.append(np.average(tempvalc, weights = 1 / np.array(tempvalec)))
    cosmgsete.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))**2 + caluncmgos**2))
    cosmgsetcal.append(caluncmgos)
    cosmgsetstat.append(np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))


    tempval = []
    tempvale = []
    tempvalc = []
    tempvalec = []

    for k in range(np.shape(df)[1]):        #Ir individual spectra(Na-like) 2nd order 

        if int(df2.iloc[2,k][3:]) == int(v):

            wave = dfwa[df2.iloc[2,k]]
            sigIr = df.iloc[:,k] 
            framIr = int(df2.iloc[4,k])
            calstring = df2.iloc[2,k]

            sigIr = sigIr / framIr 
            diffIr = sigIr - sig2

            irfit = []
            irwave = []

            irpix = []
            c=1480

            # if style == 'diff':
            #     for i in range(1, 2*r):
            #         irfit.append(diffIr.iloc[c-r+i])
            #         irwave.append(wave[c-r+i])
            #         irpix.append(c-r+i)
            # elif style == 'reg':
            #     for i in range(1, 2*r):
            #         irfit.append(framIr*sigIr.iloc[c-r+i])
            #         irwave.append(wave[c-r+i])
            #         irpix.append(c-r+i)  

            irfitc = []
            irwavec = []
            irpixc = []

            if style == 'diff':
                for i in range(1, 2*r):
                    irfit.append(diffIr.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)
                
                for i in range(1, 2*rcen):
                    irfitc.append(diffIr.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)
            elif style == 'reg':
                for i in range(1, 2*r):
                    irfit.append(framIr*sigIr.iloc[c-r+i])
                    irwave.append(wave[c-r+i])
                    irpix.append(c-r+i)  

                for i in range(1, 2*rcen):
                    irfitc.append(framIr*sigIr.iloc[c-rcen+i])
                    irwavec.append(wave[c-rcen+i])
                    irpixc.append(c-rcen+i)    

            irfit = np.array(irfit)
            irwave = np.array(irwave)

            irenergy = JeV*hp*cs / (irwave * 10**(-9))
            irphot = (3*irfit)/(irenergy / 3.6)
            irphot = framIr*(irphot+np.abs(np.min(irphot)))+offset1

            irfitc = np.array(irfitc)
            irwavec = np.array(irwavec)
            irenergyc = JeV*hp*cs / (irwavec * 10**(-9))
            irphotc = (3*irfitc)/(irenergyc/3.6)
            irphotc = framIr*(irphotc+np.abs(np.min(irphotc)))+offset1

            ###calibration uncertainty determination 
            modcalIr = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(irfit)-np.min(irfit), min=0)
            params.add('B', value = np.min(irfit))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalir = modcalIr.fit(irfit, params=params, x=irpix, weights=None, nan_policy='omit', max_nfev=None)
            params.update(resultcalir.params)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)

            colvec = parts(resultcalir.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)

            #irna2.append(centroid(irwave,irphot)['centroid'])
            #irnae2.append(np.sqrt(centroid(irwave,irphot)['cent_unc']**2 + calunc**2))

            ####Single Gauss fit 
            modIr = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(irphot)-np.min(irphot), min=0)
            params.add('B', value= np.min(irphot))
            params.add('mu', value = np.mean(irwave))
            params.add('sig', value = 0.01, min=0)

            resultir = modIr.fit(irphot, params=params, x=irwave, max_nfev = None, nan_policy='omit', weights=irphot/np.sqrt(irphot))
            params.update(resultir.params)
            xplot = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot = modIr.eval(params=params, x=xplot)

            plt.figure() 
            plt.plot(irwave,irphot)
            #plt.show() 
            plt.close()

            test1 = centroid(irwavec,irphotc, None)

            irna2c.append(test1['centroid'])
            #irnae2c.append(np.sqrt(centroid(irwavec,irphotc)['cent_unc']**2 + calunc**2))
            irnae2c.append(np.sqrt(test1['cent_unc']**2 ))
            irnatime2c.append(np.float64(df2.iloc[1,k]))

            tempvalc.append(test1['centroid'])
            tempvalec.append(np.sqrt(test1['cent_unc']**2 ))

            if str(resultir.params['mu'].stderr) == 'None':
                pass
            elif resultir.params['mu'].stderr < 0.01:
                irna2.append(resultir.params['mu'].value-estcalunc)       #estcalunc
                #irna2.append(resultir.params['mu'].value)
                irnae2.append(np.sqrt(siglev1*resultir.params['mu'].stderr**2 + calunc**2))
                irnatime2.append(np.float64(df2.iloc[1,k]))

                irna2cal.append(calunc)
                irna2stat.append(siglev1*resultir.params['mu'].stderr)
                irna2sys.append(adsysunc[v])

                tempval.append(resultir.params['mu'].value)
                tempvale.append(siglev1*resultir.params['mu'].stderr)

            else:
                pass
        
            ###Double gauss fit 
            modIr2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(irphot)-np.min(irphot), min=0)
            params2.add('A2', value = np.max(irphot)/2, min=0)
            params2.add('B', value= np.min(irphot))
            params2.add('mu', value = np.mean(irwave))
            params2.add('mu2', value = np.mean(irwave))
            params2.add('sig', value = 0.01, min=0)

            resultir2 = modIr2.fit(irphot, params=params2, x=irwave, max_nfev = None, nan_policy='omit', weights=None)
            params2.update(resultir2.params)
            xplot2 = np.linspace(np.min(irwave), np.max(irwave), num=1000)
            yplot2 = modIr2.eval(params=params2, x=xplot2)
        
    irnaset2.append(np.average(tempval, weights = 1/np.array(tempvale)))
    irnasete2.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))**2 + caluncnair2**2))
    irnasetcal2.append(caluncnair2)
    irnasetstat2.append(np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))
    
    cirnaset2.append(np.average(tempvalc, weights = 1/np.array(tempvalec)))
    cirnasete2.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))**2 + caluncnair2**2))
    cirnasetcal2.append(caluncnair2)
    cirnasetstat2.append(np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))
    

    tempval = []
    tempvale = []
    tempvalc = []
    tempvalec = []

    for b in range(np.shape(dfa)[1]):           #Os individual spectra Na-like 2nd order 

        if int(df2a.iloc[2,b][3:]) == int(v):

            wave2 = dfwab[df2a.iloc[2,b]]
            sigOs = dfa.iloc[:,b] #/ 
            framOs = int(df2a.iloc[4,b])
            calstring2 = df2a.iloc[2,b]
            
            sigOs = sigOs / framOs 
            diffOs = sigOs - sig 


            osfit = []
            oswave = []
            ospix = []
            c = 1515

            # if style == 'diff':
            #     for i in range(1, 2*r):
            #         osfit.append(diffOs[c-r+i])
            #         oswave.append(wave2[c-r+i])
            #         ospix.append(c-r+i)
            # if style == 'reg':
            #     for i in range(1, 2*r):
            #         osfit.append(framOs*sigOs[c-r+i])
            #         oswave.append(wave2[c-r+i])
            #         ospix.append(c-r+i)

            osfitc = []
            oswavec = []
            ospixc = []


            if style == 'diff':
                for i in range(1, 2*r):
                    osfit.append(diffOs[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(diffOs[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)
            if style == 'reg':
                for i in range(1, 2*r):
                    osfit.append(framOs*sigOs[c-r+i])
                    oswave.append(wave2[c-r+i])
                    ospix.append(c-r+i)

                for i in range(1, 2*rcen):
                    osfitc.append(framOs*sigOs[c-rcen+i])
                    oswavec.append(wave2[c-rcen+i])
                    ospixc.append(c-rcen+i)

            osfit = np.array(osfit)
            oswave = np.array(oswave)

            osenergy = JeV*hp*cs/(oswave * 10**(-9))
            osphot = (3*osfit) / (osenergy /3.6)
            #osphot = osphot * framOs
            osphot = framOs*(osphot+np.abs(np.min(osphot)))+offset1

            osfitc = np.array(osfitc)
            oswavec = np.array(oswavec)
            osenergyc = JeV*hp*cs / (oswavec * 10**(-9))
            osphotc = (3*osfitc)/(osenergyc/3.6)
            osphotc = framOs*(osphotc+np.abs(np.min(osphotc)))+offset1
            ###calibration uncertainty determination 
            modcalOs = Model(gauss)
            params=Parameters() 
            params.add('A', value = np.max(osfit)-np.min(osfit), min=0)
            params.add('B', value = np.min(osfit))
            params.add('mu', value = c)
            params.add('sig', value = 0.5)
            resultcalos = modcalOs.fit(osfit, params=params, x=ospix, weights=None, nan_policy='omit', max_nfev=None)
            params.update(resultcalos.params)

            for name in glob.glob(ftranloc+'\\Calibration\\'+calstring2+'\\*CovarianceMatrix.csv'):
                #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,b]+'\\*CovarianceMatrix.csv'):
                nam = name

            tempdf = pd.read_csv(r""+nam)
            coma = np.array(tempdf)

            colvec = parts(resultcalos.params['mu'].value)
            
            temp1 = np.matmul(coma, colvec)
        
            temp = tval2*np.sqrt(np.matmul(colvec.T, temp1))
            
            calunc = temp[0,0]
            calibrationuncert.append(calunc)

            test1 = centroid(oswavec, osphotc, None)

            osna2c.append(test1['centroid'])
            #osnae2c.append(np.sqrt(centroid(oswavec,osphotc)['cent_unc']**2 + calunc**2))
            osnae2c.append(np.sqrt(test1['cent_unc']**2 ))
            osnatime2c.append(np.float64(df2a.iloc[1,b]))

            tempvalc.append(test1['centroid'])
            tempvalec.append(np.sqrt(test1['cent_unc']**2 ))


            ###line center determination 
            modOs = Model(gauss)
            params = Parameters() 
            params.add('A', value = np.max(osphot)-np.min(osphot), min=0)
            params.add('B', value =np.min(osphot))
            params.add('mu', value = np.mean(oswave))
            params.add('sig', value = 0.01)
            
            resultos = modOs.fit(osphot, params=params, x=oswave, weights=osphot/np.sqrt(osphot), nan_policy='omit', max_nfev=None)
            params.update(resultos.params)
            if str(resultos.params['mu'].stderr) == 'None':
                pass
            elif resultos.params['mu'].stderr < 0.05:
                #osna2.append(resultos.params['mu'].value)   #estcalunc
                osna2.append(resultos.params['mu'].value+estcalunc)
                osnae2.append(np.sqrt(siglev1*resultos.params['mu'].stderr**2 + calunc**2))
                osnatime2.append(np.float64(df2a.iloc[1,b]))


                osna2cal.append(calunc) 
                osna2stat.append(siglev1*resultos.params['mu'].stderr)
                osna2sys.append(adsysunc[v])

                tempval.append(resultos.params['mu'].value)
                tempvale.append(siglev1*resultos.params['mu'].stderr)

            else:
                pass

            ###Double gauss fit
            modOs2 = Model(gauss2)
            params2 = Parameters() 
            params2.add('A', value = np.max(osphot)-np.min(osphot), min=0)
            params2.add('A2', value = 0.5*np.max(osphot), min=0)
            params2.add('B', value =np.min(osphot))
            params2.add('mu', value = np.mean(oswave))
            params2.add('mu2', value = np.mean(oswave))
            params2.add('sig', value = 0.01)
            
            resultos2 = modOs2.fit(osphot, params=params2, x=oswave, weights=None, nan_policy='omit', max_nfev=None)
            params2.update(resultos2.params)
    
    osnaset2.append(np.average(tempval, weights = 1/np.array(tempvale)))
    osnasete2.append(np.sqrt((np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))**2 + caluncnaos2**2))
    osnasetcal2.append(caluncnaos2)
    osnasetstat2.append(np.sqrt(np.sum(np.array(tempvale)**2)) / len(tempval))
    
    cosnaset2.append(np.average(tempvalc, weights = 1/np.array(tempvalec)))
    cosnasete2.append(np.sqrt((np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))**2 + caluncnaos2**2))
    cosnasetcal2.append(caluncnaos2)
    cosnasetstat2.append(np.sqrt(np.sum(np.array(tempvalec)**2)) / len(tempvalc))
    
    
    
    # plt.figure() 
    # plt.title('Difference spectra, scaled to 1 frame / 1 spectra , Calibration Set #'+str(v))
    # plt.plot(pixel, difference2, label='Difference of Os-Ir')
    # plt.xlabel('pixel number')
    # plt.ylabel('ADU Signal')
    # plt.minorticks_on() 
    # plt.axhline(y=0, c='k')
    # #plt.xlim(np.min(wave), np.max(wave))
    # #plt.xlim(2*7.25, 2*7.70)
    # #plt.xlim(7.25, 7.70)
    # # plt.axvline(x=2*7.29198, c='r', label='Ir Na-like (GRASP2K)')
    # # plt.axvline(x=2*7.44616, c='tab:purple', label='Os Na-like (GRASP2K)')
    # # plt.axvline(x=2*7.50724, c='k', label='Ir Mg-like (GRASP2K)')
    # # plt.axvline(x=2*7.66984, label='Os Mg-like (GRASP2K)')
    # plt.legend()
    # #plt.show() 
    # plt.close()

    ###First order 
    c = 567
    r2 = 40
    xfit = []               #used for wavelength calibration 
    yfit = []               #difference of photon counts between Os and Ir spectra
    xfit2 = []              #used for pixel location 
    yfit2 = []
    for i in range(1, 2*r2):
        xfit.append(wave.iloc[c-r2+i])
        yfit.append(difference.iloc[c-r2+i])

        xfit2.append(pixel[c-r2+i])
        yfit2.append(difference2.iloc[c-r2+i])

    mod = Model(gauss4)
    params = Parameters() 
    params.add('A', value = np.max(yfit)-np.average(yfit), min=0)
    params.add('mu', value = 7.29, min=7.26, max=7.32)
    params.add('sig', value = 0.01, min=0)

    params.add('A2', value = np.min(yfit)-np.average(yfit), max=0)
    params.add('mu2', value = 7.44, min=7.43, max=7.46)

    params.add('A3', value = np.max(yfit)-np.average(yfit), min=0)
    params.add('mu3', value = 7.49, min=7.46, max=7.51)

    params.add('A4', value = np.min(yfit)-np.average(yfit), max=0)
    params.add('mu4', value = 7.63, min=7.61, max=7.67)

    params.add('K', value = np.average(yfit))

    result = mod.fit(yfit, params=params, x=xfit, weights=None, nan_policy='omit', max_nfev=None)
    params.update(result.params)

    

    mod2 = Model(gauss4)
    params2 = Parameters()
    params2.add('A', value = np.max(yfit2)-np.average(yfit2), min=0)
    params2.add('mu', value = 544, min=542, max=546)
    params2.add('sig', value = 1, min=0)

    params2.add('A2', value = np.min(yfit2)-np.average(yfit2), max=0)
    params2.add('mu2', value = 567, min=565, max=569)

    params2.add('A3', value = np.max(yfit2)-np.average(yfit2), min=0)
    params2.add('mu3', value = 573, min=571, max=575)

    params2.add('A4', value = np.min(yfit2)-np.average(yfit2), max=0)
    params2.add('mu4', value = 597, min=595, max=599)

    params2.add('K', value = np.average(yfit2))

    result2 = mod.fit(yfit2, params=params2, x=xfit2, weights=None, nan_policy='omit', max_nfev=None)
    params2.update(result2.params)

    xplot = np.linspace(np.min(xfit), np.max(xfit), num=1000)
    xplot2 = np.linspace(np.min(xfit2), np.max(xfit2), num=1000)
    yplot = result.eval(params=params, x=xplot)
    yplot2 = result2.eval(params=params2, x=xplot2)

    # plt.figure()
    # plt.plot(xfit, yfit, label='data')
    # plt.plot(xplot, yplot, label='4-gauss fit')
    # plt.legend()
    # plt.xlim(np.min(xfit),np.max(xfit))
    # plt.minorticks_on()
    # plt.show()
    # plt.close()



    for name in glob.glob(ftranloc+'\\Calibration\\'+calstring+'\\*CovarianceMatrix.csv'):
        #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
        nam = name

    tempdf = pd.read_csv(r""+nam)
    coma = np.array(tempdf)
  
    colvec = parts(result2.params['mu'].value)
    colvec2 = parts(result2.params['mu2'].value)
    colvec3 = parts(result2.params['mu3'].value)
    colvec4 = parts(result2.params['mu4'].value)

    temp1 = np.matmul(coma, colvec)
    temp12 = np.matmul(coma, colvec2)
    temp13 = np.matmul(coma, colvec3)
    temp14 = np.matmul(coma, colvec4)


    temp = 2.4*np.sqrt(np.matmul(colvec.T, temp1))
    temp2 = 2.4 * np.sqrt(np.matmul(colvec2.T, temp12))
    temp3 = 2.4*np.sqrt(np.matmul(colvec3.T, temp13))
    temp4 = 2.4*np.sqrt(np.matmul(colvec4.T, temp14))

    calunc = temp[0,0]
    calunc2 = temp2[0,0]
    calunc3 = temp3[0,0]
    calunc4 = temp4[0,0]


    if result.params['mu'].stderr is not None or result.params['mu2'].stderr is not None or result.params['mu3'].stderr is not None or result.params['mu4'].stderr is not None:
        calibset.append(v)
        IrNa.append(result.params['mu'].value)
        IrNae.append(np.sqrt((siglev1*result.params['mu'].stderr)**2 + (calunc)**2))
        # irna1cal.append(calunc)
        # irna1stat.append(result.params['mu'].stderr)
        # irna1sys.append(adsysunc[v])

        OsNa.append(result.params['mu2'].value)
        OsNae.append(np.sqrt((siglev1*result.params['mu2'].stderr)**2 + (calunc2)**2))
        # osna1cal.append(calunc2) 
        # osna1stat.append(result.params['mu2'].stderr)
        # osna1sys.append(adsysunc[v])

        IrMg.append(result.params['mu3'].value)
        IrMge.append(np.sqrt((siglev1*result.params['mu3'].stderr)**2 + (calunc3)**2))
        # irmg1cal.append(calunc3)
        # irmg1stat.append(result.params['mu3'].stderr)
        # irmg1sys.append(adsysunc[v])

        OsMg.append(result.params['mu4'].value)
        OsMge.append(np.sqrt((siglev1*result.params['mu4'].stderr)**2 + (calunc4)**2)) 
        # osmg1cal.append(calunc4)
        # osmg1stat.append(result.params['mu4'].stderr)
        # osmg1sys.append(adsysunc[v])
    else:
        continue


    ###Second order (ONLY NA-LIKE, MG-LIKE IS COVERED BY 15nm O LINE)
    c4 = 1498
    r4 = 25
    xfit3 = []               #used for wavelength calibration 
    yfit3 = []               #difference of photon counts between Os and Ir spectra
    xfit4 = []              #used for pixel location 
    yfit4 = []
    for i in range(1, 2*r4):
        xfit3.append(wave.iloc[c4-r4+i])
        yfit3.append(difference.iloc[c4-r4+i])

        xfit4.append(pixel[c4-r2+i])
        yfit4.append(difference2.iloc[c4-r4+i])



    mod3 = Model(gauss2)
    params = Parameters() 
    params.add('A', value = np.max(yfit3)-np.average(yfit3), min=0)
    params.add('mu', value = 2*7.29, min=2*7.26, max=2*7.32)
    params.add('sig', value = 0.01, min=0)

    params.add('A2', value = np.min(yfit3)-np.average(yfit3), max=0)
    params.add('mu2', value = 2*7.44, min=2*7.43, max=2*7.46)


    params.add('B', value = np.average(yfit3))

    result = mod3.fit(yfit3, params=params, x=xfit3, weights=None, nan_policy='omit', max_nfev=None)
    params.update(result.params)


    mod4 = Model(gauss2)
    params2 = Parameters()
    params2.add('A', value = np.max(yfit4)-np.average(yfit4), min=0)
    params2.add('mu', value = 1465, min=1463, max=1467)
    params2.add('sig', value = 1, min=0)

    params2.add('A2', value = np.min(yfit4)-np.average(yfit4), max=0)
    params2.add('mu2', value = 1500, min=1502, max=1498)

    params2.add('B', value = np.average(yfit4))

    result2 = mod4.fit(yfit4, params=params2, x=xfit4, weights=None, nan_policy='omit', max_nfev=None)
    params2.update(result2.params)

    xplot3 = np.linspace(np.min(xfit3), np.max(xfit3), num=1000)
    xplot4 = np.linspace(np.min(xfit4), np.max(xfit4), num=1000)
    yplot3 = result.eval(params=params, x=xplot3)
    yplot4 = result2.eval(params=params2, x=xplot4)

    for name in glob.glob(ftranloc+'\\Calibration\\'+calstring+'\\*CovarianceMatrix.csv'):
        #for name in glob.glob('C:\\Users\\ahosi\\OneDrive\\Desktop\\FileTrans4\\Calibration\\'+df2.iloc[2,k]+'\\*CovarianceMatrix.csv'):
        nam = name

    tempdf = pd.read_csv(r""+nam)
    coma = np.array(tempdf)
  
    colvec = parts(result2.params['mu'].value)
    colvec2 = parts(result2.params['mu2'].value)

    temp1 = np.matmul(coma, colvec)
    temp12 = np.matmul(coma, colvec2)

    temp = 2.201*np.sqrt(np.matmul(colvec.T, temp1))
    temp2 = 2.201*np.sqrt(np.matmul(colvec2.T, temp12))

    calunc = temp[0,0]
    calunc2 = temp2[0,0]


    # if result.params['mu'].stderr is not None or result.params['mu2'].stderr is not None:
    #     #calibset.append(v)
    #     #IrNa.append(result.params['mu'].value)
    #     #IrNae.append(np.sqrt((result.params['mu'].stderr)**2 + (calunc)**2))
    #     irna2cal.append(calunc)
    #     irna2stat.append(result.params['mu'].stderr)
    #     irna2sys.append(adsysunc[v])

    #     #OsNa.append(result.params['mu2'].value)
    #     #OsNae.append(np.sqrt((result.params['mu2'].stderr)**2 + (calunc2)**2))
    #     osna2cal.append(calunc2) 
    #     osna2stat.append(result.params['mu2'].stderr)
    #     osna2sys.append(adsysunc[v])

    #     # IrMg.append(result.params['mu3'].value)
    #     # IrMge.append(np.sqrt((result.params['mu3'].stderr)**2 + (calunc3)**2))
    #     # irmg1cal.append(calunc3)
    #     # irmg1stat.append(result.params['mu3'].stderr)
    #     # irmg1sys.append(adsysunc[v])

    #     # OsMg.append(result.params['mu4'].value)
    #     # OsMge.append(np.sqrt((result.params['mu4'].stderr)**2 + (calunc4)**2)) 
    #     # osmg1cal.append(calunc4)
    #     # osmg1stat.append(result.params['mu4'].stderr)
    #     # osmg1sys.append(adsysunc[v])
    # else:
    #     continue



    #osna2sys.append(adsysunc[v])
    #irna2sys.append(adsysunc[v])


    # relnorm = 1520
    # relnorm = 1495
    # plt.figure()
    # plt.plot(wave, sig, label='Ir')
    # plt.plot(wave2, sig2 - (sig2[relnorm] - sig[relnorm]), label='Os')
    # #plt.plot(wave3, sig3+8100, label='Bg')
    # #plt.plot(wave4, sig4+8100, label='Ne')
    # #plt.plot(wave3, sig3 - (sig3[relnorm] - sig[relnorm]), label='Bg')
    # #plt.plot(wave4, sig4 - (sig4[relnorm] -sig[relnorm]), label='Ne')
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('ADU signal / minute')
    # plt.minorticks_on()
    # plt.title('Comparison of Os + Ir spectra, Calibration Set #' + str(v))
    # plt.xlim(7.2, 7.7)
    # #plt.xlim(14.4,15)
    # plt.axvline(x=7.4484, label='Os Na-like', c='k')
    # plt.axvline(x=7.2948, label='Ir Na-like', c='c')
    # plt.axvline(x=7.4945, label='Ir Mg-like', c='tab:purple')
    # plt.axvline(x=7.6570, label='Os Mg-like', c='g')
    # #plt.xlim(7.27*2, 7.7*2)
    # # plt.axvline(x=7.4484*2, label='Os 2nd Na-like', c='k')
    # # plt.axvline(x=7.2948*2, label='Ir 2nd Na-like', c='c')
    # # plt.axvline(x=7.5072*2, label='Ir 2nd Mg-like', c='tab:purple')
    # # plt.axvline(x=7.669*2, label='Os 2nd Mg-like', c='g')
    # plt.legend()
    # #plt.ylim(105, 150)
    # plt.show() 
    # plt.close()


    # plt.figure() 
    # plt.plot(xfit4, yfit4, c='b', label='data')
    # plt.plot(xplot4, yplot4, c='r', label='fit')
    # plt.minorticks_on()
    # plt.legend() 
    # plt.xlabel('pixel number')
    # plt.ylabel('signal (ADU)')
    # plt.title('Quad Gauss Fit, Calibration Set #' + str(v) + ' (pixel space)')
    # plt.show() 
    # plt.close() 

tsum2os = np.array(tsumosna) + np.array(tsumosmg)
tsum2ir = np.array(tsumirna) + np.array(tsumirmg)


# plt.figure()
# plt.xlabel('time [hrs]')
# plt.ylabel('total photon counts')
# plt.title('quad gaussian fitting region photon counts ~ 80pixels')
# plt.scatter(ttime1, tsum1, c='r', label='Quad Ir spectra')
# plt.scatter(ttime1os, tsum1os, c='b', label='Quad Os spectra')
# plt.scatter(ttimeosna, 2*tsum2os, c='c', label='Single Gauss Os Summed')
# plt.scatter(ttimeirna, 2*tsum2ir, c='tab:orange', label='Single Gauss Ir Summed')
# plt.ylim(100, 700)
# plt.legend()
# plt.minorticks_on()
# plt.show()
# plt.close()
 

# print(np.shape(ttime1os))
# print(np.shape(ttimeosna), np.shape(ttimeosmg))
# print(np.shape(ttime1))
# print(np.shape(ttimeirna), np.shape(ttimeirmg))
# plt.figure()
# plt.xlabel('time [hrs]')
# plt.ylabel('total photon counts')
# plt.title('single gaussian fitting region photon counts ~ '+str(2*r+1)+' pixels')
# plt.scatter(ttimeosna, tsumosna, c='r', label='Na-like Os')
# plt.scatter(ttimeirna, tsumirna, c='b', label='Na-like Ir')
# plt.scatter(ttimeosmg, tsumosmg, c='tab:orange', label='Mg-like Os')
# plt.scatter(ttimeirmg, tsumirmg, c='tab:purple', label='Mg-like Ir')
# plt.ylim(0, 150)
# plt.minorticks_on()
# plt.legend()
# plt.show()
# plt.close()



###Gaussian fitting results for absolute wavelength 
# print('#################################################################################')
# print('Os Na-like cal  ', np.sum(np.array(osna1cal)**2)/(len(osna1cal)**2), ' nm')
# print('Os Na-like stat : ', np.sum(np.array(osna1stat)**2)/(len(osna1stat)**2), ' nm')
# print('Os na-like sys : ', np.sum(np.array(osna1sys)**2)/(len(osna1sys)**2))
# print('Os Na-like 2nd cal : ', np.sum(np.array(osna2cal)**2)/(len(osna2cal)**2), ' nm')
# print('Os Na-like 2nd stat : ', np.sum(np.array(osna2stat)**2)/(len(osna2stat)**2), ' nm')
# print('Os Na-like 2nd sys : ', np.sum(np.array(osna2sys)**2)/(len(osna2sys)**2))

# print('ir Na-like cal : ', np.sum(np.array(irna1cal)**2)/(len(irna1cal)**2), ' nm')
# print('ir Na-like stat : ', np.sum(np.array(irna1stat)**2)/(len(irna1stat)**2), ' nm')
# print('ir na-like sys : ', np.sum(np.array(irna1sys)**2)/(len(irna1sys)**2))
# print('ir Na-like 2nd cal : ', np.sum(np.array(irna2cal)**2)/(len(irna2cal)**2), ' nm')
# print('ir Na-like 2nd stat : ', np.sum(np.array(irna2stat)**2)/(len(irna2stat)**2), ' nm')
# print('ir na-like 2nd sys : ', np.sum(np.array(irna2sys)**2)/(len(irna2sys)**2))

# print('Os mg-like cal : ', np.sum(np.array(osmg1cal)**2)/(len(osmg1cal)**2), ' nm')
# print('Os mg-like stat : ', np.sum(np.array(osmg1stat)**2)/(len(osmg1stat)**2), ' nm')
# print('Os mg-like sys : ', np.sum(np.array(osmg1sys)**2)/(len(osmg1sys)**2))
# print('ir mg-like cal : ', np.sum(np.array(irmg1cal)**2)/(len(irmg1cal)**2), ' nm')
# print('ir mg-like stat : ', np.sum(np.array(irmg1stat)**2)/(len(irmg1stat)**2), ' nm')
# print('ir mg-like sys : ', np.sum(np.array(irmg1sys)**2)/(len(irmg1sys)**2))
# print('#################################################################################')

##Centroid fitting results for absolute wavelength 
# print('#################################################################################')
# print('Os Na-like cal  ', np.sum(np.array(cosnasetcal)**2)/(len(cosnasetcal)**2), ' nm')
# print('Os Na-like stat : ', np.sum(np.array(cosnasetstat)**2)/(len(cosnasetstat)**2), ' nm')
# #print('Os na-like sys : ', np.sum(np.array(osna1sys)**2)/(len(osna1sys)**2))
# print('Os Na-like 2nd cal : ', np.sum(np.array(cosnasetcal2)**2)/(len(cosnasetcal2)**2), ' nm')
# print('Os Na-like 2nd stat : ', np.sum(np.array(cosnasetstat2)**2)/(len(cosnasetstat2)**2), ' nm')
# #print('Os Na-like 2nd sys : ', np.sum(np.array(osna2sys)**2)/(len(osna2sys)**2))

# print('ir Na-like cal : ', np.sum(np.array(cirnasetcal)**2)/(len(cirnasetcal)**2), ' nm')
# print('ir Na-like stat : ', np.sum(np.array(cirnasetstat)**2)/(len(cirnasetstat)**2), ' nm')
# #print('ir na-like sys : ', np.sum(np.array(irna1sys)**2)/(len(irna1sys)**2))
# print('ir Na-like 2nd cal : ', np.sum(np.array(cirnasetcal2)**2)/(len(cirnasetcal2)**2), ' nm')
# print('ir Na-like 2nd stat : ', np.sum(np.array(cirnasetstat2)**2)/(len(cirnasetstat2)**2), ' nm')
# #print('ir na-like 2nd sys : ', np.sum(np.array(irna2sys)**2)/(len(irna2sys)**2))

# print('Os mg-like cal : ', np.sum(np.array(cosmgsetcal)**2)/(len(cosmgsetcal)**2), ' nm')
# print('Os mg-like stat : ', np.sum(np.array(cosmgsetstat)**2)/(len(cosmgsetstat)**2), ' nm')
# #print('Os mg-like sys : ', np.sum(np.array(osmg1sys)**2)/(len(osmg1sys)**2))
# print('ir mg-like cal : ', np.sum(np.array(cirmgsetcal)**2)/(len(cirmgsetcal)**2), ' nm')
# print('ir mg-like stat : ', np.sum(np.array(cirmgsetstat)**2)/(len(cirmgsetstat)**2), ' nm')
# #print('ir mg-like sys : ', np.sum(np.array(irmg1sys)**2)/(len(irmg1sys)**2))
# print('#################################################################################')


##CETROID METHOD##
def CentPoly(specos, cenos, cenerros, specir, cenir, cenerrir):

    cdatResize = dataResizer2(cenos, cenerros, cenir, cenerrir, specir, specos)
    
    y1 = cdatResize['cenos_rs']
    y1e = cdatResize['cenerros_rs']
    y2 = cdatResize['cenir_rs']
    y2e = cdatResize['cenerrir_rs']
    x2 = cdatResize['specir_rs']
    x1 = cdatResize['specos_rs']

    osdatalen = np.shape(y1)[1]
    irdatalen = np.shape(y2)[1]
    if osdatalen >= irdatalen:
        arrlen = osdatalen
    else:
        arrlen = irdatalen

    data = np.empty([2,arrlen])
    x = np.empty([2,arrlen])
    err = np.empty([2,arrlen])

    data[0,:] = y1
    data[1,:] = y2

    x[0,:] = x1
    x[1,:] = x2

    err[0,:] = y1e
    err[1,:] = y2e

    resCP = dict()

    def poly(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d 

    def poly_dataset(params, i, x):
        a = params['a_%i' % (i+1)]
        b = params['b_%i' % (i+1)]
        c = params['c_%i' % (i+1)]
        d = params['d_%i' % (i+1)]
        return poly(x, a, b, c, d) 

    params = Parameters()

    def objective(params, x, data):
        ndata, _ = data.shape
        resid = 0.0*data[:]
        
        for i in range(ndata):
            resid[i, :] = (data[i, :] - poly_dataset(params, i, x[i,:])) / (err[i,:])
            
        return resid


    fit_params = Parameters()
    for iy, y in enumerate(data):
        fit_params.add('a_%i' % (iy+1), value=1e-9)
        fit_params.add('b_%i' % (iy+1), value=1e-8)
        fit_params.add('c_%i' % (iy+1), value=1e-2)
        fit_params.add('d_%i' % (iy+1), value=7)

    for iy in (1,1):
        fit_params['a_%i' % iy].expr = 'a_2'
        fit_params['b_%i' % iy].expr = 'b_2'
        fit_params['c_%i' % iy].expr = 'c_2'


    resultc = minimize(objective, fit_params, method='least_squares', args=(x,data), nan_policy='omit')
    params.update(resultc.params)
    if resultc.success is not True:
        print('flag@@@@@@@@@@')
    #xplot1 = np.linspace(np.min(x1), np.max(x1), num=1000)
    xplot1 = np.linspace(0, 50, num=1000)
    yplot1 = params['d_1'].value + params['c_1'].value * xplot1 + params['b_1'].value * xplot1**2 + params['a_1'].value * xplot1**3
    yplot2 = params['d_2'].value + params['c_1'].value * xplot1 + params['b_1'].value * xplot1**2 + params['a_1'].value * xplot1**3
    x1 = nankill1(x1)
    x2 = nankill1(x2)

    
    resCP['num'] = np.shape(x1)[1] + np.shape(x2)[1]
    resCP['covar'] = resultc.covar
    resCP['a_1'] = resultc.params['a_1'].value
    resCP['b_1'] = resultc.params['b_1'].value
    resCP['c_1'] = resultc.params['c_1'].value
    resCP['d_1'] = resultc.params['d_1'].value
    resCP['d_1e'] = resultc.params['d_1'].stderr

    resCP['a_2'] = resultc.params['a_2'].value
    resCP['b_2'] = resultc.params['b_2'].value
    resCP['c_2'] = resultc.params['c_2'].value
    resCP['d_2'] = resultc.params['d_2'].value
    resCP['d_2e'] = resultc.params['d_2'].stderr
    resCP['xplot1'] = xplot1
    #resCP['xplot2'] = xplot2
    resCP['yplot1'] = yplot1 
    resCP['yplot2'] = yplot2 
    resCP['Os-Ydata'] = cenos
    resCP['Os-Ydataerr'] = cenerros
    resCP['Os-Xdata'] = specos
    resCP['Ir-Ydata'] = cenir 
    resCP['Ir-Ydataerr'] = cenerrir 
    resCP['Ir-Xdata'] = specir 
    resCP['red-chi-sq'] = resultc.redchi
    resCP['chi-sq'] = resultc.chisqr
    #resCP['Osconfband'] = cbOs 
    #resCP['Irconfband'] = cbIr 

    return resCP


###First order systematic polynomial
osna = np.array(osna)
irna = np.array(irna)
osnae = np.array(osnae)
irnae = np.array(irnae)
irnatime = np.array(irnatime)
osnatime = np.array(osnatime)
syspoly = CentPoly(osnatime, osna, osnae, irnatime, irna, irnae)

osmg = np.array(osmg)
irmg = np.array(irmg)
osmge = np.array(osmge)
irmge = np.array(irmge)
irmgtime = np.array(irmgtime)
osmgtime = np.array(osmgtime)
mgsyspoly = CentPoly(osmgtime, osmg, osmge, irmgtime, irmg, irmge)
########################################################

##Second order systematic polynomial (gauss)
osna2 = np.array(osna2)
irna2 = np.array(irna2)
osnae2 = np.array(osnae2)
irnae2 = np.array(irnae2)
irnatime2 = np.array(irnatime2)
osnatime2 = np.array(osnatime2)
syspoly2 = CentPoly(osnatime2, osna2, osnae2, irnatime2, irna2, irnae2)

#print('abs wavelength 2nd order here: ')
#print(np.average(osna2, weights = 1 / np.array(osnae2)), ' +/- ', np.sqrt(np.sum(np.array(osnae2)**2))/len(osnae2))
#print(np.average(irna2, weights = 1 / np.array(irnae2)), ' +/- ', np.sqrt(np.sum(np.array(irnae2)**2))/len(irnae2))



##First order centroid polys 

osnacen = np.array(osnacen)
osnacenu = np.array(osnacenu)
osnacent = np.array(osnacent)
irnacen = np.array(irnacen)
irnacenu = np.array(irnacenu)
irnacent = np.array(irnacent)
osmgcen = np.array(osmgcen)
osmgcenu = np.array(osmgcenu)
osmgcent = np.array(osmgcent)
irmgcen = np.array(irmgcen)
irmgcenu = np.array(irmgcenu)
irmgcent = np.array(irmgcent)

nasyspolycen = CentPoly(osnacent, osnacen, osnacenu, irnacent, irnacen, irnacenu)
mgsyspolycen = CentPoly(osmgcent, osmgcen, osmgcenu, irmgcent, irmgcen, irmgcenu)
#####




#Second order Na centroid polynomial 

irna2c = np.array(irna2c) 
osna2c = np.array(osna2c) 
irnae2c = np.array(irnae2c)
osnae2c = np.array(osnae2c)
irnatime2c = np.array(irnatime2c)
osnatime2c = np.array(osnatime2c)
nasyspolycen2 = CentPoly(osnatime2c, osna2c, osnae2c, irnatime2c, irna2c, irnae2c)


#### 1st order Na-like Outlier Removal 

aa = osna 
ab = osnae 
ac = osnatime 
ba = irna
bb = irnae
bc = irnatime 
bad_lines = []
countout = 0

print('starting 1st order Na-like outlier detection')
OutlierDetection = OutlierDet(aa, ab, ac, ba, bb, bc, CentPoly, crit)


u = 0
while OutlierDetection['badlines'] is not None:

    c1 = OutlierDetection['osna']
    c2 = OutlierDetection['Oserr']
    c3 = OutlierDetection['osnatime']
    d1 = OutlierDetection['irna']
    d2 = OutlierDetection['Irerr'] 
    d3 = OutlierDetection['irnatime'] 
    bad_lines.append(OutlierDetection['badlines'])

    a1 = c1
    a2 = c2 
    a3 = c3 
    b1 = d1 
    b2 = d2 
    b3 = d3
    #print(b2)
    u +=1
    #print('outlier iteration number: ', u)
    if countout > 60:
        print('error outlier: too many removed')
        break
    else:
        countout +=1 

        OutlierDetection = OutlierDet(a1, a2, a3, b1, b2, b3, CentPoly, crit)


print('1st order Na-like outlier detection completed.')
print('Na-like bad lines/spectra: ', bad_lines)
###########################

aa2 = osna2 
ab2 = osnae2 
ac2 = osnatime2 
ba2 = irna2
bb2 = irnae2
bc2 = irnatime2 
bad_lines2 = []
countout2 = 0
print('starting 2nd order Na-like outlier detection')
OutlierDetection2 = OutlierDet(aa2, ab2, ac2, ba2, bb2, bc2, CentPoly, crit)


u = 0
while OutlierDetection2['badlines'] is not None:

    c12 = OutlierDetection2['osna']
    c22 = OutlierDetection2['Oserr']
    c32 = OutlierDetection2['osnatime']
    d12 = OutlierDetection2['irna']
    d22 = OutlierDetection2['Irerr'] 
    d32 = OutlierDetection2['irnatime'] 
    bad_lines2.append(OutlierDetection2['badlines'])


    a12 = c12
    a22 = c22 
    a32 = c32 
    b12 = d12 
    b22 = d22 
    b32 = d32
    #print(b2)
    u +=1
    #print('outlier iteration number: ', u)
    if countout2 > 60:
        print('error outlier: too many removed')
        break
    else:
        countout2 +=1 

        OutlierDetection2 = OutlierDet(a12, a22, a32, b12, b22, b32, CentPoly, crit)


print('2nd order Na-like outlier detection completed.')

#### Mg-like Outlier Removal 

dd = osmg 
de = osmge 
df = osmgtime 
ed = irmg
ee = irmge
ef = irmgtime 
mgbad_lines = []
countout = 0


print('starting Mg-like outlier detection')
OutlierDetectionMg = OutlierDet(dd, de, df, ed, ee, ef, CentPoly, crit)

u = 0
while OutlierDetectionMg['badlines'] is not None:
    u +=1
    #print('Mg-like outlier iteration number: ', u)
    g1 = OutlierDetectionMg['osna']
    g2 = OutlierDetectionMg['Oserr']
    g3 = OutlierDetectionMg['osnatime']
    h1 = OutlierDetectionMg['irna']
    h2 = OutlierDetectionMg['Irerr'] 
    h3 = OutlierDetectionMg['irnatime'] 
    mgbad_lines.append(OutlierDetectionMg['badlines'])

    dd = g1
    de = g2 
    df = g3 
    ed = h1 
    ee = h2 
    ef = h3
    
    #print(OutlierDetectionMg)

    
    if countout > 60:
        #print('error outlier: too many removed')
        break
    else:
        countout +=1 

        OutlierDetectionMg = OutlierDet(dd, de, df, ed, ee, ef, CentPoly, crit)

print('Mg-like outlier detection completed.')
#print('Mg-like bad lines/spectra: ', mgbad_lines)
################## Outlier removed dataset ##########################       Na-like 
nosna = nankill0(OutlierDetection['osna'])
nosnatime = nankill0(OutlierDetection['osnatime'])
nosnae = nankill0(OutlierDetection['Oserr'])
nirna = nankill0(OutlierDetection['irna'])
nirnatime = nankill0(OutlierDetection['irnatime'])
nirnae = nankill0(OutlierDetection['Irerr']) 
##


nosna2 = nankill0(OutlierDetection2['osna'])

nosnatime2 = nankill0(OutlierDetection2['osnatime'])
nosnae2 = nankill0(OutlierDetection2['Oserr'])
nirna2 = nankill0(OutlierDetection2['irna'])
nirnatime2 = nankill0(OutlierDetection2['irnatime'])
nirnae2 = nankill0(OutlierDetection2['Irerr']) 

##

nosmg = nankill0(OutlierDetectionMg['osna'])
nosmgtime = nankill0(OutlierDetectionMg['osnatime'])
nosmge = nankill0(OutlierDetectionMg['Oserr'])
nirmg = nankill0(OutlierDetectionMg['irna'])
nirmgtime = nankill0(OutlierDetectionMg['irnatime'])
nirmge = nankill0(OutlierDetectionMg['Irerr'])

#1st order 
# nOsNaavg = np.average(nosna, weights = 1 / np.array(nosnae))
# nOsMgavg = np.average(nosmg, weights = 1 / np.array(nosmge))
# nIrNaavg = np.average(nirna, weights = 1 / np.array(nirnae))
# nIrMgavg = np.average(nirmg, weights = 1 / np.array(nirmge))

nOsNaavg = np.average(osnaset, weights = 1 / np.array(osnasete))
nOsMgavg = np.average(osmgset, weights = 1 / np.array(osmgsete))
nIrNaavg = np.average(irnaset, weights = 1 / np.array(irnasete))
nIrMgavg = np.average(irmgset, weights = 1 / np.array(irmgsete))

nOsNaavge = np.sqrt(np.sum(np.array(osnasete)**2)) / len(osnaset)
nOsMgavge = np.sqrt(np.sum(np.array(osmgsete)**2)) / len(osmgset)
nIrNaavge = np.sqrt(np.sum(np.array(irnasete)**2)) / len(irnaset)
nIrMgavge = np.sqrt(np.sum(np.array(irmgsete)**2)) / len(irmgset)

#2nd order na-like 

# nOsNaavg2 = np.average(nosna2, weights = 1 / np.array(nosnae2))
# nIrNaavg2= np.average(nirna2, weights = 1 / np.array(nirnae2))

# nOsNaavge2 = np.sqrt(np.sum(np.array(nosnae2)**2)) / len(nosna2)
# nIrNaavge2 = np.sqrt(np.sum(np.array(nirnae2)**2)) / len(nirna2)

nOsNaavg2 = np.average(osnaset2, weights = 1 / np.array(osnasete2))
nIrNaavg2= np.average(irnaset2, weights = 1 / np.array(irnasete2))

nOsNaavge2 = np.sqrt(np.sum(np.array(osnasete2)**2)) / len(osnasete2)
nIrNaavge2 = np.sqrt(np.sum(np.array(irnasete2)**2)) / len(irnasete2)




# print('Centroid results: ')
# #print(np.average(osnacen, weights = 1 / np.array(osnacenu)), ' +/- ', np.sqrt(np.sum(np.array(osnacenu)**2))/len(osnacenu))
# print(np.average(osnacen), ' +/- ', np.sqrt(np.sum(np.array(osnacenu)**2))/len(osnacenu))
# print(np.average(osmgcen, weights = 1 / np.array(osmgcenu)), ' +/- ', np.sqrt(np.sum(np.array(osmgcenu)**2))/len(osmgcenu))
# print(np.average(irnacen, weights = 1 / np.array(irnacenu)), ' +/- ', np.sqrt(np.sum(np.array(irnacenu)**2))/len(irnacenu))
# print(np.average(irmgcen, weights = 1 / np.array(irmgcenu)), ' +/- ', np.sqrt(np.sum(np.array(irnacenu)**2))/len(irmgcenu))
# print('Absolute Na shift: ', -(JeV * hp * cs / (np.average(osnacen, weights = 1 / np.array(osnacenu))*10**(-9)))+(JeV*hp*cs/(np.average(irnacen, weights = 1 / np.array(irnacenu))*10**(-9))))
# print('Relative Na shift: ',-(JeV * hp * cs / (nasyspolycen['d_1']*10**(-9)))+(JeV*hp*cs/(nasyspolycen['d_2']*10**(-9))))
# print('Absolute Mg shift: ', -(JeV * hp * cs / (np.average(osmgcen, weights = 1 / np.array(osmgcenu))*10**(-9)))+(JeV*hp*cs/(np.average(irmgcen, weights = 1 / np.array(irmgcenu))*10**(-9))))
# print('Relative Mg shift: ',-(JeV * hp * cs / (mgsyspolycen['d_1']*10**(-9)))+(JeV*hp*cs/(mgsyspolycen['d_2']*10**(-9))))
# print('~~~')
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

print('Revised Centroid results for analysis type: '+str(style))
print('r == '+str(rcen))
print(np.average(csetosna, weights=1/np.array(csetosnae)), ' +/- ', np.sqrt(np.sum(np.array(csetosnae)**2))/len(csetosnae))
print(np.average(csetosmg, weights = 1 / np.array(csetosmge)), ' +/- ', np.sqrt(np.sum(np.array(csetosmge)**2))/len(csetosmge))
print(np.average(csetirna, weights = 1 / np.array(csetirnae)), ' +/- ', np.sqrt(np.sum(np.array(csetirnae)**2))/len(csetirnae))
print(np.average(csetirmg, weights = 1 / np.array(csetirmge)), ' +/- ', np.sqrt(np.sum(np.array(csetirmge)**2))/len(csetirmge))
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

print('Revised Centroid results: ')
print('r == '+str(rcen))
print(np.average(cosnaset, weights=1/np.array(osnasete)), ' +/- ', np.sqrt(np.sum(np.array(cosnasete)**2))/len(cosnasete))
print(np.average(cosmgset, weights = 1 / np.array(cosmgsete)), ' +/- ', np.sqrt(np.sum(np.array(cosmgsete)**2))/len(cosmgsete))
print(np.average(cirnaset, weights = 1 / np.array(cirnasete)), ' +/- ', np.sqrt(np.sum(np.array(cirnasete)**2))/len(cirnasete))
print(np.average(cirmgset, weights = 1 / np.array(cirmgsete)), ' +/- ', np.sqrt(np.sum(np.array(cirmgsete)**2))/len(cirmgsete))
print('Absolute Na shift: ', -(JeV * hp * cs / (np.average(cosnaset, weights=1/np.array(osnasete))*10**(-9)))+(JeV*hp*cs/(np.average(cirnaset, weights = 1 / np.array(cirnasete))*10**(-9))))
#print('Relative Na shift: ',-(JeV * hp * cs / (nasyspolycen['d_1']*10**(-9)))+(JeV*hp*cs/(nasyspolycen['d_2']*10**(-9))))
print('Absolute Mg shift: ', -(JeV * hp * cs / (np.average(cosmgset, weights = 1 / np.array(cosmgsete))*10**(-9)))+(JeV*hp*cs/(np.average(cirmgset, weights = 1 / np.array(cirmgsete))*10**(-9))))
#print('Relative Mg shift: ',-(JeV * hp * cs / (mgsyspolycen['d_1']*10**(-9)))+(JeV*hp*cs/(mgsyspolycen['d_2']*10**(-9))))

print('2nd order: ')
print(np.average(cosnaset2, weights=1/np.array(osnasete2)), ' +/- ', np.sqrt(np.sum(np.array(cosnasete2)**2))/len(cosnasete2))
print(np.average(cirnaset2, weights = 1 / np.array(cirnasete2)), ' +/- ', np.sqrt(np.sum(np.array(cirnasete2)**2))/len(cirnasete2))

print('~~~')

print(np.shape(qosmgset), np.shape(qosmgsete))
##Quad gauss fitting results for absolute wavelengths 
qnOsNaavg = np.average(qosnaset, weights = 1 / np.array(qosnasete))
qnOsMgavg = np.average(qosmgset, weights = 1 / np.array(qosmgsete))
#qnOsMgavg = np.average(qosmgset)
qnIrNaavg = np.average(qirnaset, weights = 1 / np.array(qirnasete))
qnIrMgavg = np.average(qirmgset, weights = 1 / np.array(qirmgsete))

qnOsNaavge = np.sqrt(np.sum(np.array(qosnasete)**2)) / len(qosnaset)
qnOsMgavge = np.sqrt(np.sum(np.array(qosmgsete)**2)) / len(qosmgset)
qnIrNaavge = np.sqrt(np.sum(np.array(qirnasete)**2)) / len(qirnaset)
qnIrMgavge = np.sqrt(np.sum(np.array(qirmgsete)**2)) / len(qirmgset)





# print('#################################################################################')
# print('Os Na-like cal  ', np.sum(np.array(osnasetcal)**2)/(len(osnasetcal)**2), ' nm')
# print('Os Na-like stat : ', np.sum(np.array(osnasetstat)**2)/(len(osnasetstat)**2), ' nm')
# # print('Os Na-like 2nd cal : ', np.sum(np.array(osnasetstat)**2)/(len(osna2cal)**2), ' nm')
# # print('Os Na-like 2nd stat : ', np.sum(np.array(osna2stat)**2)/(len(osna2stat)**2), ' nm')

# print('ir Na-like cal : ', np.sum(np.array(irnasetcal)**2)/(len(irnasetcal)**2), ' nm')
# print('ir Na-like stat : ', np.sum(np.array(irnasetstat)**2)/(len(irnasetstat)**2), ' nm')
# # print('ir Na-like 2nd cal : ', np.sum(np.array(irna2cal)**2)/(len(irna2cal)**2), ' nm')
# # print('ir Na-like 2nd stat : ', np.sum(np.array(irna2stat)**2)/(len(irna2stat)**2), ' nm')

# print('Os mg-like cal : ', np.sum(np.array(osmgsetcal)**2)/(len(osmgsetcal)**2), ' nm')
# print('Os mg-like stat : ', np.sum(np.array(osmgsetstat)**2)/(len(osmgsetstat)**2), ' nm')
# print('ir mg-like cal : ', np.sum(np.array(irmgsetcal)**2)/(len(irmgsetcal)**2), ' nm')
# print('ir mg-like stat : ', np.sum(np.array(irmgsetstat)**2)/(len(irmgsetstat)**2), ' nm')


# print('Os Na-like 2nd cal : ', np.sum(np.array(osnasetcal2)**2)/(len(osnasetcal2)**2), ' nm')
# print('Os Na-like 2nd stat : ', np.sum(np.array(osnasetstat2)**2)/(len(osnasetstat2)**2), ' nm')

# print('Ir Na-like 2nd cal : ', np.sum(np.array(irnasetcal2)**2)/(len(irnasetcal2)**2), ' nm')
# print('Ir Na-like 2nd stat : ', np.sum(np.array(irnasetstat2)**2)/(len(irnasetstat2)**2), ' nm')


# print('#################################################################################')

# print(osmgsetstat)
# print(irnasetstat)
# print(irmgsetstat)
#averaged for absolute wavelength, USE THESE (updated 4/19/2022)
print('#########################################################')
print('Absolute averaged wavelengths (single gaussian fit of subtracted spectra, 1st-order): ')
print('new Os Na avg: ', nOsNaavg, ' +/- ', nOsNaavge, ' nm')
print('new Os Mg avg: ', nOsMgavg, ' +/- ', nOsMgavge, ' nm')
print('new Ir Na avg: ', nIrNaavg, ' +/- ', nIrNaavge, ' nm')
print('new Ir Mg avg: ', nIrMgavg, ' +/- ', nIrMgavge, ' nm')



print('Absolute averaged wavelengths (single gaussian fit of subtracted spectra, 2nd-order): ')
print('new Os Na avg: ', nOsNaavg2, ' +/- ', nOsNaavge2, ' nm ', '(',nOsNaavg2/2,')')
print('new Ir Na avg: ', nIrNaavg2, ' +/- ', nIrNaavge2, ' nm', '(',nIrNaavg2/2,')')
print('#########################################################')

#averaged for absolute wavelength, USE THESE (updated 4/29/2022)
print('#########################################################')
print('Absolute averaged wavelengths (4 gaussian fit of subtracted spectra, 1st-order): ')
print('new Os Na avg: ', qnOsNaavg, ' +/- ', qnOsNaavge, ' nm')
print('new Os Mg avg: ', qnOsMgavg, ' +/- ', qnOsMgavge, ' nm')
print('new Ir Na avg: ', qnIrNaavg, ' +/- ', qnIrNaavge, ' nm')
print('new Ir Mg avg: ', qnIrMgavg, ' +/- ', qnIrMgavge, ' nm')

##First order poly results 
nd2 = OutlierDetection['d_2']
nd1 = OutlierDetection['d_1']
na1 = OutlierDetection['a_1']
nb1 = OutlierDetection['b_1']
nc1 = OutlierDetection['c_1'] 


nd2mg = OutlierDetectionMg['d_2']
nd1mg = OutlierDetectionMg['d_1']
na1mg = OutlierDetectionMg['a_1']
nb1mg = OutlierDetectionMg['b_1']
nc1mg = OutlierDetectionMg['c_1']

###Second order results
#GAUSS
nd22 = OutlierDetection2['d_2']
nd12 = OutlierDetection2['d_1']
na12 = OutlierDetection2['a_1']
nb12 = OutlierDetection2['b_1']
nc12 = OutlierDetection2['c_1']
#Centroid
nd22c = nasyspolycen2['d_2']
nd12c = nasyspolycen2['d_1'] 
#print(nd1-nd2)
ndE2 = (JeV * hp * cs / (nd22*10**(-9))) - (JeV*hp*cs / (nd12*10**(-9)))
#print(ndE2, nd12-nd22)
ndE2c = (JeV * hp * cs / (nd22c*10**(-9))) - (JeV*hp*cs / (nd12c*10**(-9)))

###plot for meeting update of systematic polynomial###
plt.figure()
plt.scatter(OutlierDetectionMg['syspoly']['Os-Xdata'], OutlierDetectionMg['syspoly']['Os-Ydata'], c='b', label='Os Mg-like')
plt.errorbar(OutlierDetectionMg['syspoly']['Os-Xdata'], OutlierDetectionMg['syspoly']['Os-Ydata'], OutlierDetectionMg['syspoly']['Os-Ydataerr'], c='b', ls='none')
plt.scatter(OutlierDetectionMg['syspoly']['Ir-Xdata'], OutlierDetectionMg['syspoly']['Ir-Ydata'], c='r', label='Ir Mg-like')
plt.errorbar(OutlierDetectionMg['syspoly']['Ir-Xdata'], OutlierDetectionMg['syspoly']['Ir-Ydata'],OutlierDetectionMg['syspoly']['Ir-Ydataerr'], c='r', ls='none')
plt.plot(OutlierDetectionMg['syspoly']['xplot1'], OutlierDetectionMg['syspoly']['yplot1'])
plt.plot(OutlierDetectionMg['syspoly']['xplot1'], OutlierDetectionMg['syspoly']['yplot2'])
plt.xlabel('time [hrs]')
plt.ylabel('Wavelength [nm]')
plt.title('Systematic polynomial of centroid positions')
plt.legend()
plt.minorticks_on()
#plt.show()
plt.close()

ndW = -nd2 + nd1 
ndE = (JeV * hp * cs / (nd2*10**(-9))) - (JeV*hp*cs / (nd1*10**(-9)))

ndWmg = -nd2mg + nd1mg
ndEmg = (JeV * hp * cs / (nd2mg*10**(-9))) - (JeV*hp*cs / (nd1mg*10**(-9)))

nOsres = poly(nosnatime, na1, nb1, nc1, nd1) - nosna    #residual na
nIrres = poly(nirnatime, na1, nb1, nc1, nd2) - nirna 

nOsresmg = poly(nosmgtime, na1mg, nb1mg, nc1mg, nd1mg) - nosmg      #residual mg
nIrresmg = poly(nirmgtime, na1mg, nb1mg, nc1mg, nd2mg) - nirmg 
##First order 
ntotalnum = np.shape(nosnatime)[0] + np.shape(nirnatime)[0] 
ntotalnummg = np.shape(nosmgtime)[0] + np.shape(nirmgtime)[0]
# nOsHisto = Histogram(nOsres, bins=9, bin_range = 0.005, num=ntotalnum)
# nIrHisto = Histogram(nIrres, bins=9, bin_range = 0.005, num=ntotalnum)
# nOsHistomg = Histogram(nOsresmg, bins=9, bin_range = 0.005, num=ntotalnummg)
# nIrHistomg = Histogram(nIrresmg, bins=9, bin_range = 0.005, num = ntotalnummg)

nOsHisto = Histogram(nOsres, bins=bins, bin_range = 0.005, num=np.shape(nosnatime)[0])
nIrHisto = Histogram(nIrres, bins=bins, bin_range = 0.005, num=np.shape(nirnatime)[0])
nOsHistomg = Histogram(nOsresmg, bins=7, bin_range = 0.005, num=np.shape(nosmgtime)[0])
nIrHistomg = Histogram(nIrresmg, bins=7, bin_range = 0.005, num = np.shape(nirmgtime)[0])



nosunc = nOsHisto['Line Position Error']
nirunc = nIrHisto['Line Position Error']
nosunccal = nOsHisto['mu val']
nirunccal = nIrHisto['mu val']
nosuncmg = nOsHistomg['Line Position Error']
niruncmg = nIrHistomg['Line Position Error']
nosuncmgcal = nOsHistomg['mu val']
niruncmgcal = nIrHistomg['mu val']

print('1st Na order wavelength diff: ')
print(ndW, np.sqrt(nosunc**2+nirunc**2 ))
print(nosunc, nirunc)
print('1st mg order wavelength diff: ')
print(ndWmg, np.sqrt(nosuncmg**2+niruncmg**2 ))
print('@@@@@@@@@')
print(ndWmg, nosuncmg, niruncmg)
print('@@@@@@@@@@@@@')

####
#Second order 

nOsres2 = poly(nosnatime2, na12, nb12, nc12, nd12) - nosna2
nIrres2 = poly(nirnatime2, na12, nb12, nc12, nd22) - nirna2
ntotalnum2 = np.shape(nosnatime2)[0] + np.shape(nirnatime2)[0]
# nOsHisto2 = Histogram(nOsres2,bins=bins, bin_range = 2*0.005, num=ntotalnum2)
# nIrHisto2 = Histogram(nIrres2,bins=bins,bin_range=2*0.005,num=ntotalnum2)
nOsHisto2 = Histogram(nOsres2,bins=bins, bin_range = 2*0.005, num=np.shape(nosnatime2)[0])
nIrHisto2 = Histogram(nIrres2,bins=bins,bin_range=2*0.005,num=np.shape(nirnatime2)[0])
nosunc2 = nOsHisto2['Line Position Error']
nirunc2 = nIrHisto2['Line Position Error']
# nosunc2cal = nOsHisto2['mu val']
# nirunc2cal = nIrHisto2['mu val']
ntotalunc2 = np.sqrt(nosunc2**2 + nirunc2**2)

ntotalunce2 = 2*ndE2*(ntotalunc2/((nd12-nd22)/2))
print('2nd order wavelength diff nm')
print(nd12-nd22, ntotalunc2)
print(nosunc2, nirunc2)
print('~~')
print('2nd order na-like shift (gaussian): ' ,ndE2*2,' +/- ', ntotalunce2, ' eV')
#print('2nd order na-like shift (centroid): ' ,ndE2c*2)
print('~~')

ntotalunc = np.sqrt((nosunc)**2 + (nirunc)**2 )
ntotalunce = ndE * (ntotalunc / ndW)

ntotaluncmg = np.sqrt((nosuncmg)**2 + (niruncmg)**2 )
ntotaluncemg = ndEmg * (ntotaluncmg / ndWmg)

nosxplot = np.linspace(np.min(nosnatime), np.max(nosnatime), num=1000)
nirxplot = np.linspace(np.min(nirnatime), np.max(nirnatime), num=1000) 

nosxplotmg = np.linspace(np.min(nosmgtime), np.max(nosmgtime), num=1000)
nirxplotmg = np.linspace(np.min(nirmgtime), np.max(nirmgtime), num=1000)


nosyplot = poly(nosxplot, na1, nb1, nc1, nd1)
niryplot = poly(nirxplot, na1, nb1, nc1, nd2) 

nosyplotmg = poly(nosxplotmg, na1mg, nb1mg, nc1mg, nd1mg)
niryplotmg = poly(nirxplotmg, na1mg, nb1mg, nc1mg, nd2mg)

#####################################################################

newdW = -syspoly['d_2'] + syspoly['d_1']
newdE = (JeV*hp*cs/(syspoly['d_2']*10**(-9))) - (JeV*hp*cs / (syspoly['d_1'] * 10**(-9)))
#print('newdW: ', newdW)
Osres = poly(osnatime, syspoly['a_1'], syspoly['b_1'], syspoly['c_1'], syspoly['d_1']) - osna 
Irres = poly(irnatime, syspoly['a_2'], syspoly['b_2'], syspoly['c_2'], syspoly['d_2']) - irna 

totalnum = np.shape(osnatime)[0] + np.shape(irnatime)[0]


OsHisto = Histogram(Osres, bins, bin_range=0.0025, num=np.shape(osnatime)[0])
IrHisto = Histogram(Irres, bins, bin_range=0.0025, num=np.shape(irnatime)[0])


# plt.figure() 
# plt.title('Histogram distribution Na-like')
# plt.plot(nOsHisto['xplot'], nOsHisto['yplot'], c='c')
# plt.scatter(nOsHisto['xdata'], nOsHisto['ydata'], c='b', label='Os Na-like')
# plt.plot(nIrHisto['xplot'], nIrHisto['yplot'], c='r')
# plt.scatter(nIrHisto['xdata'], nIrHisto['ydata'], c='tab:orange', label='Ir Na-like')
# plt.ylabel('Counts')
# plt.xlabel('Binned Residual (nm)')
# plt.minorticks_on() 
# plt.legend() 
# plt.show() 
# plt.close() 




# plt.figure() 
# plt.title('Histogram distribution Mg-like')
# plt.plot(nOsHistomg['xplot'], nOsHistomg['yplot'], c='c')
# plt.scatter(nOsHistomg['xdata'], nOsHistomg['ydata'], c='b', label='Os mg-like')
# plt.plot(nIrHistomg['xplot'], nIrHistomg['yplot'], c='r')
# plt.scatter(nIrHistomg['xdata'], nIrHistomg['ydata'], c='tab:orange', label='Ir mg-like')
# plt.ylabel('Counts')
# plt.xlabel('Binned Residual (nm)')
# plt.minorticks_on() 
# plt.legend() 
# plt.show() 
# plt.close() 


nosunc = nOsHisto['Line Position Error']
nirunc = nIrHisto['Line Position Error']
nosunccal = nOsHisto['mu val']
nirunccal = nIrHisto['mu val']
# print('here')
# print(nosunc, nirunc, nosunccal, nirunccal)
ntotalunc = np.sqrt(nosunc**2 + nirunc**2)
print(ntotalunc, nosunc, nirunc)
ntotalunce = newdE * (ntotalunc / newdW)

#print('Na-like new dE (pre-removal): ', newdE, ' +/- ', totalunce, ' eV')
print('Na-like new dE (outlier-free, syspoly): ', ndE, ' +/- ', ntotalunce, ' eV')
print('~~')
print('Mg-like new dE(outlier-free, syspoly): ', ndEmg , ' +/- ', ntotaluncemg, ' eV')

plt.figure() 
plt.title('Na-like #time dependence os and ir line centers (pre-removal)')
plt.scatter(osnatime, osna, c='b', label='Os')
plt.errorbar(osnatime, osna, yerr=osnae, ls='none')
plt.scatter(irnatime, irna, c='r', label='Ir')
plt.errorbar(irnatime, irna, yerr=irnae, ls='none')
plt.plot(syspoly['xplot1'], syspoly['yplot1'])
plt.plot(syspoly['xplot1'], syspoly['yplot2'])
plt.legend() 
plt.xlabel('time [hr]')
plt.ylabel('centroid [nm]')
#plt.ylim(7.25, 7.5)
#plt.show() 
plt.close() 



plt.figure() 
plt.title('Na-like Systematic Polynomial')
plt.scatter(nosnatime, nosna, c='b',label='Na-like Os')
plt.errorbar(nosnatime, nosna, yerr=nosnae, ls='none')
plt.scatter(nirnatime, nirna, c='r', label='Na-like Ir')
plt.errorbar(nirnatime, nirna, yerr=nirnae, ls='none')
plt.plot(nosxplot, nosyplot)
plt.plot(nirxplot, niryplot)
plt.xlabel('time [hr]')
plt.ylabel('centroid [nm]')
plt.legend()
plt.minorticks_on()
#plt.show() 
plt.close() 


# plt.figure() 
# plt.title('Mg-like Systematic Polynomial')
# plt.scatter(nosmgtime, nosmg, c='b', label='Mg-like Os')
# plt.errorbar(nosmgtime, nosmg, yerr=nosmge, ls='none')
# plt.scatter(nirmgtime, nirmg, c='r', label='Mg-like Ir')
# plt.errorbar(nirmgtime, nirmg, yerr=nirmge, ls='none')
# plt.plot(nosxplotmg, nosyplotmg)
# plt.plot(nirxplotmg, niryplotmg)
# plt.xlabel('time [hr]')
# plt.ylabel('centroid [nm]')
# plt.legend()
# plt.minorticks_on() 
# plt.show() 
# plt.close() 


xticklist = np.linspace(np.min(calibset), np.max(calibset), num=len(calibset))



OsNaavg = np.average(OsNa)
OsNaavge = np.sqrt(np.sum(np.array(OsNae)**2)) / len(OsNa)

IrNaavg = np.average(IrNa)
IrNaavge = np.sqrt(np.sum(np.array(IrNae)**2)) / len(IrNa)


OsMgavg = np.average(OsMg)
OsMgavge = np.sqrt(np.sum(np.array(OsMge)**2)) / len(OsMg) 

IrMgavg = np.average(IrMg)
IrMgavge = np.sqrt(np.sum(np.array(IrMge)**2)) / len(IrMg)

#print('Os Na (Quad-Gauss method avged spectra): ', OsNaavg, ' +/- ', OsNaavge, ' nm')


plt.figure()
plt.scatter(calibset, OsNa, label='Os Na')
plt.errorbar(calibset, OsNa, yerr=2*np.array(OsNae), ls='none')
plt.xticks(xticklist)
plt.xlabel('calibration set #')
plt.ylabel('wavelength (nm)')
plt.axhline(y=OsNaavg , c='r', ls='-', label='Average of "Os Na" (EBIT Jan 2020)')
plt.axhline(y=OsNaavg + 2*OsNaavge , c='r', ls='--')
plt.axhline(y=OsNaavg - 2*OsNaavge , c='r', ls='--')
plt.axhline(y=7.44616, c='tab:purple', ls='-', label='Dipti GRASP2K')
plt.axhline(y=7.44616 + 0.0013, c='tab:purple', ls='--')
plt.axhline(y=7.44616 - 0.0013, c='tab:purple', ls='--')
plt.axhline(y=JGOsNa, c='g', ls='-', label='Blundell/Gillaspy 2013')
plt.axhline(y=JGOsNa + JGOsNae, c='g', ls='--')
plt.axhline(y=JGOsNa - JGOsNae, c='g', ls='--')
plt.axhline(y=7.4508, c='k', ls='-', label='Sapirstein 2015')
plt.axhline(y=7.4508+JGOsNae, c='k', ls='--')
plt.axhline(y=7.4508-JGOsNae, c='k', ls='--')
plt.title('Os absolute wavelength Na-like')
plt.minorticks_on()
plt.legend() 
#plt.savefig(plotpath+str('OsNaAbswavelength.png'))
#plt.show() 
plt.close() 


#print('Ir Na (Quad-Gauss method, avged spectra):', IrNaavg, ' +/- ', IrNaavge, ' nm')

plt.figure()
plt.scatter(calibset, IrNa, label='Ir Na')
plt.errorbar(calibset, IrNa, yerr=2*np.array(IrNae), ls='none')
plt.axhline(y=IrNaavg, c='r', ls='-', label='Average of "Ir Na" (EBIT Jan 2020)')
plt.axhline(y=IrNaavg + 2*IrNaavge, c='r', ls='--')
plt.axhline(y=IrNaavg - 2*IrNaavge, c='r', ls='--')
plt.axhline(y=JGIrNa, c='g', ls='-', label='Blundell/Gillaspy 2013')
plt.axhline(y=JGIrNa+JGIrNae, c='g', ls='--')
plt.axhline(y=JGIrNa-JGIrNae, c='g', ls='--') 
plt.axhline(y=7.29198, c='tab:purple', ls='-', label='Dipti GRASP2K')
plt.axhline(y=7.29198+JGIrNae, c='tab:purple', ls='--')
plt.axhline(y=7.29198-JGIrNae, c='tab:purple', ls='--')
plt.axhline(y=SapIrNa, c='k', ls='-', label='Sapirstein 2015')
plt.axhline(y=SapIrNa+JGIrNae, c='k', ls='--')
plt.axhline(y=SapIrNa-JGIrNae, c='k', ls='--')
plt.xlabel('calibration set #')
plt.ylabel('wavelength (nm)')
plt.xticks(xticklist)
plt.legend() 
plt.title('Ir absolute wavelength Na-like')
plt.minorticks_on()
###plt.savefig(plotpath+str('IrNaAbswavelength.png'))
#plt.show() 
plt.close() 


#print('Os Mg (Quad-Gauss): ', OsMgavg, ' +/- ', OsMgavge, ' nm')

#print('Ir Mg(Quad-Gauss): ', IrMgavg, ' +/- ', IrMgavge, ' nm')

plt.figure()
plt.scatter(calibset, OsMg, label='Os Mg')
plt.errorbar(calibset, OsMg, yerr=2*np.array(OsMge), ls='none')
plt.axhline(y=OsMgavg, c='r', ls='-', label='Average of "Os Mg" (EBIT Jan 2020)')
plt.axhline(y=OsMgavg + 2*OsMgavge, c='r', ls='--')
plt.axhline(y=OsMgavg - 2*OsMgavge, c='r', ls='--')
plt.axhline(y=DiptiOsMg, c='k', ls='-', label='Dipti GRASP2K')
plt.axhline(y=DiptiOsMg+JGOsNae, c='k', ls='--')
plt.axhline(y=DiptiOsMg-JGOsNae, c='k', ls='--')
plt.xlabel('calibration set #')
plt.ylabel('wavelength (nm)')
plt.xticks(xticklist)
plt.legend() 
plt.title('Os absolute wavelength Mg-like')
plt.minorticks_on()
#plt.savefig(plotpath+str('OsMgAbswavelength.png'))
#plt.show() 
plt.close() 

plt.figure()
plt.scatter(calibset, IrMg, label='Ir Mg')
plt.errorbar(calibset, IrMg, yerr=2*np.array(IrMge), ls='none')
plt.axhline(y=IrMgavg, c='r', ls='-', label='Average of "Ir Mg" (EBIT Jan 2020)')
plt.axhline(y=IrMgavg + 2*IrMgavge, c='r', ls='--')
plt.axhline(y=IrMgavg - 2*IrMgavge, c='r', ls='--')
plt.axhline(y=DiptiIrMg, c='k', ls='-', label='Dipti GRASP2K')
plt.axhline(y=DiptiIrMg+JGOsNae, c='k', ls='--')
plt.axhline(y=DiptiIrMg-JGOsNae, c='k', ls='--')
plt.xlabel('calibration set #')
plt.ylabel('wavelength (nm)')
plt.xticks(xticklist)
plt.legend() 
plt.title('Ir absolute wavelngth Mg-like')
plt.minorticks_on()
#plt.savefig(plotpath+str('IrMgAbswavelength.png'))
#plt.show() 
plt.close() 



OsNaEnergy = JeV* hp * cs / (np.array(OsNa)* 10**(-9)) 
IrNaEnergy = JeV * hp * cs / (np.array(IrNa)* 10**(-9))

OsNaEnergye = OsNaEnergy * (OsNaavge / OsNaavg)
IrNaEnergye = IrNaEnergy * (IrNaavge / IrNaavg)


Osavgen = JeV * hp * cs / (OsNaavg * 10**(-9))
Iravgen = JeV * hp * cs / (IrNaavg * 10**(-9))

Osavgene = Osavgen * (OsNaavge / OsNaavg)
Iravgene = Iravgen * (IrNaavge / IrNaavg)

OsMgavgen = JeV * hp*cs / (OsMgavg * 10**(-9))
IrMgavgen = JeV * hp*cs / (IrMgavg * 10**(-9))

OsMgavgene = OsMgavgen * (OsMgavge / OsMgavg)
IrMgavgene = IrMgavgen * (IrMgavge / IrMgavg)


avgendiff = Iravgen - Osavgen
avgendiffe = np.sqrt((Osavgene)**2 + (Iravgene)**2)

avgendiffMg = IrMgavgen - OsMgavgen
avgendiffMge = np.sqrt((OsMgavgene)**2 + (IrMgavgene)**2)

endiff = IrNaEnergy - OsNaEnergy 
endiffe = np.sqrt(np.array(OsNaEnergye)**2 + np.array(IrNaEnergye)**2)
#print('Na-like avg difference (Quad-gauss): ', avgendiff, ' +/- ', avgendiffe)
#print('Mg-like avg difference (Quad-gauss): ', avgendiffMg, ' +/- ', avgendiffMge)

sAIrRMBPT = -3723
sAIrMg = -3616.29
sAIrMgdef = -3657.09
stevunc = np.sqrt(0.000025 + 7.21*10**(-8)*(ntotalunce * econv)**2)

sAIr = -3699.93
sAIrdef = -3741.68

DiptiIr = 170.028145
DiptiOs = 166.5075315

sIrMg = 165.434
sIrMge = 0.064

sOsMg = 161.924
sOsMge = 0.042


dr1Mg = econv * (avgendiffMg - sIrMg + sOsMg) / (sAIrMg)
dr2Mg = econv * (avgendiffMg - sIrMg + sOsMg) / (sAIrMgdef)

dr1 = econv * (avgendiff - DiptiIr + DiptiOs) / (sAIr)

dr2 = econv * (avgendiff - DiptiIr + DiptiOs) / (sAIrdef)

JGOsNa = 7.4491
irtheo = JeV* (hp * cs / (JGIrNa * 10**(-9)))
ostheo = JeV* (hp * cs / (JGOsNa * 10**(-9)))

#print('Steve diff: ', sIrMg - sOsMg)
# print('GRASP: ', dr1)
# print('steve unc: ', stevunc)
# print('GRASP Mg: ', dr1Mg)

#print(irtheo, ostheo)
#print(irtheo-ostheo)
dr3 = econv * (avgendiff - irtheo + ostheo) / (sAIrRMBPT)
#print('RMBPT: ', dr3)
#print('steve"s uncertainty: ', stevunc)