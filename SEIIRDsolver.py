import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.integrate import quad
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import sys
import matplotlib.dates as mdates
from timeit import default_timer as timer
np.set_printoptions(threshold = sys.maxsize)
pd.options.display.max_rows = 999
import xlrd
from matplotlib.dates import DateFormatter

N = 10230000
#N = 340000000
Ir_0 = 1
Io_0 = 5
E_0 = 10#Io_0
Ro_0 = 0
Rr_0 = 0
D_0 = 0
R_0 = 0
S_0 = N - E_0 - Io_0 - Ir_0

D0 = 2338
dd = 5.5
t1 = 22.5

START_DATE = {
  'Japan': '1/22/20',
  'Italy': '1/31/20',
  'Spain': '1/31/20',
  'Republic of Korea': '1/22/20',
  'Iran (Islamic Republic of)': '2/19/20',
  'Sweden': '1/31/20',
  'Sweden2': '2/4/20',
  'Denmark': '2/27/20',
  'Norway': '1/31/20',
  'US': '1/22/20',
  'China': '1/22/20',
  'Germany': '1/27/20',
  'Hubei': '1/22/20',
  'France': '1/24/20'
}

END_DATE = {
  'Japan': '',
  'Italy': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y')[1:],#'4/7/20',
  'Spain': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y')[1:],
  'Republic of Korea': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y')[1:],
  'Iran (Islamic Republic of)': '',
  'Sweden2': '3/11/20',
  'Sweden': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y').lstrip("0").replace("/0","/"), #[1:], #'4/13/20', 
  'Denmark': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y')[1:],
  'Norway': '',
  'US': datetime.strftime(datetime.today() - timedelta(1), '%m/%d/%y').lstrip("0").replace("/0","/"), #[1:],
  'China': '',
  'Germany': '',
  'Hubei': '',
  'France': datetime.strftime(datetime.today() - timedelta(0), '%m/%d/%y')[1:]
}
class Learner(object):
    def __init__(self, country, country2, loss):
        self.country = country
        self.country2 = country2
        self.loss = loss
        
    def load_confirmed(self, country):
      df = pd.read_csv('data/time_series_covid19_confirmed_global.csv')
      #print(df[df['Country/Region'] == 'Sweden'].iloc[0].loc['5/4/20'])
      #sys.exit()
      country_df = df[df['Country/Region'] == country].groupby('Country/Region').sum()
      #country_df = df[df['Province/State'] == country]
      return country_df.iloc[0].loc[START_DATE[country]:END_DATE[country]]#/0.1

    def load_recovered(self, country):
      df = pd.read_csv('data/time_series_covid19_recovered_global.csv')
      country_df = df[df['Country/Region'] == country].groupby('Country/Region').sum()
      #country_df = df[df['Province/State'] == country]
      return country_df.iloc[0].loc[START_DATE[country]:END_DATE[country]]#/0.1      
    def load_dead(self, country):
      df = pd.read_csv('data/time_series_covid19_deaths_global.csv')
      country_df = df[df['Country/Region'] == country].groupby('Country/Region').sum()
      #country_df = df[df['Province/State'] == country]
      return country_df.iloc[0].loc[START_DATE[country]:END_DATE[country]]

    def extend_index(self, index, new_size):
        values = index.values
        #print(index[-1])
        #sys.exit()
        current = index[-1]
        #current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            #values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
            values = np.append(values, current)
        return values

    def predict(self,psi, d, k, t0, gamma_o, gamma_r, q0, sigma, p0, Z, tau, phi, alpha, Kappa, data, recovered, deaths, country):
        predict_range = 220
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        def SEIIRD(t,y):
            S,E,Io,Ir,R,D = y
            b = beta(t,psi,d,k,t0)
            dSdt = -b*S*Ir/N - b*q0*S*Io/N
            dEdt = b*S*Ir/N + b*q0*S*Io/N - sigma*E
            dIodt = p0*sigma*E - (1-Z-Kappa)*gamma_o*Io - Z*tau*Io - Kappa*Io 
            dIrdt = (1-p0)*sigma*E - (1-phi)*gamma_r*Ir - phi*alpha*Ir + Kappa*Io
            dRdt = (1-Z-Kappa)*gamma_o*Io +(1-phi)*gamma_r*Ir
            dDdt = Z*tau*Io + phi*alpha*Ir 
            print(Z*tau*Io,phi*alpha*Ir)
            #print(S,E,Io,Ir,R,D)
            return dSdt, dEdt, dIodt, dIrdt, dRdt, dDdt
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_deaths = np.concatenate((deaths.values, [None] * (size - len(deaths.values))))
        return new_index, extended_actual, extended_recovered, extended_deaths, solve_ivp(SEIIRD, [0, size], [S_0,E_0,Io_0,Ir_0,R_0,D_0], t_eval=np.arange(0, size, 1))


#    def predict(self, psi, d, k, t0, q0, sigma, p0, gamma, data, recovered, deaths, country):
#        predict_range = 150
#        new_index = self.extend_index(data.index, predict_range)
#        size = len(new_index)
#        def SEIIRRD(t, y):
#            S,E,Ir,Io,R,D = y
#            b = beta(t,psi,d,k,t0)
#            dSdt = -b*S*Ir/N - b*q0*S*Io/N
#            dEdt = b*S*Ir/N + b*q0*S*Io/N - sigma*E
#            dIodt = p0*sigma*E - (1-tau)*gamma*Io - tau*Io
#            dIrdt = (1-p0)*sigma*E - (1-zeta)*gamma*Ir + zeta*Ir
#            dRodt = p1*(1-alpha)*((1-tau)*gamma*Io + (1-zeta)*gamma*Ir)
#            dRrdt = (1-p1)*(1-alpha)*((1-tau)*gamma*Io + (1-zeta)*gamma*Ir)
#            dDdt = alpha*(zeta*Ir + tau*Io)
#            R0 = (1-p0)*b/gamma + p0*q0*b/gamma
            #dDdtval = dDdt(t) #alpha*rho*dIdt*(t-t0)
#            return dSdt, dEdt, dIodt, dIrdt, dRodt,dRrdt, dDdt, R0
#        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
#        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
#        extended_deaths = np.concatenate((deaths.values, [None] * (size - len(deaths.values))))
#        return new_index, extended_actual, extended_recovered, extended_deaths, solve_ivp(SEIIRRD, [0, size], [S_0,E_0,Io_0,Ir_0,Ro_0,Rr_0,D_0,2.5], t_eval=np.arange(0, size, 1))

    def train(self):
        data = self.load_confirmed(self.country)
        recovered = self.load_recovered(self.country)
        deaths = self.load_dead(self.country)
        data = (data - deaths)
        lag = 12
        FHMdf = adjusted_data()
        print(FHMdf)
        fhmfile = open('fhmcoviddata.dat',"w")
        fhmfile.write(str(FHMdf))
        fhmfile.close()
        print('hej')
        sys.exit()
        FHM_I, FHM_Deaths, FHM_IVA = FHMdf['I'], FHMdf['PD'], FHMdf['IVA']
        FHM_I = FHM_I - FHM_Deaths - recovered[6:]
        
        #init = [1.5, 0.15, 0.2, 42, 1/8, 1/5, 0.55,1/5.1, 0.9, 0.000000001, 0.001, 0.01, 0.001, 0.01]
        #bound = [(1,15),(0.0000001,0.9999),(0.000001,0.9), (42,42), (1/5, 1/5),(1/5, 1/5),(0.001, 1),(1/6, 0.8), (0.89, 0.987), (0.0000000000001, 0.4), (0.00000001,0.5), (0.00001, 0.99), (0.0000001, 0.3), (0.001, 0.5)] #Not set

        #init = [2.84, 0.15, 0.23, 42, 1/5, 1/5, 0.55, 1/5.1, 0.987, 0.001, 0.001, 0.01, 0.001, 0]
        #bound = [(0,5),(0.01,0.9),(0.01,0.9), (41,43), (1/7, 1/5),(1/7, 1/5),(0.1, 0.9),(1/6, 1/4), (0.1, 0.987), (0.00000001, 1), (0.0000001,1), (0.00001, 0.99), (0.0000001, 0.3), (0, 0)] 

        init = [2.84, 0.15, 0.23, 42, 1/5, 1/5, 0.55, 1/5.1, 0.987, 0, 0, 0.1, 0.1, 0] #FHM plus flex, deaths from both groups
        bound = [(2.84,2.84),(0.01,0.9),(0.23,23), (30,60), (1/5, 1/5),(1/5, 1/5),(0.1, 0.99),(1/5.1, 1/5.1), (0.1, 0.987), (0, 0), (0,0), (0.1, 1), (0.1, 0.1), (0, 0)]

        #init = [2.84, 0.14, 0.23, 42, 1/5, 1/5, 0.55, 1/5.1, 0.987, 0, 0, 0.0317, 0.24, 0]
        #fig1init = [2.84, 0.14, 0.23, 42, 1/5, 1/5, 0.55, 1/5.1, 0.987, 0, 0, 0, 0, 0.001]
        #fig1bound = [(2,3),(0.1, 0.99),(0.01,0.99), (42,42), (1/20, 1/5),(1/5, 1/5),(0.1, 0.99),(1/5.1, 1/5.1), (0.9, 0.999), (0,0), (0,0), (0, 0), (0, 0),(0.0001,0.999)] #(0.0317, 0.0317), (0.24, 0.24), (0.0001,0.99)]
        #bound = [(2,3),(0.1, 0.99),(0.01,0.99), (42,42), (1/20, 1/5),(1/5, 1/5),(0.1, 0.99),(1/5.1, 1/5.1), (0.9, 0.999), (0,0), (0,0), (0.0001, 0.), (0.999, 0.001),(0,0)] #(0.0317, 0.0317), (0.24, 0.24), (0.0001,0.99)] #no Kappa, no Iodeath
        #fig2init = [2.1, 0.199, 0.079, 42, 0.0772, 0.2, 0.2935, 1/5.1, 0.9986, 0, 0, 0.01, 0.1, 0.04215]
        #fig2bound = [(2.1,2.1),(0.199, 0.199),(0.079,0.079), (42,42), (0.0772, 0.0772),(0.2, 0.2),(0.2935, 0.2935),(1/5.1, 1/5.1), (0.9986, 0.9986), (0.0001,1), (0.0001,1), (0.000001, 1), (0.1, 0.1),(0.04215,0.04215)] #Fitted SEIIR values
        #fig2bound = [(0.5,2.1),(0.15, 0.25),(0.079,0.079), (42,42), (0.0772, 0.0772),(0.2, 0.2),(0.2935, 0.2935),(1/5.1, 1/5.1), (0.1, 0.9986), (0.0001,1), (0.0001,1), (0.000001, 1), (0.09, 0.11),(0.04215,0.04215)] #(0.0317, 0.0317), (0.24, 0.24), (0,0)] #FHM
        #bound = [(2.84,2.84),(0.01,0.99),(0.23,0.23), (42,42), (1/5, 1/5),(1/5, 1/5),(0.55, 0.55),(1/5.1, 1/5.1), (0.987, 0.987), (0,0), (0,0), (0.00001, 0.99), (0.0000001, 0.99), (0, 0)] #FHM with flex
        optimal = minimize(loss, init, args=(FHM_I, recovered, FHM_Deaths), method='L-BFGS-B', bounds = bound)
        print(optimal)
        psi, d, k,t0, gamma_o, gamma_r, q0, sigma, p0, Z, tau, phi, alpha, Kappa = optimal.x
        new_index, extended_actual, extended_recovered, extended_deaths, prediction = self.predict(psi, d, k,t0, gamma_o, gamma_r, q0, sigma, p0, Z, tau, phi, alpha, Kappa, FHM_I[:-lag],recovered, FHM_Deaths[:-lag], self.country) #,dd,D0,t1
        print('Deaths: ',prediction.y[5][-1])
        print('Total infections: ',prediction.y[5][-1] + prediction.y[4][-1])
        #dday.values = np.insert(dday.values,0, [0]*50)
        #dday = extend_data(new_index, dday)
        df = pd.DataFrame({'$New_{Confirmed}$': extended_actual, 'Recovered':extended_recovered, 'Dead':extended_deaths,'S': prediction.y[0], 'E': prediction.y[1],'$I_{unknown}$': prediction.y[2],'$I_{confirmed}$': prediction.y[3], 'R': prediction.y[4], 'D': prediction.y[5]}, index=new_index)
        for i in [10000, 40000, 200000, 1000000, 1500000, N]:
            fig, ax = plt.subplots(figsize=(15, 10))
            #ax.set_xticks(df.index)
            #ax = df.plot(x_compat = True)
            #ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.ylim([0,i])
            styles = ['b-','g-','k-','r--','--','b--','g--','k--','-','-']
            lws = [1,1,4,2,2,2,2,2,2,2]
            for col, style, lw in zip(df.columns,styles, lws):
                df[col].plot(style = style, lw=lw, ax=ax)
            ax.legend(loc = 'upper right', prop = {'size': 15}) 
        #ax.grid(which = 'major', linestyle = '-',linewidth = '0.5',color = 'blue')
        #ax.grid(which = 'minor', linestyle = ':',linewidth = '0.5',color = 'black')
        #ax.set_axisbelow(True)
        #ax.minorticks_on()
        #print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
            fig.savefig(f"{self.country}_SEIIRD" + str(i) + ".png", format = 'png')
            fig.savefig(f"{self.country}_SEIIRD" + str(i) + ".eps", format = 'eps')
            plt.clf()
            
        wfile = open(self.country + 'SEIIRD_fig1.dat','w')
        wf = '{: >13}'*(len(prediction.y) + 1)
        wfile.write(wf.format(*['Day','S','E','Io','Ir','R','D'])+'\n')
        parwrite = ['theta = ' + str(psi) + ', delta = ' + str(d) + ', eps = ' + str(k) + ', t_b = ' + str(t0) + ', gamma_o = ' + str(gamma_o) + ', gamma_r = ' + str(gamma_r) + ', q_0 = ' + str(q0)  + ', sigma = ' + str(sigma) + ', p_0 = ' +str(p0) +  ', Z = '  + str(Z) + ', tau = ' + str(tau) + ', phi = ' + str(phi) + ', alpha = ' + str(alpha) + ', K = ' + str(Kappa)]
        wf2 = '{: >13}'*len(parwrite)
        wfile.write(wf2.format(*parwrite)+ '\n')
        for i in range(len(prediction.y[1])):
            wr = np.array([round(x[i]) for x in prediction.y])
            wr = np.insert(wr,0, int(i))
            wfile.write(wf.format(*wr) + '\n')
        wfile.close()
        sys.exit()
        plt.clf()
        df = pd.DataFrame({'Recovered': extended_recovered, 'R': prediction.y[3], 'D': deaths}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country, fontsize = 30)
        df.plot(ax=ax, style = ['-','-'], linewidth = 3, fontsize = 20)
        ax.legend(loc = 'upper right', prop = {'size': 10})
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.country}_SEIIR.png")

def loss2(point, deaths):
    D0, dd, t1 = point
    t = [i for i in range(len(deaths))]
    #t = np.linspace(1,len(deaths))
    y = De(t,D0,dd,t1)  
    #print(len(y),len(deaths),len(t))
    #sys.exit()
    return np.sqrt((np.mean(y - deaths)**2))


def beta(t,psi,d,k,t0):
    return psi*(d + (1 - d)/(1+np.exp(-k*(t0-t))))

def dDdtf(t):
    #if t > 50:
    #    tt = t - 50
    val = (D0/dd)*np.exp((t1-t)/dd)/(1+np.exp((t1-t)/dd))**2
    #else:
    #    val = 0
    
    
    #sys.exit()
    return val

def De(t,D0, dd, t1):
    return D0/(1+np.exp((t1-t)/dd))

def loss(point, data, recovered, deaths):
    size = len(data)
    psi, d, k, t0, gamma_o, gamma_r, q0, sigma, p0, Z, tau, phi, alpha, Kappa = point 
    def SEIIRD(t,y):
        S,E,Io,Ir,R,D = y
        b = beta(t,psi,d,k,t0)
        dSdt = -b*S*Ir/N - b*q0*S*Io/N
        dEdt = b*S*Ir/N + b*q0*S*Io/N - sigma*E
        dIodt = p0*sigma*E - (1-Z-Kappa)*gamma_o*Io - Z*tau*Io - Kappa*Io 
        dIrdt = (1-p0)*sigma*E - (1-phi)*gamma_r*Ir - phi*alpha*Ir + Kappa*Io
        dRdt = (1-Z-Kappa)*gamma_o*Io +(1-phi)*gamma_r*Ir
        dDdt = Z*tau*Io + phi*alpha*Ir 
        return dSdt, dEdt, dIodt, dIrdt, dRdt, dDdt
    start = timer()
    solution = solve_ivp(SEIIRD, [0, size], [S_0,E_0,Io_0,Ir_0, R_0, D_0], t_eval=np.arange(0, size, 1), method = 'RK23',vectorized=True)
    stop = timer()
    #print(stop - start)
    l1 = np.sqrt(np.mean((solution.y[3] - data.values)**2))
    #l1 = np.sqrt(np.mean((confirmed - data)**2))
    #l2 = np.sqrt(np.mean((solution.y[4] - recovered)**2))
    l3 = np.sqrt(np.mean((solution.y[5] - deaths)**2))
    a = 0.1
    val = np.sqrt(np.mean((solution.y[3] - data)**2 + 5*(solution.y[5] - deaths)**2))
    #val = np.sqrt(np.mean((solution.y[3] - data)**2 + (solution.y[4] - recovered)**2 + (solution.y[5] - deaths)**2))
    
    print(l3)
    return l3
    #return val
    #return a*l1 + (1-a)/2*(2*l3)

def extract_data(country, area):
    datalist = []
    df = pd.read_csv('data/time_series_covid19_confirmed_global.csv')
    country_df = df[df[area] == country]
    confirmed = country_df.iloc[0].loc[START_DATE[country]:]
    datalist.append(confirmed.array)
    dates = confirmed.axes[0]
    df = pd.read_csv('data/time_series_covid19_recovered_global.csv')
    country_df = df[df[area] == country]
    recovered = country_df.iloc[0].loc[START_DATE[country]:]
    datalist.append(recovered.array)
    df = pd.read_csv('data/time_series_covid19_deaths_global.csv')
    country_df = df[df[area] == country]
    deaths = country_df.iloc[0].loc[START_DATE[country]:]
    datalist.append(deaths.array)
      
def adjusted_data():
    df = pd.read_excel('data/Folkhalsomyndigheten_Covid19_review.xlsx', sheet_name = 'Antal avlidna per dag', usecols = "A:N").iloc[2:80]
    df.columns = ['col' + str(i) for i in range(len(df.columns))]
    df.index = df['col1']
    df = df.drop(['col0', 'col1', 'col2', 'col3','col7','col8','col9','col10'], axis = 1)
    df.columns = ['col' + str(i) for i in range(len(df.columns))]
    #df['col1'].loc[2:38]= pd.to_datetime(df['col1']).dt.date
    df2 = pd.read_excel('data/Folkhalsomyndigheten_Covid19.xlsx', sheet_name = 'Antal per dag region', usecols = "A:B").iloc[2:]
    df2.columns = ['col1', 'col2']
    df2['col1'] = [pd.Timestamp(x) for x in df2['col1']]
    df2.index = df2['col1']
    #df2.index = pd.to_datetime(df2['col1'])
    df2 = df2.drop('col1', axis = 1)
    for col in range(len(df.columns)):
        df2.insert(col + 1, "new" + str(col + 2), df["col" + str(col)], True)
    df2.columns = ['col' + str(i) for i in range(len(df2.columns))]
    Itot = [sum(df2['col0'].values[:i + 1]) for i in range(len(df2['col0']))]
    df2.insert(1, "new", Itot, True)
    colnames = ['I/d', 'I', 'PD', 'Plag','PD/d','D', 'D/d', 'dD/rp']  
    df2.columns = [colnames[i] for i in range(len(df2.columns))]
    df3 = pd.read_excel('data/Folkhalsomyndigheten_Covid19.xlsx', sheet_name = 'Antal intensivvårdade per dag', usecols = "A:B")
    df3.columns = ['col1', 'col2']
    df3.index = df3['col1']
    df3 = df3.drop('col1', axis = 1)
    df2.insert(0, "IVA", df3['col2'], True)
    df2 = df2.fillna(0)
    folkfile = open('folkhalsomyndigheten.dat', "w")
    folkfile.write(str(df2))
    return df2

def deaths_by_age():
    for i in range(1,36):
        df = pd.read_excel('FHM\Folkhalsomyndigheten_Covid19 (' + str(i) + '5).xlsx', sheet_name = 'Totalt antal per åldersgrupp', usecols = "A:D")
        print(df)
        sys.exit()
    

def extend_data(new_index, data):
    values = data.values
    i = 0
    for i in range(len(new_index)):
        if i < 43:
            values = np.insert(values,0,0)
        else:
            values = np.append(values, None)
        if len(values) >= len(new_index):
            break
    return values

#learner = Learner('Japan', loss)
#learner.train()
#learner = Learner('Germany', loss)
#learner.train()
#learner = Learner('Korea,South','Sweden', loss)
#learner.train()
#learner = Learner('Italy','Sweden', loss)
#learner.train()
#learner = Learner('Spain', 'Sweden', loss)
#learner.train()
#learner = Learner('Iran (Islamic Republic of)', loss)
#learner.train()
learner = Learner('Sweden', 'Sweden2', loss)
learner.train()
#learner = Learner('France', loss)
#learner.train()
#learner = Learner('Denmark','Sweden', loss)
#learner.train()
#learner = Learner('Norway', loss)
#learner.train()
#learner = Learner('California', loss)
#learner.train()
#learner = Learner('Minnesota', loss)
#learner.train()
#learner = Learner('US', 'Sweden', loss)
#learner.train()
sys.exit()
countries = ['Sweden', 'Norway', 'Italy', 'Spain', 'US', 'Japan']
provinces = ['Hubei']
for c in range(len(countries + provinces)):
    titles = []
    titles.extend(['','I', 'R','D'])
    tstr = '{: >17} ' + '{: >8} '*3
    fstr = '{: >36} '
    if c > len(countries) - 1 :
        d = c - len(countries)
        datafile = open(provinces[d] + '.dat','w')
        datafile.write(fstr.format(*[provinces[d]]) + '\n')
        datafile.write(tstr.format(*titles) + '\n')
        extract_data([provinces[d]],'Province/State')    
    else:
        datafile = open(countries[c] + '.dat','w')
        datafile.write(fstr.format(*[countries[c]]) + '\n')
        datafile.write(tstr.format(*titles) + '\n')
        extract_data([countries[c]], 'Country/Region')    
    datafile.close()


