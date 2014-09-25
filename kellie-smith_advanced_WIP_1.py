# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------- #
# import modules

import numpy as np
from math import sin, cos, log, exp
import matplotlib.pyplot as plt
from scipy import integrate

# TODO
# Tth ueberfueheren in #ith
# #impact und utility implementieren


# ------------------------------------------------------------------------- #
# set parameter values

cpi     = 280.  # atmospheric carbon dioxide concentration at 
			    # pre-industrial level
dt2c    = 3.    # climate sensitivity (temperature increase for C = 2 * CPI)
tau_T   = 50.   # charaCTEAEtaristic climate time scale of temperature 
			    # (heat capacity)
tau_c   = 5000.   # charaCTEAEtaristic time scale of carbon uptake (ocean and land)
phi     = tau_T/tau_c # rate of tau
alpha   = dt2c/log(2)#
eta0     = 0.025 # carbon intensity
mu0      = 0.1  # decarbonization rate
xi      = 0.02  # background economic growth rate
# delta   = 0.5   # what is delta? -> ein Fragment aus Kellie-Smith
w0 = 160.       # initial global wealth in trillion dollars
#----------------------------------------------------------------------------#
Tth     = 0.9/alpha # Threshold for onset of awareness, normiert!
mumax   = 0.08 # maximum decarbonization rate per year 
v       = 1.  # Vulnerability
cA      = 0.02 # Einkopplung Temp anstieg in awareness pro jahr; bestimmt, wie schnell die Sättigung 
#                   der decarbonization-rate eintritt
a       = tau_T*cA*v*alpha  # irgendein geiler parameter 
Kw      = 0.36 # Einkopplung Schaden Wirtschaft, geil!!
tau_I  =   25. #  
r_T_Tpunkt_on_I = 0.5  ## 0 = nur Tpunkt, 1 = nur T
tau_A   = 20 # in Jahren, equilibrium time scale for awareness
s_AT    = 0.5 # in 1/K, asymptotic sensitivity of awareness
phi_A   = tau_T/tau_A # time scale ratio
# Auf jeden Fall auch für r = 0 testen, für welche Konstellationen aus 
# Schadensrate auf die Wirtschaft und Zeitskala für die Wirkung der Awareness

#-----------------------------------------------------------------------------#
# ------------------------------------------------------------------------- #
# set final integration time and timestep

tf = 300./tau_T    # total integration time, aber! normierte Zeit!!!
dt = 1/tau_T     # delta t [years]
ts = tf / dt  # total number of timesteps
# tv = 1:1:ts
tv = np.linspace (0., tf, ts )  #  time vector

#------------------------------------------------------------------------------#
# set initial values (non real)

C0 = (380./cpi -1)
T0 = 0.7/alpha
E0 = eta0 * w0 * tau_T / cpi
A0 = 0.
Eta0 = eta0/eta0

CTEAEtainit=np.array([C0,T0,E0,A0,Eta0])

CTEAEta=np.zeros((ts,5))
CTEAEta[0]=CTEAEtainit

#------------------------------------------------------------------------------#
#definiere Funktionen zum Lösen der DGL,Berechnen der echten(nicht-normierten) Größen und Darstellen

def mu(A): 
    to_return = mu0 + (mumax-mu0)*(1-exp(-A))
    return to_return


def CTEAEtadot(C,T,E,A,Eta,ti): # (C,T,E,A)-punkt als vektor geschrieben, normierte DGL, siehe Kelly-Smith (2.7)-(2.9)
    
    Tpunkt = log(1+C) - T
    
    if T > Tth :
        
            Apunkt =  a * ( r_T_Tpunkt_on_I  * (T-Tth) +  (1 - r_T_Tpunkt_on_I ) * tau_I/tau_T * Tpunkt   ) - phi_A *(A - s_AT* alpha *T)
    else:
        Apunkt = 0.
        
    # end if
    
    Deri  =  np.array (  [E-phi*C , \
                        Tpunkt , \
                        tau_T * E * (xi * (1 - Kw * alpha * T) - mu(A)) , \
                        Apunkt,\
                        - mu(A)*tau_T *Eta ]  )
    return Deri
        
def IntegrateRK4(L): # numiersche Integration, hier mit Runge-Kutta-4ter-Ordnung
    for i in range(1, int(ts)):
        #
        K1=CTEAEtadot(L[i-1,0],L[i-1,1],L[i-1,2],L[i-1,3],L[i-1,4],(i-1)*dt)
        #
        Ltemp1=L[i-1]+0.5*K1*dt
        #
        K2  =  CTEAEtadot ( Ltemp1[0], Ltemp1[1], Ltemp1[2], Ltemp1[3], Ltemp1[4], (i-1)*dt+0.5*dt )
        #
        Ltemp2=L[i-1]+0.5*K2*dt
        #
        K3=CTEAEtadot(Ltemp2[0],Ltemp2[1],Ltemp2[2],Ltemp2[3],Ltemp2[4],(i-1)*dt+0.5*dt)
        #
        Ltemp3=L[i-1]+K3*dt
        #
        K4=CTEAEtadot(Ltemp3[0],Ltemp3[1],Ltemp3[2],Ltemp3[3],Ltemp3[4],i*dt)
        #
        Kall=dt*(K1+2*K2+2*K3+K4)/6
        #
        L[i]=L[i-1]+Kall
        


def realCTEAEta(L): # von normierten Größen zu echten Größen
    global CTEAEtareal # erzeugt eine globale Variable außerhalb der Funktion
    CTEAEtareal=np.transpose(np.array([cpi*(L[:,0])+np.ones(ts) , alpha*L[:,1] , (cpi/tau_T)*L[:,2] , L[:,3] , eta0*L[:,4]]))
    global tv
    tv=tau_T*tv# rechnet normierte Zeit auf echte Zeit zurück
    
def realW(L):# berechnet Wreal=Wealth aus Ereal (realCTEAEta)
    global Wreal
    Wreal=np.zeros(ts)
    for i in range(1,int(ts)+1):
        Wreal[i-1]=(L[i-1,2]/L[i-1,4])
    return Wreal 

def plotCTEAEtaW(L):# stellt C,T,E und W dar
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(611)
    ax2 = fig.add_subplot(612)
    ax3 = fig.add_subplot(613)
    ax4 = fig.add_subplot(614)
    ax5 = fig.add_subplot(615)
    ax6 = fig.add_subplot(616)

    ax1. plot(tv, L[:,2], label="E")
    ax2. plot(tv, L[:,0], label="C")
    ax3. plot(tv, L[:,1], label="T")
    ax4. plot(tv, realW(L), label="W")
    ax5. plot(tv, L[:,3], label="A")
    ax6. plot(tv, L[:,4], label="Eta")

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()


    plt.show()
    #fig.savefig("Kellie-smith_1.eps")             
#------------------------------------------------------------------------------#
# Alles ausführen und freuen ;)

IntegrateRK4(CTEAEta)# löst normierte DGL
realCTEAEta(CTEAEta)# erzeugt CTEAEtareal
plotCTEAEtaW(CTEAEtareal)# stellt alles dar


