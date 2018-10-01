# BSM call price and volatility calibration
import numpy as np
import pandas as pd
import scipy.optimize as sop
import math
import scipy.stats as ss
import time
# parameter setting
# Today: Sep 27, 2018
# Maturity: Dec 31, 2018
S0=290.68 #Spot price
K=288 #strike
r=0.02 #interest rate
Cm=9.23 #Market call price
T=95/365
sigma=0.3
def d1f(St, K, t, T, r, sigma):
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
          * (T - t)) / (sigma * math.sqrt(T - t))
    return d1
def BSM_call_value(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * ss.norm.cdf(d1) - math.exp(-r * (T - t)) * K * ss.norm.cdf(d2)
    return call_value
# calculate BSM call price with given sigma
BSM_CV=BSM_call_value(S0, K, 0, T, r, sigma)
print('the BSM call price with 30% volatility=', BSM_CV)
# C0=19.77, Cm=9.23, C0>>Cm
# calibration the implied volatility 
def BSM_error_function(sigma):
    global min_RMSE
    model_value = BSM_call_value(S0, K, 0, T, r, sigma)
    RMSE = math.sqrt((model_value-Cm)**2)
    min_RMSE = min(min_RMSE, RMSE)
    return RMSE
i = 0  # counter initialization
min_RMSE = 200  # minimal RMSE initialization
opt = sop.fmin(BSM_error_function, 0.5,
               maxiter=500, maxfun=750,
               xtol=0.000001, ftol=0.000001)
print('the implied volatility=',opt[0])
# the implied volatility= 0.11823408603668178  #the implied volatility with given call price
# check
BSM_call_value(S0, K, 0, T, r, 0.11823409)
9.230000740041135   #with the volatility=11.8%, the BSM call price=market call price
