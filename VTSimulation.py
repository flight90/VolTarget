import numpy as np
import pandas as pd
import scipy.stats

rng = np.random.default_rng()

class heston():
    def __init__(self,v0,theta,correl,k,vov):
        self.v0 = v0
        self.theta = theta
        self.correl = correl
        self.k = k
        self.vov = vov

def realizedVol(prices:pd.Series):
    l = len(prices)-1
    s1 = prices[:-1].reset_index(drop=True)
    s2 = prices[1:].reset_index(drop=True)
    logret = np.log(s2/s1)
    var=logret*logret

    return np.sqrt(var.sum()/l*252)


class VT():
    def __init__(self,s0,w0,vt,lookback):
        self.s0 = s0
        self.w0 = w0
        self.vt = vt
        self.lookback = lookback

    def calculate(self,riskyseries):
        start = self.lookback+1
        initialvol = realizedVol(riskyseries[:start])
        self.w0 = self.vt/initialvol
        risky0 = riskyseries[start-1]

        remaining = riskyseries[start:].reset_index(drop=True)

        s = self.s0
        w=self.w0

        s_series = [s]
        w_series = [w]

        for i in range(len(remaining)):
            risky1 = remaining[i]
            s = s*(1+w*(risky1/risky0-1))
            s_series.append(s)

            rv = realizedVol(riskyseries[i+1:i+1+start])
            w=self.vt/rv

            w_series.append(w)

            risky0 = risky1

        return pd.Series(s_series),pd.Series(w_series)

def simulation(s0,volModel:heston,steps):
    s = s0
    v = volModel.v0

    theta = volModel.theta
    correl = volModel.correl
    k = volModel.k
    vov = volModel.vov

    spot = [s]
    vol=[v]

    sqrt252 = np.sqrt(252)

    for i in range(steps):
        z1,z2 = rng.standard_normal(2)
        zv = correl * z1 +np.sqrt(1-correl*correl)*z2

        s = s*v*z1/sqrt252+s
        v = np.sqrt(abs(v*vov*zv/sqrt252+v*v+k/252*(theta*theta-v*v)))

        spot.append(s)
        vol.append(v)

    return pd.Series(spot),pd.Series(vol)

def simRealizedVolVT(h,vt):
    s,v= simulation(100,h,252)
    vts,vtw = vt.calculate(s)
    return realizedVol(vts)

if __name__ == '__main__':
    vt = VT(100,0.3,0.05,10)
    for i in range(1):
        vov = 0.5*(i/10)
        hestonVol = heston(0.15,0.15,-0.8,20,vov)
        sim = [simRealizedVolVT(hestonVol,vt) for i in range(100)]

        volVT = np.average(sim)
        std = np.std(sim)
        n=len(sim)
        h = std * scipy.stats.t.ppf((1+0.95)/2, n-1)/np.sqrt(n)

        print("for vol of vol = ", vov)
        print("vol of VT = ", volVT," with 95% CI +- ", h)



