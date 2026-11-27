"""Compare interior minima to the faster endpoint; a=1 sets the time unit."""
from pathlib import Path
from itertools import product
import math
import csv
import numpy as np
from stages import direct_cycle_decimal

def main():
    K=7*math.pi*math.sqrt(3)/6
    bound=math.exp(-K-1)/K
    rows=[]
    # Positive clockwise and counterclockwise rates for all scanned b.
    for n in (9,12,24,60,240,math.inf):
        best=None
        for b,r in product(np.linspace(.35,.577,24),np.geomspace(.008,2.,55)):
            c=np.linspace(.025,.975,77); d=1+r*(1-c)
            theta=1.5*np.pi+np.arctan(b/(1+r))-2*np.arctan(b/d)
            if np.isinf(n):
                tau=theta/b
                flight=np.exp(-r*tau)
                amp=np.exp(-(1+r)*tau)
                k=np.full(c.size,np.nan)
            else:
                k=b/np.tan(theta/n)-1-r
                keep=k>0
                if not keep.any(): continue
                c,d,theta,k=c[keep],d[keep],theta[keep],k[keep]
                tau=n/k
                flight=np.exp(-n*np.log1p(r/k))
                amp=np.exp(n*(np.log(k)-np.log(np.hypot(k+1+r,b))))
            # Dimensionless endpoint gaps: each equals (T(endpoint)-T*) / T(0).
            factor=2*r*r*b*np.hypot(1+r,b)*amp/((3-flight)*(d*d+b*b))
            gap0=factor*c*c/((1+r)**2+b*b)
            gap1=factor*(1-c)**2/(1+b*b)
            gain=np.minimum(gap0,gap1)/(1+np.minimum(0.,gap1-gap0))
            at=np.argmax(gain)
            entry=[gain[at],b,r,c[at],k[at],tau[at]]
            if best is None or entry[0]>best[0]:best=entry
        g,b,r,c,k,tau=best
        assert 0<g<bound
        if n!=math.inf:
            f,back=1/3+b/math.sqrt(3),1/3-b/math.sqrt(3)
            ec=direct_cycle_decimal(k,r,f,back,n,c)
            e1=direct_cycle_decimal(k,r,f,back,n,1.)
            baseline=(3*math.exp(n*math.log1p(r/k))-1)/r
            np.testing.assert_allclose(g,(min(0.,e1)-ec)/(baseline+min(0.,e1)),rtol=1e-8)
        rows.append([n,b,r,c,k,tau,g])
        print(n, 'gain',g,'c',c,'r',r,'b',b)
    with Path(__file__).with_name('gain.csv').open('w',newline='') as file:
        w=csv.writer(file);w.writerow(['stages','b_over_a','r_over_a','retention','kappa_over_a','mean_flight_in_a_units','fractional_gain_vs_better_endpoint']);w.writerows(rows)
if __name__=='__main__': main()
