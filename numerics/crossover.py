"""Reproduce the retention crossover and the two independent-sample contrasts."""
import csv
from pathlib import Path
from math import hypot
import numpy as np
from reportlab.pdfgen import canvas
from layer import excess_per_state, layer_limit

HERE = Path(__file__).resolve().parent
RATES = (1., .05, 2., .5)

def root(f):
    lo, hi = 0., 12.
    assert f(lo) < 0 < f(hi)
    for _ in range(52):
        mid = (lo+hi)/2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2

def main():
    pdf = canvas.Canvas(str(HERE/'crossover.pdf'), pagesize=(420,175))
    X = lambda z: 47+29*z
    Y = lambda d: 28+280*(d+.37)
    pdf.setFont('Helvetica',8)
    for d in [-.3,-.2,-.1,0,.1]:
        pdf.setStrokeGray(.4 if d==0 else .86)
        pdf.line(X(0),Y(d),X(12),Y(d))
        pdf.drawRightString(40,Y(d)-3,f'{d:.1f}')
    for z in range(0,13,2):
        pdf.drawCentredString(X(z),17,str(z))
    pdf.drawCentredString(215,4,'Scaled refresh rate, z = m r (1 - c)')
    pdf.saveState()
    pdf.translate(12,94); pdf.rotate(90)
    pdf.drawCentredString(0,0,'Excess time per state'); pdf.restoreState()
    colors = [(0.66,.26,.12),(.13,.40,.54),(.12,.13,.16)]
    with (HERE/'crossover.csv').open('w',newline='') as stream:
        out=csv.writer(stream)
        out.writerow(['number_of_states','z','excess_time_per_state'])
        for i,m in enumerate([64,256,None]):
            f = (lambda z: layer_limit(z,*RATES[:3])) if m is None else (
                lambda z: excess_per_state(m,1-z/(m*RATES[3]),*RATES))
            name = str(m) if m else 'limit'
            print(name,root(f))
            pdf.setStrokeColorRGB(*colors[i]); pdf.setDash([4,2] if i==1 else [])
            pdf.setLineWidth(1.6 if m is None else 1.)
            path=pdf.beginPath()
            for j,z in enumerate(np.linspace(0,12,121)):
                d=f(z); out.writerow([name,z,d])
                (path.moveTo if j==0 else path.lineTo)(X(z),Y(d))
            pdf.drawPath(path)
            pdf.line(49+i*91,164,66+i*91,164)
            pdf.drawString(70+i*91,161,'m = '+name if m else 'Large-ring limit')
    pdf.save()
    with (HERE/'checks.csv').open() as file:
        rows=list(csv.DictReader(file))
    for gate in ('driven','reversible'):
        a,b=[r for r in rows if r['gate']==gate and float(r['retention']) in (0,1)]
        d=float(b['mc_time'])-float(a['mc_time'])
        half=1.96*hypot(float(a['mc_se']),float(b['mc_se']))
        print(gate,d,d-half,d+half)

if __name__=='__main__':
    main()
