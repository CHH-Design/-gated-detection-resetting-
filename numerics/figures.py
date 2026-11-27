"""Rebuild transport.pdf with NumPy and ReportLab; run reproduce.py first."""
from pathlib import Path
import csv
import math
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white
from reproduce import spectral_time


BASE = Path(__file__).resolve().parent
BLUE, RED = '#14577C', '#A64A27'
INK, GRID = '#202A30', '#DFE4E6'


def main():
    pdf = canvas.Canvas(str(BASE / 'transport.pdf'), pagesize=(460, 185))
    pdf.setTitle('Gate retention and Erlang stage thresholds')
    pdf.setLineJoin(1)
    pdf.setLineCap(1)

    def text(x, y, label, size=8, weight='Helvetica', align='left'):
        pdf.setFillColor(HexColor(INK))
        pdf.setFont(weight, size)
        method = {'left': pdf.drawString, 'center': pdf.drawCentredString,
                  'right': pdf.drawRightString}[align]
        method(x, y, label)

    def line(points, color=INK, width=0.75, dash=False):
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(width)
        pdf.setDash([3.5, 2.0] if dash else [])
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
        pdf.drawPath(path)
        pdf.setDash([])

    def axis(left, right, xspan, yspan, xticks, yticks, xlabel, ylabel):
        bottom, top = 32, 143
        X = lambda x: left + (right-left)*(x-xspan[0])/(xspan[1]-xspan[0])
        Y = lambda y: bottom + (top-bottom)*(y-yspan[0])/(yspan[1]-yspan[0])
        for x in xticks:
            line([(X(x), bottom-2.5), (X(x), bottom)], width=0.65)
            text(X(x), bottom-12, f'{x:g}', align='center')
        for y in yticks:
            line([(left, Y(y)), (right, Y(y))], GRID, width=0.55)
            text(left-5, Y(y)-2.8, str(y), align='right')
        line([(left,top),(left,bottom),(right,bottom)])
        text((left+right)/2, 5, xlabel, 9, align='center')
        pdf.saveState()
        pdf.translate(left-26, (top+bottom)/2)
        pdf.rotate(90)
        text(0, 0, ylabel, 8.8, align='center')
        pdf.restoreState()
        return X, Y

    def legend(x, label, color, dashed):
        line([(x,157), (x+14,157)], color, 1.5, dashed)
        text(x+18,154,label,8)

    def marker(x, y, closed, color):
        pdf.setStrokeColor(HexColor(color))
        pdf.setFillColor(HexColor(color) if closed else white)
        pdf.setLineWidth(1.15)
        if closed:
            pdf.circle(x,y,2.1,stroke=1,fill=1)
        else:
            pdf.rect(x-2.25,y-2.25,4.5,4.5,stroke=1,fill=1)

    text(8,172,'(a)  Mean detection time',11,'Helvetica-Bold')
    legend(43, 'Driven', BLUE, False)
    legend(111, 'Reversible', RED, True)
    X, Y = axis(42, 219, (-.025,1.025), (124.4,139),
                [0,.25,.5,.75,1], [125,128,131,134,137],
                'Retention probability, c', 'Mean time, T')
    for forward, backward, color, dashed in [(1,.05,BLUE,False),(.525,.525,RED,True)]:
        points = [(X(c),Y(spectral_time(12,forward,backward,4,.2,c)))
                  for c in np.linspace(0,1,151)]
        line(points,color,1.65,dashed)
    with (BASE/'checks.csv').open(encoding='utf-8') as file:
        for row in csv.DictReader(file):
            c = float(row['retention'])
            driven = row['gate']=='driven'
            color = BLUE if driven else RED
            x = X(c) + ((-1.5 if driven else 1.5) if c==0 else 0)
            low, high = Y(float(row['ci95_low'])), Y(float(row['ci95_high']))
            line([(x,low),(x,high)],color,.95)
            for y in (low, high):
                line([(x-2.5,y),(x+2.5,y)],color,.95)
            marker(x,Y(float(row['mc_time'])),driven,color)

    text(239,172,'(b)  Erlang stage threshold',11,'Helvetica-Bold')
    legend(272, 'b/a = 0.57', BLUE, False)
    legend(361, 'b/a = 0.40', RED, True)
    X, Y = axis(273,451,(0,1.02),(7,23),[0,.25,.5,.75,1],
                [8,12,16,20], 'Scaled reset rate, r/a', 'Minimum stages')
    for ratio, color, dashed in [(.57,BLUE,False),(.4,RED,True)]:
        angle = 1.5*math.pi - 2*math.atan(ratio)
        def stages(x):
            return 2 + math.floor(angle / math.atan(ratio / (1+x)))
        lo, hi = .02, 1.
        n = stages(lo)
        points = [(X(lo),Y(n))]
        for k in range(n-1, stages(hi)-1):
            jump = ratio/math.tan(angle/k)-1
            if lo < jump < hi:
                points.extend([(X(jump),Y(n)), (X(jump),Y(n+1))])
                n += 1
        points.append((X(hi),Y(n)))
        line(points,color,1.6,dashed)
    pdf.showPage()
    pdf.save()


if __name__ == '__main__':
    main()
