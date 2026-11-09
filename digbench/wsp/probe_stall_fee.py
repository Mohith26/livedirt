import glob, numpy as np, json, sys, math
sys.path.insert(0,'/home/ubuntu/livedirt')
from digbench.attitude import lip_offset
fs=sorted(glob.glob('/home/ubuntu/v4/scoops/campaign/*.npz'))+sorted(glob.glob('/home/ubuntu/v4/scoops/loop-adaptive/*.npz'))
X=[];y=[];comp=[];runs=[];fstop=[]
C=0.05
for f in fs:
    z=np.load(f); T=z['terrain'].astype(float); M=float(z['margin']); P=z['poses']; meta=json.loads(str(z['meta'])); soil=json.loads(str(z['soil']))
    phi=math.radians(float(soil.get('phi',30))); c=float(soil.get('c',0)); rho=float(soil.get('rho',1800))
    j=int((1.0+M)/C)  # bucket centreline column
    d=[]; 
    for row in P:
        lx,lz=lip_offset(row[3]); x=row[1]+lx; zl=row[2]+lz
        i=int((x+M)/C)
        if 0<=i<T.shape[0]: d.append(max(0.0,T[i,j-8:j+9].mean()-zl))
        else: d.append(0.0)
    d=np.array(d)
    Kp=math.tan(math.pi/4+phi/2)**2
    # FEE-style resistance per unit width (passive wedge): 0.5 gamma d^2 Kp + 2 c d sqrt(Kp), width 1 m
    F=0.5*rho*9.81*d**2*Kp+2*c*d*math.sqrt(Kp)
    swept=np.cumsum(d)*np.r_[0,np.abs(np.diff(P[:,1]))].mean()
    X.append([F.max()/1e3, d.max(), swept[-1], math.degrees(phi), math.log10(1+c)]); 
    y.append(meta['completion']<0.9); comp.append(meta['completion']); runs.append(f.split('/')[-1].rsplit('_',1)[0])
    # predicted stop: first pose where F exceeds a threshold, as fraction of path
    fstop.append(F)
X=np.array(X); y=np.array(y); comp=np.array(comp)
print('bites',len(y),'stalled',y.sum(), 'base rate %.2f'%y.mean())
# single-feature threshold on peak FEE force (leave-one-run-out)
ur=sorted(set(runs)); runs=np.array(runs); acc=[]; 
pred=np.zeros(len(y),bool)
for r in ur:
    tr=runs!=r; te=runs==r
    best=None
    for th in np.quantile(X[tr,0],np.linspace(0.05,0.95,91)):
        a=((X[tr,0]>th)==y[tr]).mean()
        if best is None or a>best[0]: best=(a,th)
    pred[te]=X[te,0]>best[1]
print('stall classification from peak FEE force alone (held out by run): accuracy %.2f, recall %.2f, precision %.2f' % ((pred==y).mean(), (pred&y).sum()/max(1,y.sum()), (pred&y).sum()/max(1,pred.sum())))
# where it stops: for stalled bites, predicted stop fraction = first index F > th_global vs actual completion
th=np.quantile(X[:,0],1-y.mean())
err=[]
for k in np.where(y)[0]:
    F=fstop[k]; idx=np.argmax(F>th*1e3) if (F>th*1e3).any() else len(F)-1
    err.append(abs(idx/len(F)-comp[k]))
print('stalled bites: |predicted stop fraction - actual completion| median %.2f, mean %.2f (actual completion mean %.2f)'%(np.median(err),np.mean(err),comp[y].mean()))
