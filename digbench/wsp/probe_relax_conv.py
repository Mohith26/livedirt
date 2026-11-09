import glob, numpy as np, torch, torch.nn.functional as Fn, sys, math, time, json
sys.path.insert(0,'/home/ubuntu/livedirt')
from digbench.scoop_model import relax, swept_envelope
# hand-written weights: forward difference kernels and their transposes (the "physicist" writes these, no training)
kx=torch.tensor([[[[-1.0],[1.0]]]],dtype=torch.float64); ky=torch.tensor([[[[-1.0, 1.0]]]],dtype=torch.float64)
class RelaxLayer(torch.nn.Module):
    """One relax iteration = 2 fixed convs (differences), a ReLU (excess over the repose slope), 2 fixed transposed
    convs (move material). Exactly mass-conserving; same update order as the reference (x then y)."""
    def __init__(self, iters=200): super().__init__(); self.iters=iters
    def forward(self, H, lim):
        for _ in range(self.iters):
            d=Fn.conv2d(H,kx); ex=torch.sign(d)*Fn.relu(d.abs()-lim)*0.25
            H=H+Fn.conv_transpose2d(ex,-kx)
            d=Fn.conv2d(H,ky); ex=torch.sign(d)*Fn.relu(d.abs()-lim)*0.25
            H=H+Fn.conv_transpose2d(ex,-ky)
        return H
fs=sorted(glob.glob('/home/ubuntu/v4/scoops/loop-adaptive/gate*.npz'))[:20]
L=RelaxLayer(200); errs=[]; masserr=[]
for f in fs:
    z=np.load(f); T=z['terrain'].astype(float); env=swept_envelope(z['poses'],*T.shape,float(z['margin'])); phi=json.loads(str(z['soil'])).get('phi',30)
    H0=np.where(np.isfinite(env),np.minimum(T,env),T)  # cut
    ref=relax(H0,phi,0.05,200)
    lim=math.tan(math.radians(phi))*0.05
    out=L(torch.tensor(H0)[None,None],lim)[0,0].numpy()
    errs.append(np.abs(out-ref).max()); masserr.append(abs(out.sum()-H0.sum())*0.0025)
print('max |conv layer - reference| over 20 bites: %.2e m; mass change %.2e m3' % (max(errs), max(masserr)))
