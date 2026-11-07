!nvidia-smi --query-gpu=name,memory.total --format=csv
!rm -rf /content/livedirt && git clone -q https://github.com/Mohith26/livedirt.git /content/livedirt && git -C /content/livedirt log --oneline -1
!pip -q install taichi==1.7.4
%cd /content/livedirt
import subprocess, time
runs = [
  ("mpm-dx2.5mm", "--dx 0.0025"),
  ("mpm-dx5mm", "--dx 0.005"),
  ("mpm-dx1.25mm", "--dx 0.00125"),
  ("abl-klar3d", "--dx 0.0025 --yield-fit klar3d"),
  ("abl-phi22.3", "--dx 0.0025 --phi 22.3"),
  ("abl-phi24.7", "--dx 0.0025 --phi 24.7"),
  ("abl-E4e6", "--dx 0.0025 --E 4e6"),
  ("abl-coulombfloor", "--dx 0.0025 --floor coulomb --floor-mu 0.462"),
]
for tag, a in runs:
    t0 = time.time()
    out = subprocess.run(f"python -m digbench.run_collapse --arch cuda --tag {tag} {a}", shell=True, capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if not l.startswith("[Taichi]")]
    print(f"=== {tag} ({time.time()-t0:.0f}s)"); print("\n".join(lines[-10:]), flush=True)
