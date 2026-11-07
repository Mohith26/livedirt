!git -C /content/livedirt pull -q && git -C /content/livedirt log --oneline -1
%cd /content/livedirt
import subprocess, time
Q = "--wall-mu 0.21 --side-W 0.15 --side-mu 0.21"
runs2 = [
  ("q2d-const-dx2.5mm", f"--dx 0.0025 {Q}"),
  ("q2d-muI-dx2.5mm", f"--dx 0.0025 --rheology muI {Q}"),
  ("q2d-dptan-dx2.5mm", f"--dx 0.0025 --yield-fit dp_tan {Q}"),
  ("q2d-const-dx1.25mm", f"--dx 0.00125 {Q}"),
  ("q2d-muI-dx1.25mm", f"--dx 0.00125 --rheology muI {Q}"),
  ("q2d-const-phi22.3", f"--dx 0.0025 --phi 22.3 {Q}"),
  ("q2d-const-phi24.7", f"--dx 0.0025 --phi 24.7 {Q}"),
]
for tag, a in runs2:
    t0 = time.time()
    out = subprocess.run(f"python -m digbench.run_collapse --arch cuda --tag {tag} {a}", shell=True, capture_output=True, text=True)
    lines = [l for l in (out.stdout + out.stderr).splitlines() if not l.startswith("[Taichi]")]
    print(f"=== {tag} ({time.time()-t0:.0f}s)"); print("\n".join(lines[-10:]), flush=True)
!cd /content/livedirt/digbench/results && zip -q -r /content/digbench-results.zip . && ls -la /content/digbench-results.zip
from google.colab import files
files.download("/content/digbench-results.zip")
