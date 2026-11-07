"""Print results as base64 chunks between markers so they can be read from the Colab page (no downloads)."""
import base64, glob, io, sys, zipfile
pat = sys.argv[1] if len(sys.argv) > 1 else "digbench/results/*.json"
part = int(sys.argv[2]) if len(sys.argv) > 2 else 0
size = int(sys.argv[3]) if len(sys.argv) > 3 else 400_000
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(pat)):
        z.write(f, f.split("/")[-1])
b = base64.b64encode(buf.getvalue()).decode()
n = (len(b) + size - 1) // size
print(f"<<<B64 part={part} of={n} total={len(b)}>>>" + b[part * size:(part + 1) * size] + "<<<END>>>")
