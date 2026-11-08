> The `volume delta` column is INVALID (the bench summed masked cells; fixed 2026-11-08).

| run | model | on grade % | MAE cm | over-dig m^3 | unplanned cut m^3 | volume delta m^3 | clipped | floored | peak drag kN | network share | verified |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline` | heightfield (cut+relax) | 34.5 | 5.34 | 0.331 | 0.077 | +nan | 0 | 0 | - | - | False |
| `neural-it0-v1` | soil-v1.onnx | 0.5 | 108.35 | 0.025 | 0.067 | +7.58 | 385022 | 1001850 | 39.9 | 100% | False |
| `neural-it0-v2` | soil-v2.onnx | 0.0 | 319.83 | 0.001 | 0.005 | +26.64 | 1916447 | 6048480 | 28.7 | 100% | False |
| `neural-it1-d-cgate` | soil-it1-d.onnx | 4.9 | 38.71 | 0.188 | 0.556 | -1.45 | 58600 | 1488679 | 166.2 | 98% | False |
| `neural-it1-d` | soil-it1-d.onnx | 3.9 | 59.15 | 0.156 | 0.386 | +0.54 | 108466 | 2368417 | 180 | 100% | False |
| `neural-it1-ens-gated-cgate` | soil-it1-d.onnx,soil-it1-d-s1.onnx,soil-it1-d-s2.onnx | 4.9 | 40.43 | 0.215 | 0.568 | -1.48 | 81432 | 1964717 | 150.5 | 99% | False |
| `neural-it1-ens-gated` | soil-it1-d.onnx,soil-it1-d-s1.onnx,soil-it1-d-s2.onnx | 3.9 | 56.02 | 0.208 | 0.443 | +0.11 | 109571 | 2417288 | 165.9 | 100% | False |
| `neural-it1-ens` | soil-it1-d.onnx,soil-it1-d-s1.onnx,soil-it1-d-s2.onnx | 3.9 | 56.02 | 0.208 | 0.443 | +0.11 | 109571 | 2417288 | 165.9 | 100% | False |
| `neural-it1-flux-acct` | soil-it1-flux.onnx | 2.0 | 78.19 | 0.008 | 0.039 | +6.11 | 90125 | 1480175 | 125.1 | 100% | False |
| `neural-it1-flux` | soil-it1-flux.onnx | 2.0 | 78.19 | 0.008 | 0.039 | +6.11 | 90125 | 1480175 | 125.1 | 100% | False |
| `neural-it1-full-norelief` | soil-it1-full.onnx | 3.0 | 65.08 | 0.015 | 0.118 | +4.22 | 123816 | 1403538 | 399.8 | 100% | False |
| `neural-it1-full` | soil-it1-full.onnx | 2.4 | 66.64 | 0.010 | 0.076 | +4.59 | 105467 | 1435950 | 185.2 | 100% | False |
| `neural-it1-sym` | soil-it1-sym.onnx | 4.4 | 50.45 | 0.156 | 0.465 | -0.33 | 56726 | 1622979 | 156.8 | 100% | False |
| `neural-nofloor-it1-d-cgate` | soil-it1-d.onnx | 3.8 | 44.95 | 2.392 | 6.938 | -11.59 | 4966 | 0 | 88.2 | 98% | False |
| `neural-nofloor-it1-d-nofloor` | soil-it1-d.onnx | 4.7 | 41.72 | 2.220 | 6.802 | -11.77 | 35203 | 0 | 194.2 | 100% | False |
| `neural-nofloor-it1-flux-nofloor` | soil-it1-flux.onnx | 3.0 | 48.70 | 1.782 | 10.877 | -14.54 | 67493 | 0 | 194.5 | 100% | False |
| `neural-it2-fv-floor-on` | soil-it2-fv.onnx | 0.0 | 117.38 | 0.000 | 0.000 | +12.53 | 118326 | 1226675 | 69.2 | 100% | False |
| `neural-nofloor-it2-full-mu0.7` | soil-it2-full.onnx | 3.6 | 44.57 | 1.804 | 8.027 | -9.42 | 59898 | 0 | 193.7 | 100% | False |
| `neural-nofloor-it2-full` | soil-it2-full.onnx | 2.5 | 44.70 | 1.750 | 8.511 | -10.28 | 71136 | 0 | 180.1 | 100% | False |
| `neural-nofloor-it2-fv` | soil-it2-fv.onnx | 2.2 | 44.54 | 1.609 | 8.209 | -10.47 | 80262 | 0 | 821.6 | 100% | False |
| `neural-nofloor-it2-sym-mu0.7` | soil-it2-sym.onnx | 7.2 | 38.50 | 0.560 | 2.673 | -6.75 | 37802 | 0 | 361.7 | 100% | False |
| `neural-nofloor-it2-sym` | soil-it2-sym.onnx | 3.6 | 40.86 | 0.682 | 3.377 | -5.35 | 55505 | 0 | 304.8 | 100% | False |
