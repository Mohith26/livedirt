"""COLMAP SfM over ALL extracted frames (train + held-out test registered together) with pycolmap.
Usage: python sfm.py <work_dir>   (expects <work_dir>/frames/*.jpg)"""
import sys, time, json, pathlib, pycolmap
work = pathlib.Path(sys.argv[1]); frames = work / "frames"; db = work / "database.db"; out = work / "sparse"
out.mkdir(exist_ok=True)
if db.exists(): db.unlink()
t0 = time.time()
ext = pycolmap.FeatureExtractionOptions(); ext.max_image_size = 1024; ext.sift.max_num_features = 4096
pycolmap.extract_features(db, frames, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="SIMPLE_RADIAL", extraction_options=ext, device=pycolmap.Device.cpu)
t1 = time.time(); print(f"features {t1-t0:.0f}s", flush=True)
pair = pycolmap.SequentialPairingOptions(); pair.overlap = 10; pair.loop_detection = False
pycolmap.match_sequential(db, pairing_options=pair, device=pycolmap.Device.cpu)
t2 = time.time(); print(f"matching {t2-t1:.0f}s", flush=True)
maps = pycolmap.incremental_mapping(db, frames, out)
t3 = time.time(); print(f"mapping {t3-t2:.0f}s", flush=True)
best = max(maps.values(), key=lambda r: r.num_reg_images()) if maps else None
res = {"features_s": round(t1-t0), "matching_s": round(t2-t1), "mapping_s": round(t3-t2), "models": len(maps),
       "registered": best.num_reg_images() if best else 0, "frames": len(list(frames.glob("*.jpg"))),
       "points3D": best.num_points3D() if best else 0, "mean_reproj_error_px": best.compute_mean_reprojection_error() if best else None}
if best:
    (out / "best").mkdir(exist_ok=True); best.write(out / "best")
    print(best.summary())
json.dump(res, open(work / "sfm_result.json", "w"), indent=1); print(res)
