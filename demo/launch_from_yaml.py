#!/usr/bin/env python
"""
Read infer_jobs.yaml and run the patched batch-inference script once
per checkpoint, automatically passing --time_sampling extracted
from the TI<number> tag in the filename.
"""
import subprocess, pathlib, re, yaml, sys

cfg = yaml.safe_load(open("demo/infer_jobs.yaml"))
script    = pathlib.Path(cfg["script"]).resolve()
mix       = cfg["mix"]
out_root  = pathlib.Path(cfg["out_root"]).resolve()
out_root.mkdir(exist_ok=True)

pat_ti = re.compile(r"_TI(\d+)_")          # captures the number after TI

for model_path in cfg["models"]:
    model = pathlib.Path(model_path).resolve()
    m     = pat_ti.search(model.stem)
    if not m:
        sys.exit(f"✗  Cannot parse TI<number> in {model.name}")
    t_eval = int(m.group(1))               # 32, 64, …

    outdir = out_root / (model.stem + "_run")
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = ["python", str(script),
           "--model", str(model),
           "--mix",   mix,
           "--time_sampling", str(t_eval),
           "--outdir", str(outdir)]
    #print("▶", " ".join(cmd))
    print(model_path)
    subprocess.run(cmd, check=True)
