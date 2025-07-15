# sweep_runner.py ----------------------------------------------------
import itertools, subprocess, pathlib, yaml, datetime as dt

cfg = yaml.safe_load(pathlib.Path("demo/sweep_config.yaml").read_text())

# build Cartesian product of all hyper-parameter values
keys, values = zip(*cfg["grid"].items())
for combo in itertools.product(*values):
    kwargs = dict(zip(keys, combo))

    # unique model name e.g. 2025-07-11T15-00_TS32_CH16_Lspectral
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M")
    tag   = "_".join(f"{k[:2]}{v}" for k, v in kwargs.items())
    model_dir = pathlib.Path(cfg["output_root"], f"{stamp}_{tag}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(str(model_dir))

    # assemble CLI arguments expected by your training script
    args = [
        "python", cfg["script"],
        "--time_sampling", str(kwargs["TIME_SAMPLING"]),
        "--pde", kwargs["PDE"],
        "--learn_rate",    str(kwargs["LEARN_RATE"]),
        "--loss",          kwargs["LOSS"],
        "--model_filename", str(model_dir),
        "--fire_rate", str(kwargs["FIRE_RATE"]),
        "--state_reg", str(kwargs["STATE_REGULARISER"])
    ]
    print("Launching:", " ".join(args))
    subprocess.run(args, check=True)
