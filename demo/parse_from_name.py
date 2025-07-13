# parse_from_name.py ------------------------------------------------
import re, pathlib

PATTERN = re.compile(
    r"""
    (?P<ti>TI(?P<TI>\d+))      [_]
    (?:LE[0-9eE.+-]+)          [_]   # we ignore LR at inference
    (?P<fi>FI(?P<FI>[0-9.]+))  [_]
    (?P<st>ST(?P<ST>[0-9.]+))  [_]
    (?:CH(?P<CH>\d+)[_])?            # optional
    LO(?P<loss>[A-Za-z0-9_]+)
    """, re.VERBOSE)

def static_from_name(path: str | pathlib.Path):
    m = PATTERN.search(pathlib.Path(path).stem)
    if not m:
        raise ValueError(f"Cannot parse static params from “{path}”")
    groups = m.groupdict()
    return dict(
        time_sampling = int(groups["TI"]),
        fire_rate     = float(groups["FI"]),
        state_reg     = float(groups["ST"]),
        channels      = int(groups["CH"] or 16),   # default 16
        loss_tag      = groups["loss"],
    )
