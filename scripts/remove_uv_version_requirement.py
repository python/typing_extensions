#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["tomlkit>=0.13.2"]
# ///

"""Remove uv version requirements from a third-party checkout.

CI pins uv, which may not satisfy a checkout's uv version requirement.
Preserve the checkout's other uv settings.
"""

import argparse
from pathlib import Path

import tomlkit

parser = argparse.ArgumentParser()
parser.add_argument("repo_dir", type=Path)
repo_dir = parser.parse_args().repo_dir
if not repo_dir.is_dir():
    parser.error(f"{repo_dir} is not a directory")

for filename in ("pyproject.toml", "uv.toml"):
    path = repo_dir / filename
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        continue

    config = tomlkit.parse(contents)
    uv_config = (
        config.get("tool", {}).get("uv", {}) if filename == "pyproject.toml" else config
    )
    if "required-version" in uv_config:
        del uv_config["required-version"]
        path.write_text(tomlkit.dumps(config), encoding="utf-8")
