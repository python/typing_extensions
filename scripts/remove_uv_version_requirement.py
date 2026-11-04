"""Remove uv's required-version from a third-party checkout.

CI pins uv, which may not satisfy a checkout's uv version requirement.
Preserve the checkout's other uv settings in pyproject.toml and uv.toml.
Fail if a requirement cannot be removed safely as a single-line assignment.

Using tomlkit would make this simpler and support more TOML layouts, but we
would need to declare the dependency in a PEP 723 script. We would then
want a uv-pre-commit hook to check in CI that the script's lockfile is up to
date, and Renovate configuration to update the hook alongside CI's uv pin
so the hook uses the same uv version as CI. We therefore keep this script
stdlib-only for now, to avoid that spiralling complexity. This can be
reconsidered if we ever need a more general solution in the future.
"""

import argparse
import tomllib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("repo_dir", type=Path)
repo_dir = parser.parse_args().repo_dir
if not repo_dir.is_dir():
    parser.error(f"{repo_dir} is not a directory")

# Collect each updated file's path and contents so neither file is changed if
# a requirement in either file cannot be removed safely.
config_updates: dict[Path, str] = {}

for config_filename in ("pyproject.toml", "uv.toml"):
    config_path = repo_dir / config_filename
    try:
        raw_config = config_path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        continue

    parsed_config = tomllib.loads(raw_config)
    uv_config = (
        parsed_config.get("tool", {}).get("uv", {})
        if config_filename == "pyproject.toml"
        else parsed_config
    )
    if "required-version" in uv_config:
        del uv_config["required-version"]
        raw_config_lines = raw_config.splitlines(keepends=True)
        for index, config_line in enumerate(raw_config_lines):
            if "=" not in config_line:
                continue
            try:
                key = tomllib.loads(f"{config_line.partition('=')[0]} = true")
                if key != {"required-version": True}:
                    continue
                updated_config = "".join(
                    raw_config_lines[:index] + raw_config_lines[index + 1 :]
                )
                if tomllib.loads(updated_config) == parsed_config:
                    config_updates[config_path] = updated_config
                    break
            except tomllib.TOMLDecodeError:
                continue
        else:
            raise SystemExit(
                f"cannot safely remove required-version from {config_path}; "
                "expected a single-line required-version assignment"
            )

for config_path, updated_raw_config in config_updates.items():
    config_path.write_bytes(updated_raw_config.encode("utf-8"))
