import argparse
import re
import sys
import tomllib
from pathlib import Path


class ValidationError(Exception):
    pass


def check(github_ref: str | None) -> None:
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject.exists():
        raise ValidationError("pyproject.toml not found")
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    pyproject_version = data["project"]["version"]

    if github_ref is not None and github_ref.startswith("refs/tags/"):
        version = github_ref.removeprefix("refs/tags/")
        if version != pyproject_version:
            raise ValidationError(
                f"Version mismatch: GitHub ref is {version}, "
                f"but pyproject.toml is {pyproject_version}"
            )

    requires_python = data["project"]["requires-python"]
    assert sys.version_info[0] == 3, "Rewrite this script when Python 4 comes out"
    match = re.fullmatch(r">=3\.(\d+)", requires_python)
    if not match:
        raise ValidationError(f"Invalid requires-python: {requires_python!r}")
    lowest_minor = int(match.group(1))

    description = data["project"]["description"]
    if not description.endswith(f"3.{lowest_minor}+"):
        raise ValidationError(f"Description should mention Python 3.{lowest_minor}+")

    classifiers = set(data["project"]["classifiers"])
    for should_be_supported in range(lowest_minor, sys.version_info[1] + 1):
        if (
            f"Programming Language :: Python :: 3.{should_be_supported}"
            not in classifiers
        ):
            raise ValidationError(
                f"Missing classifier for Python 3.{should_be_supported}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to check the package metadata")
    parser.add_argument(
        "github_ref", type=str, help="The current GitHub ref", nargs="?"
    )
    args = parser.parse_args()
    try:
        check(args.github_ref)
    except ValidationError as e:
        print(e)
        sys.exit(1)
