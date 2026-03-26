import argparse
from pathlib import Path


def print_section(title: str):
    print(f"\n== {title} ==")


def resolve_dump_dir(path: Path, latest: bool) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"dump directory not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"not a directory: {path}")
    if not latest:
        return path

    children = [p for p in path.iterdir() if p.is_dir()]
    if not children:
        raise SystemExit(f"no dumped kernel directories found under: {path}")
    return max(children, key=lambda p: p.stat().st_mtime)


def list_files(root: Path):
    files = sorted(p for p in root.iterdir() if p.is_file())
    if not files:
        print("(none)")
        return
    for path in files:
        print(path.name)


def list_tree(root: Path, prefix: str = ""):
    entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    if not entries:
        print("(empty)")
        return
    for entry in entries:
        label = f"{prefix}{entry.name}"
        if entry.is_dir():
            print(f"{label}/")
            list_tree(entry, prefix=label + "/")
        else:
            print(label)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Triton dump directory and summarize IR artifacts."
    )
    parser.add_argument(
        "dump_dir",
        help="Path to one dumped kernel directory, or the root dump directory when using --latest",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Treat dump_dir as the root dump directory and inspect the newest dumped kernel subdirectory",
    )
    args = parser.parse_args()

    dump_dir = resolve_dump_dir(Path(args.dump_dir), args.latest)

    print(f"dump_dir: {dump_dir}")

    print_section("Top-Level Files")
    list_files(dump_dir)

    npuir_dir = dump_dir / "npuir_passes"
    print_section("AscendNPU-IR Pass Tree")
    if npuir_dir.exists():
        list_tree(npuir_dir)
    else:
        print("(missing)")


if __name__ == "__main__":
    main()
