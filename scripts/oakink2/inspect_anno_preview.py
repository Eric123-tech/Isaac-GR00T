import argparse
import pickle
from pathlib import Path


def print_obj(obj, name="root", depth=0, max_depth=3):
    indent = "  " * depth
    print(f"{indent}{name}: type={type(obj)}")

    if depth >= max_depth:
        return

    if isinstance(obj, dict):
        print(f"{indent}  num_keys={len(obj)}")
        for i, (k, v) in enumerate(obj.items()):
            if i >= 30:
                print(f"{indent}  ...")
                break
            print_obj(v, name=f"[{repr(k)}]", depth=depth + 1, max_depth=max_depth)

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}  len={len(obj)}")
        for i, v in enumerate(obj[:5]):
            print_obj(v, name=f"[{i}]", depth=depth + 1, max_depth=max_depth)

    else:
        if hasattr(obj, "shape"):
            print(f"{indent}  shape={obj.shape}")
        else:
            s = str(obj)
            print(f"{indent}  value={s[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    args = parser.parse_args()

    pkl_path = Path(args.pkl).expanduser()
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)

    print_obj(obj, max_depth=4)


if __name__ == "__main__":
    main()

