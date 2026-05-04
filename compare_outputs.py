import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", required=True)
    parser.add_argument("--cpp", required=True)
    args = parser.parse_args()

    ref = pd.read_csv(args.ref)
    cpp = pd.read_csv(args.cpp)

    # assume same ordering and length
    ref_vals = ref.iloc[:, 1:].values
    cpp_vals = cpp.iloc[:, 1:].values

    diff = ref_vals - cpp_vals

    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    print(f"Max abs error:  {max_abs:.10f}")
    print(f"Mean abs error: {mean_abs:.10f}")
    print(f"RMSE:           {rmse:.10f}")

    if max_abs < 1e-3:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()