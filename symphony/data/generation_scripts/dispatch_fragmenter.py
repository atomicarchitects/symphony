import argparse
import os
import time

import numpy as np

from symphony.data import qm9
from symphony.data.generation_scripts import fragmenter


def main(
    chunk: int,
    start_seed: int,
    end_seed: int,
    root_dir: str,
    mode: str,
    heavy_first: bool,
    beta_com: float,
):
    qm9_data = qm9.load_qm9("qm9_data")
    starts = list(range(0, len(qm9_data), chunk))
    del qm9_data

    execution_time = []
    processes = []

    for seed in range(start_seed, end_seed):
        for start in starts:
            end = start + chunk
            path = f"{root_dir}/fragments_{seed:02d}_{start:06d}_{end:06d}"

            if os.path.exists(path):
                print(f"Skip {path}")
                continue

            print(f"Dispatching {path}")
            t0 = time.time()

            fragmenter.main(
                seed=seed,
                start=start,
                end=end,
                output=path,
                mode=mode,
                heavy_first=heavy_first,
                beta_com=beta_com,
            )

            t1 = time.time()
            execution_time.append(t1 - t0)
            print(f"Execution time: {t1 - t0:.0f} seconds")
            done = len(execution_time)
            todo = len(starts) * (end_seed - start_seed) - done
            print(
                f"Estimated time remaining: {todo * np.mean(execution_time):.0f} seconds"
            )

    print("Waiting for processes to finish...")
    for p in processes:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=2976)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--end_seed", type=int, default=8)
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default="nn")
    parser.add_argument("--heavy_first", action="store_true")
    parser.add_argument("--beta_com", type=float, default=0.0)
    args = parser.parse_args()
    main(**vars(args))
