import argparse
import os

# import subprocess
import sys
import time

import numpy as np

import fragmenter

sys.path.append("..")

import qm9  # noqa: E402


def main(
    chunk: int,
    start_seed: int,
    end_seed: int,
    root_dir: str,
    mode: str,
):
    if os.path.exists(root_dir):
        print("Root directory already exists.")
    else:
        os.makedirs(root_dir)
        print(f"Created root directory {root_dir}")

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

            # run non-blocking
            # p = subprocess.Popen(
            #     [
            #         # "srun",
            #         # "--mem=4G",
            #         # "--ntasks=1",
            #         # "--cpus-per-task=8",
            #         # "--gres=gpu:1",
            #         "python",
            #         "fragmenter.py",
            #         "--seed",
            #         str(seed),
            #         "--start",
            #         str(start),
            #         "--end",
            #         str(end),
            #         "--output",
            #         path,
            #         "--mode",
            #         mode,
            #     ]
            # )
            # processes.append(p)

            # wait a bit to avoid overloading the scheduler
            # time.sleep(10.0)

            # actually wait for the process to finish
            # p.wait()

            fragmenter.main(
                seed=seed,
                start=start,
                end=end,
                output=path,
                mode=mode,
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
    args = parser.parse_args()
    main(**vars(args))
