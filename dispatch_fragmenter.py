import argparse
import os
import subprocess
import time

from qm9 import load_qm9


def main(chunk: int = 3000, num_seeds: int = 8):
    qm9 = load_qm9("qm9_data")
    processes = []

    for seed in range(num_seeds):
        for start in range(0, len(qm9), chunk):
            end = start + chunk
            path = f"data/fragments_{seed}_{start}_{end}.pkl"

            if os.path.exists(path):
                print(f"Skip {path}")
                continue

            print(f"Dispatching {path}")

            # run non-blocking
            p = subprocess.Popen(
                [
                    "srun",
                    "--mem=4G",
                    "--ntasks=1",
                    "--cpus-per-task=8",
                    # "--gres=gpu:1",
                    "python",
                    "fragmenter.py",
                    "--seed",
                    str(seed),
                    "--start",
                    str(start),
                    "--end",
                    str(end),
                    "--output",
                    path,
                ]
            )
            processes.append(p)

            # wait a bit to avoid overloading the scheduler
            time.sleep(3.0)

    print("Waiting for processes to finish...")
    for p in processes:
        p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=3000)
    parser.add_argument("--num_seeds", type=int, default=8)
    args = parser.parse_args()
    main(**vars(args))
