import logging
import os
import zipfile
from functools import cache
from typing import List
from urllib.request import urlopen
from urllib.error import URLError

from ase.atoms import Atoms

QM9_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
)


def download_url(url: str, root: str) -> str:
    """Download if file does not exist in root already. Returns path to file."""
    filename = url.rpartition("/")[2]
    file_path = os.path.join(root, filename)

    try:
        from tqdm import tqdm

        progress = True
    except ImportError:
        progress = False

    try:
        if os.path.exists(file_path):
            logging.info(f"Using downloaded file: {file_path}")
            return file_path
        data = urlopen(url)
    except URLError:
        # No internet connection
        if os.path.exists(file_path):
            logging.info(f"No internet connection! Using downloaded file: {file_path}")
            return file_path

        raise

    chunk_size = 1024
    total_size = int(data.info()["Content-Length"].strip())

    if os.path.exists(file_path):
        if os.path.getsize(file_path) == total_size:
            logging.info(f"Using downloaded and verified file: {file_path}")
            return file_path

    logging.info(f"Downloading {url} to {file_path}")

    with open(file_path, "wb") as f:
        if progress:
            with tqdm(total=total_size) as pbar:
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(chunk_size)
        else:
            while True:
                chunk = data.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

    return file_path


def extract_zip(path: str, root: str):
    """Extract zip if content does not exist in root already."""
    logging.info(f"Extracting {path} to {root}...")
    with zipfile.ZipFile(path, "r") as f:
        for name in f.namelist():
            if name.endswith("/"):
                logging.info(f"Skip directory {name}")
                continue
            out_path = os.path.join(root, name)
            file_size = f.getinfo(name).file_size
            if os.path.exists(out_path) and os.path.getsize(out_path) == file_size:
                logging.info(f"Skip existing file {name}")
                continue
            logging.info(f"Extracting {name} to {root}...")
            f.extract(name, root)


def read_sdf(f):
    while True:
        name = f.readline()
        if not name:
            break

        f.readline()
        f.readline()

        L1 = f.readline().split()
        try:
            natoms = int(L1[0])
        except IndexError:
            print(L1)
            break

        positions = []
        symbols = []
        for _ in range(natoms):
            line = f.readline()
            x, y, z, symbol = line.split()[:4]
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])

        yield Atoms(symbols=symbols, positions=positions)

        while True:
            line = f.readline()
            if line.startswith("$$$$"):
                break


@cache
def load_qm9(root_dir: str) -> List[Atoms]:
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    path = download_url(QM9_URL, root_dir)
    extract_zip(path, root_dir)

    with open(os.path.join(root_dir, "gdb9.sdf")) as f:
        return list(read_sdf(f))
