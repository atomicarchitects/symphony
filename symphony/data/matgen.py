import json
from mp_api.client import MPRester

def get_materials(root_dir, save, **query):
    with MPRester("NA4RS6zGonPp3S3TQTSBPkzevjE3jAPt") as mpr:
        materials = mpr.summary.search(**query)
        if save:
            with open("materials.json", "w") as f:
                json.dump(materials, f)
        return materials