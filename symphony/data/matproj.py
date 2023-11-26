import json
import os
from mp_api.client import MPRester

def get_materials(root_dir, save, **query):
    with MPRester("NA4RS6zGonPp3S3TQTSBPkzevjE3jAPt") as mpr:
        print(query)
        materials = mpr.materials.summary.search(**query)
        if save:
            with open(os.path.join(root_dir, "materials.json"), "w") as f:
                json.dump(materials, f)
        return materials