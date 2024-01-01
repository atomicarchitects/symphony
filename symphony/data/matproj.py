import json
import os
from mp_api.client import MPRester

def get_materials(query, save=False, root_dir=None):
    with MPRester("NA4RS6zGonPp3S3TQTSBPkzevjE3jAPt") as mpr:
        materials = mpr.materials.summary.search(**query)
        if save:
            with open(os.path.join(root_dir, "materials.json"), "w") as f:
                json.dump(materials, f)
        return materials