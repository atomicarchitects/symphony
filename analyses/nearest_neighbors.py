"""
CrystalNN and VoronoiNN for pymatgen Molecules.
Original code for Structures: https://github.com/materialsproject/pymatgen/blob/master/pymatgen/analysis/local_env.py
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple

import numpy as np
from scipy.spatial import Voronoi

from pymatgen.analysis.local_env import (
    NearNeighbors,
    _get_radius,
    _get_default_radius,
    _is_in_targets,
    solid_angle,
    vol_tetra,
)
from pymatgen.core import Molecule, Structure


class CrystalNN(NearNeighbors):
    """
    This is a custom near-neighbor method intended for use in all kinds of periodic structures
    (metals, minerals, porous structures, etc). It is based on a Voronoi algorithm and uses the
    solid angle weights to determine the probability of various coordination environments. The
    algorithm can also modify probability using smooth distance cutoffs as well as Pauling
    electronegativity differences. The output can either be the most probable coordination
    environment or a weighted list of coordination environments.
    """

    NNData = namedtuple("NNData", ["all_nninfo", "cn_weights", "cn_nninfo"])

    def __init__(
        self,
        weighted_cn=False,
        cation_anion=False,
        distance_cutoffs=(0.5, 1),
        x_diff_weight=3.0,
        porous_adjustment=True,
        search_cutoff=7,
        fingerprint_length=None,
    ):
        """
        Initialize CrystalNN with desired parameters. Default parameters assume
        "chemical bond" type behavior is desired. For geometric neighbor
        finding (e.g., structural framework), set (i) distance_cutoffs=None,
        (ii) x_diff_weight=0 and (optionally) (iii) porous_adjustment=False
        which will disregard the atomic identities and perform best for a purely
        geometric match.

        Args:
            weighted_cn: (bool) if set to True, will return fractional weights
                for each potential near neighbor.
            cation_anion: (bool) if set True, will restrict bonding targets to
                sites with opposite or zero charge. Requires an oxidation states
                on all sites in the structure.
            distance_cutoffs: ([float, float]) - if not None, penalizes neighbor
                distances greater than sum of covalent radii plus
                distance_cutoffs[0]. Distances greater than covalent radii sum
                plus distance_cutoffs[1] are enforced to have zero weight.
            x_diff_weight: (float) - if multiple types of neighbor elements are
                possible, this sets preferences for targets with higher
                electronegativity difference.
            porous_adjustment: (bool) - if True, readjusts Voronoi weights to
                better describe layered / porous structures
            search_cutoff: (float) cutoff in Angstroms for initial neighbor
                search; this will be adjusted if needed internally
            fingerprint_length: (int) if a fixed_length CN "fingerprint" is
                desired from get_nn_data(), set this parameter
        """
        self.weighted_cn = weighted_cn
        self.cation_anion = cation_anion
        self.distance_cutoffs = distance_cutoffs
        self.x_diff_weight = x_diff_weight if x_diff_weight is not None else 0
        self.search_cutoff = search_cutoff
        self.porous_adjustment = porous_adjustment
        self.fingerprint_length = fingerprint_length

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return False

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return True

    def get_nn_info(self, molecule: Molecule, n: int) -> list[dict]:
        """
        Get all near-neighbor information.

        Args:
            molecule: (Molecule) pymatgen Molecule
            n: (int) index of target site

        Returns:
            siw (list[dict]): each dictionary provides information
                about a single near neighbor, where key 'site' gives access to the
                corresponding Site object, 'image' gives the image location, and
                'weight' provides the weight that a given near-neighbor site contributes
                to the coordination number (1 or smaller), 'site_index' gives index of
                the corresponding site in the original molecule.
        """
        nn_data = self.get_nn_data(molecule, n)

        if not self.weighted_cn:
            max_key = max(nn_data.cn_weights, key=lambda k: nn_data.cn_weights[k])
            nn = nn_data.cn_nninfo[max_key]
            for entry in nn:
                entry["weight"] = 1
            return nn

        for entry in nn_data.all_nninfo:
            weight = 0
            for cn in nn_data.cn_nninfo:
                for cn_entry in nn_data.cn_nninfo[cn]:
                    if entry["site"] == cn_entry["site"]:
                        weight += nn_data.cn_weights[cn]

            entry["weight"] = weight

        return nn_data.all_nninfo

    def get_nn_data(self, molecule: Molecule, n: int, length=None):
        """
        The main logic of the method to compute near neighbor.

        Args:
            molecule: (Molecule) enclosing molecule object
            n: (int) index of target site to get NN info for
            length: (int) if set, will return a fixed range of CN numbers

        Returns:
            a namedtuple (NNData) object that contains:
            - all near neighbor sites with weights
            - a dict of CN -> weight
            - a dict of CN -> associated near neighbor sites
        """
        length = length or self.fingerprint_length

        # determine possible bond targets
        target = None
        if self.cation_anion:
            target = []
            m_oxi = molecule[n].specie.oxi_state
            for site in molecule:
                oxi_state = getattr(site.specie, "oxi_state", None)
                if oxi_state is not None and oxi_state * m_oxi <= 0:  # opposite charge
                    target.append(site.specie)
            if not target:
                raise ValueError(
                    "No valid targets for site within cation_anion constraint!"
                )

        # get base VoronoiNN targets
        cutoff = self.search_cutoff
        vnn = VoronoiNN(weight="solid_angle", targets=target, cutoff=cutoff, allow_pathological=True)
        nn = vnn.get_nn_info(molecule, n)

        # solid angle weights can be misleading in open / porous structures
        # adjust weights to correct for this behavior
        if self.porous_adjustment:
            for x in nn:
                x["weight"] *= x["poly_info"]["solid_angle"] / x["poly_info"]["area"]

        # adjust solid angle weight based on electronegativity difference
        if self.x_diff_weight > 0:
            for entry in nn:
                X1 = molecule[n].specie.X
                X2 = entry["site"].specie.X

                if math.isnan(X1) or math.isnan(X2):
                    chemical_weight = 1
                else:
                    # note: 3.3 is max deltaX between 2 elements
                    chemical_weight = 1 + self.x_diff_weight * math.sqrt(
                        abs(X1 - X2) / 3.3
                    )

                entry["weight"] = entry["weight"] * chemical_weight

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x["weight"], reverse=True)
        if nn[0]["weight"] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        # renormalize weights so the highest weight is 1.0
        highest_weight = nn[0]["weight"]
        for entry in nn:
            entry["weight"] = entry["weight"] / highest_weight

        # adjust solid angle weights based on distance
        if self.distance_cutoffs:
            r1 = _get_radius(molecule[n])
            for entry in nn:
                r2 = _get_radius(entry["site"])
                if r1 > 0 and r2 > 0:
                    diameter = r1 + r2
                else:
                    warnings.warn(
                        f"CrystalNN: cannot locate an appropriate radius for {entry['site']}, "
                        "covalent or atomic radii will be used, this can lead "
                        "to non-optimal results."
                    )
                    diameter = _get_default_radius(molecule[n]) + _get_default_radius(
                        entry["site"]
                    )

                dist = np.linalg.norm(molecule[n].coords - entry["site"].coords)
                dist_weight: float = 0

                cutoff_low = diameter + self.distance_cutoffs[0]
                cutoff_high = diameter + self.distance_cutoffs[1]

                if dist <= cutoff_low:
                    dist_weight = 1
                elif dist < cutoff_high:
                    dist_weight = (
                        math.cos(
                            (dist - cutoff_low) / (cutoff_high - cutoff_low) * math.pi
                        )
                        + 1
                    ) * 0.5
                entry["weight"] = entry["weight"] * dist_weight

        # sort nearest neighbors from highest to lowest weight
        nn = sorted(nn, key=lambda x: x["weight"], reverse=True)
        print([(n['site'].specie, n['site_index'], n['weight']) for n in nn])
        if nn[0]["weight"] == 0:
            return self.transform_to_length(self.NNData([], {0: 1.0}, {0: []}), length)

        for entry in nn:
            entry["weight"] = round(entry["weight"], 3)
            del entry["poly_info"]  # trim

        # remove entries with no weight
        nn = [x for x in nn if x["weight"] > 0]

        # get the transition distances, i.e. all distinct weights
        dist_bins: list[float] = []
        for entry in nn:
            if not dist_bins or dist_bins[-1] != entry["weight"]:
                dist_bins.append(entry["weight"])
        dist_bins.append(0)

        # main algorithm to determine fingerprint from bond weights
        cn_weights = {}  # CN -> score for that CN
        cn_nninfo = {}  # CN -> list of nearneighbor info for that CN
        for idx, val in enumerate(dist_bins):
            if val != 0:
                nn_info = []
                for entry in nn:
                    if entry["weight"] >= val:
                        nn_info.append(entry)
                cn = len(nn_info)
                cn_nninfo[cn] = nn_info
                cn_weights[cn] = self._semicircle_integral(dist_bins, idx)

        # add zero coord
        cn0_weight = 1 - sum(cn_weights.values())
        if cn0_weight > 0:
            cn_nninfo[0] = []
            cn_weights[0] = cn0_weight

        return self.transform_to_length(self.NNData(nn, cn_weights, cn_nninfo), length)

    def get_cn(self, molecule: Molecule, n: int, **kwargs) -> float:  # type: ignore
        """
        Get coordination number, CN, of site with index n in molecule.

        Args:
            molecule (Molecule): input molecule.
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).
            on_disorder ('take_majority_strict' | 'take_majority_drop' | 'take_max_species' | 'error'):
                What to do when encountering a disordered molecule. 'error' will raise ValueError.
                'take_majority_strict' will use the majority specie on each site and raise
                ValueError if no majority exists. 'take_max_species' will use the first max specie
                on each site. For {{Fe: 0.4, O: 0.4, C: 0.2}}, 'error' and 'take_majority_strict'
                will raise ValueError, while 'take_majority_drop' ignores this site altogether and
                'take_max_species' will use Fe as the site specie.

        Returns:
            cn (float): coordination number.
        """
        use_weights = kwargs.get("use_weights", False)
        if self.weighted_cn != use_weights:
            raise ValueError(
                "The weighted_cn parameter and use_weights parameter should match!"
            )

        return super().get_cn(molecule, n, **kwargs)

    def get_cn_dict(
        self, molecule: Molecule, n: int, use_weights: bool = False, **kwargs
    ):
        """
        Get coordination number, CN, of each element bonded to site with index n in molecule.

        Args:
            molecule (Molecule): input molecule
            n (int): index of site for which to determine CN.
            use_weights (bool): flag indicating whether (True)
                to use weights for computing the coordination number
                or not (False, default: each coordinated site has equal
                weight).

        Returns:
            cn (dict): dictionary of CN of each element bonded to site
        """
        if self.weighted_cn != use_weights:
            raise ValueError(
                "The weighted_cn parameter and use_weights parameter should match!"
            )

        return super().get_cn_dict(molecule, n, use_weights)

    @staticmethod
    def _semicircle_integral(dist_bins, idx):
        """
        An internal method to get an integral between two bounds of a unit
        semicircle. Used in algorithm to determine bond probabilities.

        Args:
            dist_bins: (float) list of all possible bond weights
            idx: (float) index of starting bond weight

        Returns:
            (float) integral of portion of unit semicircle
        """
        r = 1

        x1 = dist_bins[idx]
        x2 = dist_bins[idx + 1]

        if dist_bins[idx] == 1:
            area1 = 0.25 * math.pi * r**2
        else:
            area1 = 0.5 * (
                (x1 * math.sqrt(r**2 - x1**2))
                + (r**2 * math.atan(x1 / math.sqrt(r**2 - x1**2)))
            )

        area2 = 0.5 * (
            (x2 * math.sqrt(r**2 - x2**2))
            + (r**2 * math.atan(x2 / math.sqrt(r**2 - x2**2)))
        )

        return (area1 - area2) / (0.25 * math.pi * r**2)

    @staticmethod
    def transform_to_length(nn_data, length):
        """
        Given NNData, transforms data to the specified fingerprint length

        Args:
            nn_data: (NNData)
            length: (int) desired length of NNData.
        """
        if length is None:
            return nn_data

        if length:
            for cn in range(length):
                if cn not in nn_data.cn_weights:
                    nn_data.cn_weights[cn] = 0
                    nn_data.cn_nninfo[cn] = []

        return nn_data


class VoronoiNN(NearNeighbors):
    """
    Uses a Voronoi algorithm to determine near neighbors for each site in a
    structure.
    """

    def __init__(
        self,
        tol=0,
        targets=None,
        cutoff=13.0,
        allow_pathological=False,
        weight="solid_angle",
        extra_nn_info=True,
        compute_adj_neighbors=True,
    ):
        """
        Args:
            tol (float): tolerance parameter for near-neighbor finding. Faces that are
                smaller than `tol` fraction of the largest face are not included in the
                tessellation. (default: 0).
            targets (Element or list of Elements): target element(s).
            cutoff (float): cutoff radius in Angstrom to look for near-neighbor
                atoms. Defaults to 13.0.
            allow_pathological (bool): whether to allow infinite vertices in
                determination of Voronoi coordination.
            weight (string) - Statistic used to weigh neighbors (see the statistics
                available in get_voronoi_polyhedra)
            extra_nn_info (bool) - Add all polyhedron info to `get_nn_info`
            compute_adj_neighbors (bool) - Whether to compute which neighbors are
                adjacent. Turn off for faster performance.
        """
        super().__init__()
        self.tol = tol
        self.cutoff = cutoff
        self.allow_pathological = allow_pathological
        self.targets = targets
        self.weight = weight
        self.extra_nn_info = extra_nn_info
        self.compute_adj_neighbors = compute_adj_neighbors

    @property
    def structures_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Structure
        objects?
        """
        return False

    @property
    def molecules_allowed(self) -> bool:
        """
        Boolean property: can this NearNeighbors class be used with Molecule
        objects?
        """
        return True

    def get_voronoi_polyhedra(self, molecule: Molecule, n: int):
        """
        Gives a weighted polyhedra around a site.

        See ref: A Proposed Rigorous Definition of Coordination Number,
        M. O'Keeffe, Acta Cryst. (1979). A35, 772-775

        Args:
            molecule (Molecule): molecule for which to evaluate the
                coordination environment.
            n (int): site index.

        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """
        # Assemble the list of neighbors used in the tessellation. Gets all atoms within a certain radius
        targets = molecule.elements if self.targets is None else self.targets

        center = molecule[n]
        cutoff = self.cutoff

        # max cutoff is the longest distance between atoms, plus noise
        coords = molecule.cart_coords
        noise = 0.01
        dist_matrix = np.linalg.norm(np.vstack([coords] * len(molecule)) - np.repeat(
            coords, len(molecule), axis=0,
        ), axis=-1)
        max_cutoff = dist_matrix.max() + noise

        while True:
            try:
                neighbors = molecule.get_neighbors(center, cutoff)
                neighbors = [i[0] for i in sorted(neighbors, key=lambda s: s[1])]

                # Run the Voronoi tessellation
                qvoronoi_input = [s.coords for s in neighbors]
                print(neighbors)

                voro = Voronoi(
                    qvoronoi_input
                )  # can give seg fault if cutoff is too small

                # Extract data about the site in question
                cell_info = self._extract_cell_info(
                    0, neighbors, targets, voro, self.compute_adj_neighbors
                )
                break

            except RuntimeError as e:
                if cutoff >= max_cutoff:
                    if e.args and "vertex" in e.args[0]:
                        # pass through the error raised by _extract_cell_info
                        raise e
                    raise RuntimeError(
                        "Error in Voronoi neighbor finding; max cutoff exceeded"
                    )
                cutoff = min(cutoff * 2, max_cutoff + 0.001)
        return cell_info

    def get_all_voronoi_polyhedra(self, molecule: Molecule):
        """Get the Voronoi polyhedra for all site in a simulation cell.

        Args:
            molecule (Molecule): Molecule to be evaluated

        Returns:
            A dict of sites sharing a common Voronoi facet with the site
            n mapped to a directory containing statistics about the facet:
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
        """
        # Special case: For atoms with 1 site, the atom in the root image is not
        # included in the get_all_neighbors output. Rather than creating logic to add
        # that atom to the neighbor list, which requires detecting whether it will be
        # translated to reside within the unit cell before neighbor detection, it is
        # less complex to just call the one-by-one operation
        if len(molecule) == 1:
            return [self.get_voronoi_polyhedra(molecule, 0)]

        # Assemble the list of neighbors used in the tessellation
        targets = molecule.elements if self.targets is None else self.targets

        # Initialize the list of sites with the atoms in the origin unit cell
        # The `get_all_neighbors` function returns neighbors for each site's image in
        # the original unit cell. We start off with these central atoms to ensure they
        # are included in the tessellation

        # sites = [x.to_unit_cell() for x in molecule]
        sites = [x for x in molecule]
        indices = [(i, 0, 0, 0) for i, _ in enumerate(molecule)]

        # Get all neighbors within a certain cutoff. Record both the list of these neighbors and the site indices.
        for i in range(len(molecule)):
            neighs = molecule.get_neighbors(molecule[i], self.cutoff)
            sites.extend([x[0] for x in neighs])
            indices.extend([(x[2],) + tuple(x.coords.tolist()) for x in neighs])

        # Get the non-duplicates (using the site indices for numerical stability)
        indices = np.array(indices, dtype=int)  # type: ignore
        indices, uniq_inds = np.unique(indices, return_index=True, axis=0)  # type: ignore[assignment]
        sites = [sites[i] for i in uniq_inds]

        # Sort array such that atoms in the root image are first
        # Exploit the fact that the array is sorted by the unique operation such that
        # the images associated with atom 0 are first, followed by atom 1, etc.
        (root_images,) = np.nonzero(np.abs(indices[:, 1:]).max(axis=1) == 0)  # type: ignore

        del indices  # Save memory (tessellations can be costly)

        # Run the tessellation
        qvoronoi_input = [s.coords for s in sites]
        voro = Voronoi(qvoronoi_input)

        # Get the information for each neighbor
        return [
            self._extract_cell_info(i, sites, targets, voro, self.compute_adj_neighbors)
            for i in root_images.tolist()
        ]

    def _extract_cell_info(
        self, site_idx, sites, targets, voro, compute_adj_neighbors=False
    ):
        """Get the information about a certain atom from the results of a tessellation.

        Args:
            site_idx (int) - Index of the atom in question
            sites ([Site]) - List of all sites in the tessellation
            targets ([Element]) - Target elements
            voro - Output of qvoronoi
            compute_adj_neighbors (boolean) - Whether to compute which neighbors are adjacent

        Returns:
            A dict of sites sharing a common Voronoi facet. Key is facet id
             (not useful) and values are dictionaries containing statistics
             about the facet:
                - site: Pymatgen site
                - solid_angle - Solid angle subtended by face
                - angle_normalized - Solid angle normalized such that the
                    faces with the largest
                - area - Area of the facet
                - face_dist - Distance between site n and the facet
                - volume - Volume of Voronoi cell for this face
                - n_verts - Number of vertices on the facet
                - adj_neighbors - Facet id's for the adjacent neighbors
        """
        # Get the coordinates of every vertex
        all_vertices = voro.vertices

        # Get the coordinates of the central site
        center_coords = sites[site_idx].coords

        # Iterate through all the faces in the tessellation
        results = {}
        for nn, vind in voro.ridge_dict.items():
            # Get only those that include the site in question
            if site_idx in nn:
                other_site = nn[0] if nn[1] == site_idx else nn[1]
                if -1 in vind:
                    # -1 indices correspond to the Voronoi cell
                    #  missing a face
                    if self.allow_pathological:
                        continue

                    raise RuntimeError(
                        "This molecule is pathological, infinite vertex in the Voronoi construction"
                    )

                # Get the solid angle of the face
                facets = [all_vertices[i] for i in vind]
                angle = solid_angle(center_coords, facets)

                # Compute the volume of associated with this face
                volume = 0
                # qvoronoi returns vertices in CCW order, so I can break
                # the face up in to segments (0,1,2), (0,2,3), ... to compute
                # its area where each number is a vertex size
                for j, k in zip(vind[1:], vind[2:]):
                    volume += vol_tetra(
                        center_coords,
                        all_vertices[vind[0]],
                        all_vertices[j],
                        all_vertices[k],
                    )

                # Compute the distance of the site to the face
                face_dist = np.linalg.norm(center_coords - sites[other_site].coords) / 2

                # Compute the area of the face (knowing V=Ad/3)
                face_area = 3 * volume / face_dist

                # Compute the normal of the facet
                normal = np.subtract(sites[other_site].coords, center_coords)
                normal /= np.linalg.norm(normal)

                # Store by face index
                results[other_site] = {
                    "site": sites[other_site],
                    "normal": normal,
                    "solid_angle": angle,
                    "volume": volume,
                    "face_dist": face_dist,
                    "area": face_area,
                    "n_verts": len(vind),
                }

                # If we are computing which neighbors are adjacent, store the vertices
                if compute_adj_neighbors:
                    results[other_site]["verts"] = vind

        # all sites should have at least two connected ridges in periodic system
        if len(results) == 0:
            raise ValueError(
                "No Voronoi neighbors found for site - try increasing cutoff"
            )

        # Get only target elements
        result_weighted = {}
        for nn_index, nn_stats in results.items():
            # Check if this is a target site
            nn = nn_stats["site"]
            if nn.is_ordered:
                if nn.specie in targets:
                    result_weighted[nn_index] = nn_stats
            else:  # if nn site is disordered
                for disordered_sp in nn.species:
                    if disordered_sp in targets:
                        result_weighted[nn_index] = nn_stats

        # If desired, determine which neighbors are adjacent
        if compute_adj_neighbors:
            # Initialize storage for the adjacent neighbors
            adj_neighbors = {i: [] for i in result_weighted}

            # Find the neighbors that are adjacent by finding those
            #  that contain exactly two vertices
            for a_ind, a_nn_info in result_weighted.items():
                # Get the indices for this site
                a_verts = set(a_nn_info["verts"])

                # Loop over all neighbors that have an index lower that this one
                # The goal here is to exploit the fact that neighbor adjacency is
                # symmetric (if A is adj to B, B is adj to A)
                for b_ind, b_nninfo in result_weighted.items():
                    if b_ind > a_ind:
                        continue
                    if len(a_verts.intersection(b_nninfo["verts"])) == 2:
                        adj_neighbors[a_ind].append(b_ind)
                        adj_neighbors[b_ind].append(a_ind)

            # Store the results in the nn_info
            for key, neighbors in adj_neighbors.items():
                result_weighted[key]["adj_neighbors"] = neighbors

        return result_weighted

    def get_nn_info(self, molecule: Molecule, n: int):
        """
        Get all near-neighbor sites as well as the associated image locations
        and weights of the site with index n in molecule
        using Voronoi decomposition.

        Args:
            molecule (Molecule): input molecule.
            n (int): index of site for which to determine near-neighbor sites.

        Returns:
            siw (list of tuples (Site, array, float)): tuples, each one
                of which represents a coordinated site, its image location,
                and its weight.
        """
        # Run the tessellation
        nns = self.get_voronoi_polyhedra(molecule, n)

        # Extract the NN info
        return self._extract_nn_info(molecule, nns)

    def get_all_nn_info(self, molecule: Molecule):
        """
        Args:
            molecule (Molecule): input molecule.

        Returns:
            All nn info for all sites.
        """
        all_voro_cells = self.get_all_voronoi_polyhedra(molecule)
        return [self._extract_nn_info(molecule, cell) for cell in all_voro_cells]

    def _extract_nn_info(self, molecule: Molecule, nns):
        """Given Voronoi NNs, extract the NN info in the form needed by NearestNeighbors.

        Args:
            molecule (molecule): molecule being evaluated
            nns ([dicts]): Nearest neighbor information for a molecule

        Returns:
            list[tuple[PeriodicSite, np.ndarray, float]]: tuples of the form
                (site, image, weight). See nn_info.
        """
        # Get the target information
        targets = molecule.elements if self.targets is None else self.targets

        # Extract the NN info
        siw = []
        max_weight = max(nn[self.weight] for nn in nns.values())
        for nstats in nns.values():
            site = nstats["site"]
            if nstats[self.weight] > self.tol * max_weight and _is_in_targets(
                site, targets
            ):
                nn_info = {
                    "site": site,
                    "image": self._get_image(molecule, site),
                    "weight": nstats[self.weight] / max_weight,
                    "site_index": self._get_original_site(molecule, site),
                }

                if self.extra_nn_info:
                    # Add all the information about the site
                    poly_info = nstats
                    del poly_info["site"]
                    nn_info["poly_info"] = poly_info
                siw.append(nn_info)
        return siw
    
    @staticmethod
    def _get_image(molecule, site):
        """Private convenience method for get_nn_info,
        gives lattice image from provided Site and Molecule.

        Image is defined as displacement from original site in molecule to a given site.
        i.e. if molecule has a site at (-0.1, 1.0, 0.3), then (0.9, 0, 2.3) -> jimage = (1, -1, 2).
        Note that this method takes O(number of sites) due to searching an original site.

        Args:
            molecule: Molecule Object
            site: PeriodicSite Object

        Returns:
            image: ((int)*3) Lattice image
        """
        original_site = molecule[NearNeighbors._get_original_site(molecule, site)]
        image = np.around(np.subtract(site.coords, original_site.coords))
        return tuple(image.astype(int))
