"""DatasetAugmentationWorkChain to generate a training dataset."""

import math
import random
import time

from random import randint, uniform

import numba
import numpy as np

from aiida import load_profile
from aiida.engine import WorkChain, calcfunction, if_
from aiida.orm import Bool, Float, Int, List
from aiida.plugins import DataFactory
from ase import Atoms
from ase.build import surface

load_profile()


StructureData = DataFactory("core.structure")
SinglefileData = DataFactory("core.singlefile")
PESData = DataFactory("pesdata")


def ase_to_dict(ase_structure):
    """Convert an ASE structure to a dictionary."""
    return {
        "cell": ase_structure.get_cell().tolist(),
        "symbols": ase_structure.get_chemical_symbols(),
        "positions": ase_structure.get_positions().tolist(),
        "pbc": ase_structure.get_pbc(),
    }


def check_vacuum(structure, vacuum):
    """Check if vacuum along non periodic directions is enough and add it if necessary.

    :param structure: An ASE structure
    :param vacuum: The minimum vacuum along non periodic directions
    """
    cell = structure.get_cell()
    pbc = structure.get_pbc()
    positions = structure.get_positions()
    for i in range(3):
        if not pbc[i]:
            if cell[i, i] - np.max(positions[:, i]) + np.min(positions[:, i]) < vacuum:
                cell[i, i] = np.max(positions[:, i]) - np.min(positions[:, i]) + vacuum
    structure.set_cell(cell)
    return structure


def check_min_distace(atm, min_dist):
    """Check if the minimum distance between atomic PBC replicas is greater than min_dist.

    :param atm: An ASE structure
    :param min_dist: The minimum distance between atoms
    """
    cell = atm.get_cell()
    pbc = atm.get_pbc()
    for i in range(-1, 2):
        if not pbc[0] and i != 0:
            continue
        for j in range(-1, 2):
            if not pbc[1] and j != 0:
                continue
            for k in range(-1, 2):
                if not pbc[2] and k != 0:
                    continue
                if i == 0 and j == 0 and k == 0:
                    continue
                if min([np.linalg.norm((i, j, k) @ cell)]) < min_dist:
                    return True, np.abs([i, j, k])
    return False, [0, 0, 0]


def replicate(atm, min_dist, max_atoms=1000):
    """Replicate the structure to have a minimum distance between atoms greater than min_dist.
    However, the number of atoms in the structure must be less than max_atoms.

    :param atm: An ASE structure
    :param min_dist: The minimum distance between atoms
    :param max_atoms: The maximum number of atoms in the structure
    """
    pbc = atm.get_pbc()
    cell_vectors_norm = np.linalg.norm(atm.get_cell(), axis=1)
    min_replicas_x = math.ceil(min_dist / cell_vectors_norm[0]) if pbc[0] else 1
    min_replicas_y = math.ceil(min_dist / cell_vectors_norm[1]) if pbc[1] else 1
    min_replicas_z = math.ceil(min_dist / cell_vectors_norm[2]) if pbc[2] else 1
    replicas = [min_replicas_x, min_replicas_y, min_replicas_z]
    atm2 = atm.copy()
    atm2 = atm2.repeat((min_replicas_x, min_replicas_y, min_replicas_z))
    if len(atm2) > max_atoms:
        replicas = [1, 1, 1]
        atm2 = atm.copy()
        atm2 = atm2.repeat(replicas)
    to_continue, fail_dir = check_min_distace(atm2, min_dist)
    last_modifies = [-1, -1]
    while to_continue:
        for ii, val in enumerate(fail_dir):
            if val and ii not in last_modifies[-1 * np.sum(fail_dir) + 1 :] and np.sum(fail_dir) > 1:
                replicas[ii] += 1
                last_modifies.append(ii)
                break
            elif val and np.sum(fail_dir) == 1:
                replicas[ii] += 1
                last_modifies.append(ii)
                break
        atm_old = atm2.copy()
        atm2 = atm.copy()
        atm2 = atm2.repeat(replicas)
        if len(atm2) > max_atoms:
            atm2 = atm_old.copy()
            break
        to_continue, fail_dir = check_min_distace(atm2, min_dist)
    return atm2


def wrap_and_restore_pbc(atoms: Atoms) -> Atoms:
    """Checks for atoms outside the cell, wraps them, and moves all atoms near the cell center.
    Restores the original PBC settings.

    Parameters:
    atoms (ase.Atoms): Input structure.

    Returns:
    ase.Atoms: Structure with wrapped and centered atoms.
    """
    # Get current cell dimensions
    cell = atoms.get_cell().array
    if np.linalg.det(cell) == 0:
        raise ValueError("Cell is degenerate. Ensure the structure has a valid cell.")

    # Check if any atom is outside the cell boundaries
    positions = atoms.get_positions()
    is_out_of_bounds = np.any((positions < 0) | (positions >= cell.diagonal()), axis=1)

    if np.any(is_out_of_bounds):
        # Save the current PBC configuration
        original_pbc = atoms.get_pbc()

        center_of_cell = cell.diagonal() / 2.0
        com = atoms.get_center_of_mass()
        shift_vector = center_of_cell - com
        atoms.positions += shift_vector

        # Temporarily set PBC to True to ensure correct wrapping
        atoms.set_pbc([True, True, True])
        atoms.wrap()

        # Restore the original PBC settings
        atoms.set_pbc(original_pbc)

    return atoms


@calcfunction
def RattleStrainDefectsStructureGenerator(
    n_configs,
    rattle_fraction,
    max_compressive_strain,
    max_tensile_strain,
    frac_vacancies,
    vacancies_per_config,
    vacuum,
    input_structures,
):
    """Generate structures.

    :param in_structure_list: A list of AiiDA `StructureData` nodes
    :param n_configs: Int with the number of configurations to generate
    :param rattle_fraction: Float with the rattle fraction
    :param max_compressive_strain: Float with the maximum compressive strain factor
    :param max_tensile_strain: Float with the maximum tensile strain factor
    :param frac_vacancies: Float with the fraction of vacancies
    :param vacancies_per_config: Int with the number of vacancies per configuration
    :param vacuum: Float with the vacuum along non periodic directions
    :param input_structures: A PESData dataset with the input structures
    """
    structures = []
    input_structures = input_structures.get_ase_list()
    for structure in input_structures:
        if vacuum.value > 0:
            ase_structure = check_vacuum(structure, vacuum.value)
        else:
            ase_structure = structure
        min_interatomic_distances = get_min_interatomic_distances(
            ase_structure.get_positions(), np.array(ase_structure.get_cell())
        )

        for i in range(int(n_configs)):
            if i < int(n_configs) * frac_vacancies:
                n_vacancies = vacancies_per_config.value
            else:
                n_vacancies = 0

            mod_structure = ase_structure.copy()
            sigma_strain = uniform(1 - max_compressive_strain.value, 1 + max_tensile_strain.value)
            mod_structure.set_cell(ase_structure.get_cell() * sigma_strain, scale_atoms=True)
            mod_structure.set_positions(
                uniform_random_atomic_displacement(
                    mod_structure.get_positions(),
                    min_interatomic_distances * sigma_strain,
                    rattle_fraction.value,
                )
            )
            for _ in range(int(n_vacancies)):
                rnd = randint(0, len(mod_structure.get_positions()) - 1)
                del mod_structure[rnd]

            structures.append(ase_to_dict(wrap_and_restore_pbc(mod_structure)))
            structures[-1]["rattle_fraction"] = rattle_fraction.value
            structures[-1]["sigma_strain"] = sigma_strain
            structures[-1]["n_vacancies"] = n_vacancies
            structures[-1]["gen_method"] = "RATTLE_STRAIN_DEFECTS"

    pes_dataset = PESData(structures)
    return {"rattle_strain_defects_structures": pes_dataset}


@calcfunction
def InputStructureGenerator(vacuum, input_structures):
    """Add input structures to the dataset.

    :param vacuum: Float with the vacuum along non periodic directions
    :param input_structures: A PESData dataset with the input structures
    """
    structures = []
    input_structures = input_structures.get_ase_list()
    for structure in input_structures:
        if vacuum.value > 0:
            ase_structure = check_vacuum(structure, vacuum.value)
        else:
            ase_structure = structure

        structures.append(ase_to_dict(wrap_and_restore_pbc(ase_structure)))
        structures[-1]["gen_method"] = "INPUT_STRUCTURE"

    pes_dataset = PESData(structures)
    return {"input_structures": pes_dataset}


@calcfunction
def IsolatedStructureGenerator(vacuum, input_structures):
    """Generate isolated atoms.

    :param vacuum: Float with the vacuum along all directions
    :param input_structures: A PESData dataset with the input structures
    """
    structures = []
    done_types = []
    input_structures = input_structures.get_ase_list()
    for structure in input_structures:
        for atm_type in list(set(structure.get_chemical_symbols())):
            if atm_type not in done_types:
                done_types.append(atm_type)
                isolated_structure = Atoms(
                    atm_type,
                    positions=[[0.0, 0.0, 0.0]],
                    cell=[[vacuum, 0.0, 0.0], [0.0, vacuum, 0.0], [0.0, 0.0, vacuum]],
                    pbc=False,
                )

                structures.append(ase_to_dict(wrap_and_restore_pbc(isolated_structure)))
                structures[-1]["gen_method"] = "ISOLATED_ATOM"

    pes_dataset = PESData(structures)
    return {"isolated_atoms_structure": pes_dataset}


@calcfunction
def ClustersGenerator(n_clusters, max_atoms, interatomic_distance, vacuum, input_structures):
    """Generate clusters.

    :param n_clusters: Int with the number of clusters to generate
    :param n_atoms: Int with the maximum number of atoms in each cluster
    :param interatomic_distance: Float with the interatomic distance
    :param vacuum: Float with the vacuum along non periodic directions
    :param input_structures: A PESData dataset with the input structures
    """
    atomic_species = []
    input_structures = input_structures.get_ase_list()
    for structure in input_structures:
        for atm_type in list(set(structure.get_chemical_symbols())):
            if atm_type not in atomic_species:
                atomic_species.append(atm_type)

    structures = []
    n_clusters = n_clusters.value
    max_atoms = max_atoms.value
    interatomic_distance = interatomic_distance.value
    for _ in range(n_clusters):
        species = [random.choice(atomic_species)]
        positions = [np.array([0, 0, 0])]
        for _ in range(random.randint(2, max_atoms)):
            species.append(random.choice(atomic_species))
            while True:
                position = (
                    np.array([random.uniform(-interatomic_distance, interatomic_distance) for _ in range(3)])
                    + positions[random.randint(0, len(positions) - 1)]
                )

                if all(np.linalg.norm(position - np.array(pos)) >= interatomic_distance for pos in positions):
                    break
            positions.append(position)
            atoms = check_vacuum(Atoms(symbols=species, positions=positions, pbc=False), vacuum)
        structures.append(ase_to_dict(wrap_and_restore_pbc(atoms)))
        structures[-1]["gen_method"] = "CLUSTER"

    return {"cluster_structures": PESData(structures)}


@calcfunction
def SubstitutionGenerator(fraction_substitutions, fraction_structures, **input_datasets):
    """Generate structures with substitutions.

    :param input_datasets: PESData datasets with the input structures
    :param fraction_substitutions: Float with the fraction of substitutions
    :param fraction_structures: Float with the fraction of structures to generate
    """
    structures = []
    for _, input_dataset in input_datasets.items():
        input_structures = input_dataset.get_ase_list()
        for structure in input_structures:
            if np.random.random() < fraction_structures.value:
                structures.append(
                    ase_to_dict(wrap_and_restore_pbc(atoms_substitution(structure, fraction_substitutions.value)))
                )
                structures[-1]["gen_method"] = "SUBSTITUTION"
    pes_dataset = PESData(structures)
    return {"substituted_structures": pes_dataset}


@calcfunction
def SlabsGenerator(miller_indices, min_thickness, max_atoms, vacuum, input_structures):
    """Generate slabs.

    :param n_slabs: Int with the number of slabs to generate
    :param miller_indices: List of lists with the Miller indices
    :param min_thickness: Float with the minimum thickness of the slab
    :param max_atoms: Int with the maximum number of atoms in the slab
    :param vacuum: Float with the vacuum to add
    :param input_structures: A PESData dataset with the input structures
    """
    structures = []
    miller_indices = miller_indices.get_list()
    vacuum = vacuum.value
    min_thickness = min_thickness.value
    input_structures = input_structures.get_ase_list()
    for ase_structure in input_structures:
        if not ase_structure.get_pbc().all():
            continue
        for indices in miller_indices:
            slab = ase_structure.copy()
            slab = surface(
                indices=tuple(indices),
                layers=1,
                vacuum=vacuum / 2,
                lattice=ase_structure,
            )
            layers = 1
            while min_thickness > slab.get_cell()[2, 2] - vacuum:
                slab = surface(
                    indices=tuple(indices),
                    layers=layers,
                    vacuum=vacuum / 2,
                    lattice=ase_structure,
                )
                if len(slab) > max_atoms.value:
                    slab = surface(
                        indices=tuple(indices),
                        layers=layers - 1,
                        vacuum=vacuum / 2,
                        lattice=ase_structure,
                    )
                    break
                layers += 1
            structures.append(ase_to_dict(wrap_and_restore_pbc(slab)))
            structures[-1]["gen_method"] = "SLAB"
    pes_dataset = PESData(structures)
    return {"slab_structures": pes_dataset}


@calcfunction
def ReplicateStructures(min_dist, max_atoms, vacuum, input_structures):
    """Replicate structures to have a minimum distance between atoms greater than min_dist.
    However, the number of atoms in the structure must be less than max_atoms.

    :param min_dist: Float with the minimum distance between atoms
    :param max_atoms: Int with the maximum number of atoms in the structure
    :param in_structure_list: List of AiiDA `StructureData` nodes
    :param vacuum: Float with the vacuum along non periodic directions
    :param input_structures: A PESData dataset with the input structures
    """
    structures = []
    min_dist = min_dist.value
    max_atoms = max_atoms.value
    input_structures = input_structures.get_ase_list()
    for structure in input_structures:
        if vacuum.value > 0:
            ase_structure = check_vacuum(structure, vacuum.value)
        else:
            ase_structure = structure
        replicated_structure = replicate(ase_structure, min_dist, max_atoms)
        replicated_structure.wrap()
        structures.append(ase_to_dict(replicated_structure))

    pes_dataset = PESData(structures)
    return {"replicated_structures": pes_dataset}


@calcfunction
def AlloysGenerator(fixed_species, alloy_species, num_structures, alloy_fractions=None, **input_datasets):
    """Generate structures with random substitutions to create alloys."""
    alloys = []
    input_structures = []
    for _, input_dataset in input_datasets.items():
        input_structures += input_dataset.get_ase_list()
    rng = np.random.default_rng(int(time.time()))
    input_structures = [input_structures[0]]
    while len(alloys) < num_structures:
        sel = input_structures[rng.integers(len(input_structures))]
        alloy = random_substitute_atoms(sel, fixed_species, alloy_species, alloy_fractions)
        alloys.append(ase_to_dict(alloy))
        alloys[-1]["gen_method"] = "ALLOY"
    pes_dataset = PESData(alloys)
    return {"substituted_structures": pes_dataset}


@calcfunction
def WriteDataset(**dataset_in):
    """Combine all generated datasets into a single PESData dataset."""
    dataset_out = []
    for _, dataset in dataset_in.items():
        dataset_out.extend(dataset)
    pes_dataset_out = PESData(dataset_out)
    return {"global_structures": pes_dataset_out}


@numba.njit(parallel=True)
def get_min_interatomic_distances(positions, cell):
    """For each atom, calculate the minimum distance to any other atom in the structure.

    :param positions: A numpy array of atomic positions
    :param cell: A numpy array of the cell vectors
    """
    N_P, _ = positions.shape
    N_C, _ = cell.shape
    min_dist = np.zeros(N_P)
    dist = np.zeros((N_P, N_P))
    for ii in numba.prange(N_P):
        for jj in numba.prange(N_P):
            hidden_dist = np.zeros((N_C, N_C, N_C))
            for i in numba.prange(-1, 2):
                for j in numba.prange(-1, 2):
                    for k in numba.prange(-1, 2):
                        for li in numba.prange(N_C):
                            hidden_dist[i, j, k] += (
                                positions[ii, li]
                                - positions[jj, li]
                                + i * cell[0, li]
                                + j * cell[1, li]
                                + k * cell[2, li]
                            ) ** 2
                        hidden_dist[i, j, k] = np.sqrt(hidden_dist[i, j, k])
            dist[ii, jj] = np.min(hidden_dist)
            if ii == jj:
                dist[ii, jj] = np.inf
        min_dist[ii] = np.min(dist[ii, :])
    return min_dist


@numba.njit(parallel=True)
def uniform_random_atomic_displacement(positions, min_distances, max_displacement_fraction):
    """Displace atoms randomly in a uniform manner.

    :param positions: A numpy array of atomic positions
    :param min_distances: A numpy array of minimum interatomic distances
    :param max_displacement_fraction: A float that determines the maximum displacement
        as a fraction of the minimum interatomic distance
    """
    N_P, _ = positions.shape
    for ii in numba.prange(N_P):
        rand_dir = np.array([uniform(0, 1), uniform(0, 1), uniform(0, 1)])
        rand_dir /= np.sqrt(rand_dir[0] ** 2 + rand_dir[1] ** 2 + rand_dir[2] ** 2)
        positions[ii] += uniform(0, 1) * min_distances[ii] * max_displacement_fraction * rand_dir
    return positions


def random_substitute_atoms(
    atoms: Atoms,
    fixed_species=None,
    substitute_species=None,
    fractions=None,
) -> Atoms:
    """Sostitutes randomly the species of the input `atoms`, keeping fixed those in `fixed_species`."""
    if substitute_species is None or len(substitute_species) == 0:
        return None

    rng = np.random.default_rng(int(time.time()))
    fixed_species = list(fixed_species) if fixed_species else []

    syms = atoms.get_chemical_symbols()
    replace_idx = [i for i, s in enumerate(syms) if s not in fixed_species]
    R = len(replace_idx)
    if R == 0:
        return atoms.copy()

    subs = list(substitute_species)
    k = len(subs)

    if fractions is None:
        # Random fractions
        p = rng.dirichlet(np.ones(k))
    elif isinstance(fractions, dict):
        p = np.array([float(fractions.get(sp, 0.0)) for sp in subs], dtype=float)
        p /= p.sum()
    else:
        p = np.array(fractions, dtype=float)
        if p.shape[0] != k:
            return None
        p /= p.sum()

    raw = p * R
    counts = np.floor(raw).astype(int)
    rem = R - counts.sum()
    if rem > 0:
        order = np.argsort(-(raw - counts))  # descending fractional parts
        for t in range(rem):
            counts[order[t % k]] += 1

    # pool e shuffle
    pool = [sp for sp, c in zip(subs, counts, strict=False) for _ in range(int(c))]
    if len(pool) < R:
        pool += [subs[0]] * (R - len(pool))
    elif len(pool) > R:
        pool = pool[:R]
    rng.shuffle(pool)

    # apply
    new_syms = syms[:]
    for idx, sp in zip(replace_idx, pool, strict=False):
        new_syms[idx] = sp

    out = atoms.copy()
    out.set_chemical_symbols(new_syms)
    return out


def atoms_substitution(structure, fraction_substitution):
    """Substitute atoms in the structure with random atoms from the same structure.

    :param structure: An ASE structure
    :param fraction_substitution: A float that determines the fraction of atoms to be substituted
    """
    symbols = structure.get_chemical_symbols()
    fraction_substitution = 0.2

    num_substitutions = np.random.randint(0, len(symbols)) * fraction_substitution
    count_substitutions = 0
    substituted_symbols = symbols.copy()
    while count_substitutions < num_substitutions:
        rnd1 = np.random.randint(0, len(symbols))
        rnd2 = np.random.randint(0, len(symbols))
        if symbols[rnd1] == substituted_symbols[rnd1] and symbols[rnd2] == substituted_symbols[rnd2]:
            substituted_symbols[rnd1] = symbols[rnd2]
            substituted_symbols[rnd2] = symbols[rnd1]
            count_substitutions += 1

    structure.set_chemical_symbols(substituted_symbols)
    return structure


class DatasetAugmentationWorkChain(WorkChain):
    """WorkChain to generate a training dataset."""

    ######################################################
    ##                 DEFAULT VALUES                   ##
    ######################################################
    DEFAULT_RSD_rattle_fraction = Float(0.3)
    DEFAULT_RSD_max_compressive_strain = Float(0.2)
    DEFAULT_RSD_max_tensile_strain = Float(0.6)
    DEFAULT_RSD_n_configs = Int(50)
    DEFAULT_RSD_frac_vacancies = Float(0.3)
    DEFAULT_RSD_vacancies_per_config = Int(2)
    DEFAULT_clusters_n_clusters = Int(20)
    DEFAULT_clusters_max_atoms = Int(10)
    DEFAULT_clusters_interatomic_distance = Float(1.5)
    DEFAULT_slabs_miller_indices = List([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    DEFAULT_slabs_min_thickness = Float(10.0)
    DEFAULT_slabs_max_atoms = Int(450)
    DEFAULT_replicate_min_dist = Float(18.0)
    DEFAULT_replicate_max_atoms = Int(450)
    DEFAULT_vacuum = Float(15.0)
    DEFAULT_max_substitution_fraction = Float(0.2)
    DEFAULT_substitution_fraction = Float(0.2)
    DEFAULT_alloys_num_structures = Int(100)

    DEFAULT_do_rattle_strain_defects = Bool(True)
    DEFAULT_do_input = Bool(True)
    DEFAULT_do_isolated = Bool(True)
    DEFAULT_do_clusters = Bool(True)
    DEFAULT_do_slabs = Bool(True)
    DEFAULT_do_replicate = Bool(True)
    DEFAULT_do_check_vacuum = Bool(True)
    DEFAULT_do_substitution = Bool(True)
    DEFAULT_do_alloys = Bool(True)
    ######################################################

    @classmethod
    def define(cls, spec):
        """Specify inputs and outputs."""
        super().define(spec)
        spec.input(
            "structures",
            valid_type=PESData,
            required=True,
            help="PESData, dataset containing input structures.",
        )

        spec.input(
            "do_rattle_strain_defects",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_rattle_strain_defects,
            required=False,
            help="Perform rattle calculations (random atomic displacements, cell stretch/compression, "
            "vacancies. Permutations and replacements are not yet implemented). "
            f"Default: {cls.DEFAULT_do_rattle_strain_defects}",
        )
        spec.input(
            "do_input",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_input,
            required=False,
            help=f"Add input structures to the dataset. Default: {cls.DEFAULT_do_input}",
        )
        spec.input(
            "do_isolated",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_isolated,
            required=False,
            help=f"Add isolated atoms configurations to the dataset. Default: {cls.DEFAULT_do_isolated}",
        )
        spec.input(
            "do_clusters",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_clusters,
            required=False,
            help=f"Add clusters to the dataset. Default: {cls.DEFAULT_do_clusters}",
        )
        spec.input(
            "do_slabs",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_slabs,
            required=False,
            help=f"Add slabs to the dataset. Default: {cls.DEFAULT_do_slabs}",
        )
        spec.input(
            "do_check_vacuum",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_check_vacuum,
            required=False,
            help="Check if vacuum along non periodic directions is enough and add it if necessary. "
            f"Default: {cls.DEFAULT_do_check_vacuum}",
        )
        spec.input(
            "do_replication",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_replicate,
            required=False,
            help="Replicate structures to have a minimum distance between atoms greater than min_dist. "
            f"Default: {cls.DEFAULT_do_replicate}",
        )
        spec.input(
            "do_substitution",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_substitution,
            required=False,
            help=f"Add substituted structures to the dataset. Default: {cls.DEFAULT_do_substitution}",
        )
        spec.input(
            "do_alloys",
            valid_type=Bool,
            default=lambda: cls.DEFAULT_do_alloys,
            required=False,
            help=f"Add alloy structures to the dataset. Default: {cls.DEFAULT_do_alloys}",
        )

        spec.input(
            "rsd.params.rattle_fraction",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_RSD_rattle_fraction,
            required=False,
            help="Atoms are displaced by a rattle_fraction of the minimum interatomic distance. "
            f"Default: {cls.DEFAULT_RSD_rattle_fraction}",
        )
        spec.input(
            "rsd.params.max_compressive_strain",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_RSD_max_compressive_strain,
            required=False,
            help="Maximum compressive strain factor. Cell can be compressed up to this fraction "
            f"of cell parameters. Default: {cls.DEFAULT_RSD_max_compressive_strain}",
        )
        spec.input(
            "rsd.params.max_tensile_strain",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_RSD_max_tensile_strain,
            required=False,
            help="Maximum tensile strain factor. Cell can be stretched up to this fraction of "
            f"cell parameters. Default: {cls.DEFAULT_RSD_max_tensile_strain}",
        )
        spec.input(
            "rsd.params.n_configs",
            valid_type=Int,
            default=lambda: cls.DEFAULT_RSD_n_configs,
            required=False,
            help="Number of configurations to generate per each input structure. "
            f"Default: {cls.DEFAULT_RSD_n_configs}",
        )
        spec.input(
            "rsd.params.frac_vacancies",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_RSD_frac_vacancies,
            required=False,
            help=f"Fraction of configurations with vacancies. Default: {cls.DEFAULT_RSD_frac_vacancies}",
        )
        spec.input(
            "rsd.params.vacancies_per_config",
            valid_type=Int,
            default=lambda: cls.DEFAULT_RSD_vacancies_per_config,
            required=False,
            help=f"Number of vacancies per configuration. Default: {cls.DEFAULT_RSD_vacancies_per_config}",
        )

        spec.input(
            "clusters.n_clusters",
            valid_type=Int,
            default=lambda: cls.DEFAULT_clusters_n_clusters,
            required=False,
            help=f"Number of clusters to generate. Default: {cls.DEFAULT_clusters_n_clusters}",
        )
        spec.input(
            "clusters.max_atoms",
            valid_type=Int,
            default=lambda: cls.DEFAULT_clusters_max_atoms,
            required=False,
            help=f"Maximum number of atoms in each cluster. Default: {cls.DEFAULT_clusters_max_atoms}",
        )
        spec.input(
            "clusters.interatomic_distance",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_clusters_interatomic_distance,
            required=False,
            help=f"Interatomic distance. Default: {cls.DEFAULT_clusters_interatomic_distance}",
        )

        spec.input(
            "slabs.miller_indices",
            valid_type=List,
            default=lambda: cls.DEFAULT_slabs_miller_indices,
            required=False,
            help=f"List of lists with the Miller indices. Default: {cls.DEFAULT_slabs_miller_indices}",
        )
        spec.input(
            "slabs.min_thickness",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_slabs_min_thickness,
            required=False,
            help=f"Minimum thickness of the slab. Default: {cls.DEFAULT_slabs_min_thickness}",
        )
        spec.input(
            "slabs.max_atoms",
            valid_type=Int,
            default=lambda: cls.DEFAULT_slabs_max_atoms,
            required=False,
            help=f"Maximum number of atoms. Default: {cls.DEFAULT_slabs_max_atoms}",
        )

        spec.input(
            "replicate.min_dist",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_replicate_min_dist,
            required=False,
            help="Minimum distance between atoms in PBC replicas, unless max_atoms is reached. "
            f"Default: {cls.DEFAULT_replicate_min_dist}",
        )
        spec.input(
            "replicate.max_atoms",
            valid_type=Int,
            default=lambda: cls.DEFAULT_replicate_max_atoms,
            required=False,
            help="Maximum number of atoms in the supercell. Stronger criteria respect to min_dist. "
            f"Default: {cls.DEFAULT_replicate_max_atoms}",
        )

        spec.input(
            "substitution.switches_fraction",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_substitution_fraction,
            required=False,
            help=f"Fraction of atoms to be substituted. Default: {cls.DEFAULT_substitution_fraction}",
        )
        spec.input(
            "substitution.structures_fraction",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_max_substitution_fraction,
            required=False,
            help=f"Fraction of structures to be substituted. Default: {cls.DEFAULT_max_substitution_fraction}",
        )

        spec.input(
            "alloys.fixed_species",
            valid_type=List,
            required=False,
            help="List of species that will not be substituted in the alloy generation.",
        )
        spec.input(
            "alloys.alloy_species",
            valid_type=List,
            required=False,
            help="List of species to be used for alloy generation.",
        )
        spec.input(
            "alloys.num_structures",
            valid_type=Int,
            default=lambda: cls.DEFAULT_alloys_num_structures,
            required=False,
            help=f"Number of alloy structures to generate. Default: {cls.DEFAULT_alloys_num_structures}",
        )
        spec.input(
            "alloys.fractions",
            valid_type=List,
            required=False,
            help="List of fractions for each alloy species. If not provided, random fractions will be used.",
        )

        spec.input(
            "vacuum",
            valid_type=(Int, Float),
            default=lambda: cls.DEFAULT_vacuum,
            required=False,
            help=f"Minimum vacuum along non periodic directions. Default: {cls.DEFAULT_vacuum}",
        )

        spec.output_namespace("structures", valid_type=PESData, dynamic=True, help="Augmented datasets.")

        spec.inputs.validator = cls.validate_inputs

        spec.outline(cls.setup, if_(cls.do_replication)(cls.replicate), cls.run_dataset_generation)

    @classmethod
    def get_builder_with_structures(cls, structures):
        """Return a builder."""
        builder = cls.get_builder()
        builder.structures = {f"s{ii}": s for ii, s in enumerate(structures)}
        return builder

    @classmethod
    def validate_inputs(cls, inputs, _):  # noqa: PLR0911
        """Check inputs."""
        if inputs["do_rattle_strain_defects"]:
            # ERRORS
            if inputs["rsd"]["params"]["rattle_fraction"] < 0.0 or inputs["rsd"]["params"]["rattle_fraction"] > 1.0:
                return "rattle_fraction must be between 0 and 1"
            if inputs["rsd"]["params"]["max_tensile_strain"] < 0.0:
                return "max_tensile_strain must be greater than 0"
            if (
                inputs["rsd"]["params"]["max_compressive_strain"] < 0.0
                or inputs["rsd"]["params"]["max_compressive_strain"] > 1.0
            ):
                return "max_compressive_strain must be between 0 and 1"
            if inputs["rsd"]["params"]["n_configs"] < 1:
                return "n_configs must be at least 1"
            if inputs["rsd"]["params"]["frac_vacancies"] < 0.0 or inputs["rsd"]["params"]["frac_vacancies"] > 1.0:
                return "frac_vacancies must be between 0 and 1"
            if inputs["rsd"]["params"]["vacancies_per_config"] < 0:
                return "vacancies_per_config must be non-negative"
            if inputs["do_alloys"]:
                if "alloy_species" not in inputs["alloys"] or len(inputs["alloys"]["alloy_species"]) == 0:
                    return "alloy_species must be specified when do_alloys is True"
                if inputs["alloys"]["num_structures"] < 1:
                    return "num_structures must be at least 1"

    def setup(self):
        """Setup workchain."""
        self.ctx.initial_dataset = self.inputs.structures
        if self.inputs.do_check_vacuum:
            self.ctx.vacuum = self.inputs.vacuum
        else:
            self.ctx.vacuum = Float(0)
        if self.inputs.do_alloys:
            self.ctx.alloys_fractions = []
            if "fractions" in self.inputs.alloys:
                self.ctx.alloys_fractions = self.inputs.alloys.fractions
            self.ctx.alloys_fixed_species = []
            if "fixed_species" in self.inputs.alloys:
                self.ctx.alloys_fixed_species = self.inputs.alloys.fixed_species

    def do_replication(self):  # noqa: D102
        return bool(self.inputs.do_replication)

    def replicate(self):
        """Replicate structures."""
        self.report("Replicating structures")
        self.ctx.initial_dataset = ReplicateStructures(
            min_dist=self.inputs.replicate.min_dist,
            max_atoms=self.inputs.replicate.max_atoms,
            vacuum=self.ctx.vacuum,
            input_structures=self.ctx.initial_dataset,
        )["replicated_structures"]

    def run_dataset_generation(self):
        """Generate datasets."""
        dataset = {}
        if self.inputs.do_input:
            dataset["input_structures"] = InputStructureGenerator(
                vacuum=self.ctx.vacuum, input_structures=self.ctx.initial_dataset
            )["input_structures"]
        if self.inputs.do_isolated:
            dataset["isolated_atoms_structure"] = IsolatedStructureGenerator(
                vacuum=self.ctx.vacuum, input_structures=self.ctx.initial_dataset
            )["isolated_atoms_structure"]
        if self.inputs.do_rattle_strain_defects:
            dataset["rattle_strain_defects_structures"] = RattleStrainDefectsStructureGenerator(
                self.inputs.rsd.params.n_configs,
                self.inputs.rsd.params.rattle_fraction,
                self.inputs.rsd.params.max_compressive_strain,
                self.inputs.rsd.params.max_tensile_strain,
                self.inputs.rsd.params.frac_vacancies,
                self.inputs.rsd.params.vacancies_per_config,
                vacuum=self.ctx.vacuum,
                input_structures=self.ctx.initial_dataset,
            )["rattle_strain_defects_structures"]
        if self.inputs.do_clusters:
            dataset["clusters"] = ClustersGenerator(
                self.inputs.clusters.n_clusters,
                self.inputs.clusters.max_atoms,
                self.inputs.clusters.interatomic_distance,
                vacuum=self.ctx.vacuum,
                input_structures=self.ctx.initial_dataset,
            )["cluster_structures"]
        if self.inputs.do_slabs:
            dataset["slabs"] = SlabsGenerator(
                self.inputs.slabs.miller_indices,
                self.inputs.slabs.min_thickness,
                self.inputs.slabs.max_atoms,
                vacuum=self.ctx.vacuum,
                input_structures=self.ctx.initial_dataset,
            )["slab_structures"]
        if self.inputs.do_substitution:
            datasets_to_substitute = {}
            if self.inputs.do_input:
                datasets_to_substitute["input_structures"] = dataset["input_structures"]
            if self.inputs.do_rattle_strain_defects:
                datasets_to_substitute["rattle_strain_defects_structures"] = dataset["rattle_strain_defects_structures"]
            if self.inputs.do_slabs:
                datasets_to_substitute["slabs"] = dataset["slabs"]
            dataset["substituted"] = SubstitutionGenerator(
                self.inputs.substitution.switches_fraction,
                self.inputs.substitution.structures_fraction,
                **datasets_to_substitute,
            )["substituted_structures"]
        if self.inputs.do_alloys:
            datasets_for_alloys = {}
            if self.inputs.do_input:
                datasets_for_alloys["input_structures"] = dataset["input_structures"]
            if self.inputs.do_rattle_strain_defects:
                datasets_for_alloys["rattle_strain_defects_structures"] = dataset["rattle_strain_defects_structures"]
            if self.inputs.do_slabs:
                datasets_for_alloys["slabs"] = dataset["slabs"]
            dataset["alloys"] = AlloysGenerator(
                self.ctx.alloys_fixed_species,
                self.inputs.alloys.alloy_species,
                self.inputs.alloys.num_structures,
                self.ctx.alloys_fractions,
                **datasets_for_alloys,
            )["alloy_structures"]
        dataset["global_structures"] = WriteDataset(**dataset)["global_structures"]
        self.out("structures", dataset)
