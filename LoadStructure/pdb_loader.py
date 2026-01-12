# ruff: noqa
import os

from aiida.engine import calcfunction
from aiida.orm import List
from aiida.orm import StructureData
from ase.io import read


@calcfunction
def load_structures_from_folder(folder_path):
    structures = []

    full_folder_path = os.path.join(f"{os.path.abspath(os.getcwd())}", str(folder_path.value))
    # Iterate through all files in the folder
    for filename in os.listdir(full_folder_path):
        # Check if the file is a PDB file

        if filename.endswith(".cif") or filename.endswith(".xyz"):
            pdb_file = os.path.join(full_folder_path, filename)
            # Read the structure from the PDB file using ASE
            atoms = read(pdb_file)
            # Create a StructureData node
            structure = StructureData(ase=atoms)
            structure.store()
            structures.append(structure)

    structure_uuids = [structure.uuid for structure in structures]

    return {"uuids": List(structure_uuids)}
