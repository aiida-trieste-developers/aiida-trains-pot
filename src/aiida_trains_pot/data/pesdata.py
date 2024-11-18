from aiida.orm import Data
import tempfile
import os
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import io
from contextlib import redirect_stdout
from ase.io import write

class PESData(Data):
    
    @property
    def _list_key(self):
        """Generate a unique filename for the list based on the node's UUID."""
        return f"psedata_{self.uuid}.npz"  # Unique filename with the node's UUID
    
    def __init__(self, data=None, **kwargs):
        """
        Initialize a PESData instance.

        :param data: Optional list of data to initialize the PESData node.
        :param kwargs: Additional keyword arguments passed to the parent Data class.
        """
        super().__init__(**kwargs)  # Initialize the parent class
        if data:
            self.set_list(data)

    def __iter__(self):
        """Return an iterator over the dataset list."""
        self._index = 0
        self._data = self.get_list()  # Load the list
        return self
    
    def __iadd__(self, other):
        """Support the += operation for combining two PESData objects by creating a new node."""
        if not isinstance(other, PESData):
            raise TypeError(f"Cannot add {type(other)} to PESData")

        # Get lists from both PESData objects
        current_list = self.get_list()
        other_list = other.get_list()

        # Concatenate the lists
        combined_list = current_list + other_list

        # Create a new PESData node with the combined list
        new_pes_data = PESData()
        new_pes_data.set_list(combined_list)

        return new_pes_data


    def __next__(self):
        """Return the next item from the dataset list."""
        if self._index < len(self._data):
            result = self._data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
    def get_list(self):
        """Return the contents of this node as a list."""
        try:            
            # Open the file and load its contents
            with self.base.repository.as_path(self._list_key) as f:
                data = np.load(f, allow_pickle=True)
                return [item for _, val in data.items() for item in (val.tolist() if isinstance(val, np.ndarray) else val)]
                

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []

        except Exception as e:
            print(f"An error occurred while reading {self._list_key}: {e}")
            return []

    def set_list(self, data):
        """Set the contents of this node by saving a list as a file."""
        # Ensure data is a list
        if not isinstance(data, list):
            raise TypeError("Input data must be a list.")

        try:
            # Create a temporary directory to save the file
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_temp_file = os.path.join(temp_dir, self._list_key)

                # Save the data using numpy
                np.savez(dataset_temp_file, data=data)

                # Store the file in the AiiDA repository
                self.base.repository.put_object_from_file(dataset_temp_file, self._list_key)

        except Exception as e:
            print(f"An error occurred while saving {self._list_key}: {e}")

    def get_ase_list(self):
        """Convert dataset list to an ASE list."""

        ase_list = []
        dataset_list = self.get_list()
        for config in dataset_list:
            ase_list.append(Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell']))
            if 'dft_stress' in config.keys():
                s = config['dft_stress']
                stress = [s[0][0] ,s[1][1], s[2][2], s[1][2], s[0][2], s[0][1]]
            if 'dft_energy' in config.keys() and 'dft_forces' in config.keys():
                ase_list[-1].set_calculator(SinglePointCalculator(ase_list[-1], energy=config['dft_energy'], forces=config['dft_forces'], stress=stress))
        
        return ase_list
    
    def get_txt(self):
        """Convert dataset list to xyz file."""

        dataset_txt = ''

        exclude_params = ['cell', 'symbols', 'positions', 'forces', 'stress', 'energy', 'dft_forces', 'dft_stress', 'dft_energy', 'md_forces', 'md_stress', 'md_energy']
        for config in self.get_list():
            params = [key for key in config.keys() if key not in exclude_params]
            atm = Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'])
            atm.info = {}
            for key in params:
                atm.info[key] = config[key]
            if 'dft_stress' in config.keys():
                s = config['dft_stress']
                atm.info['dft_stress'] = f"{s[0][0]:.6f} {s[1][1]:.6f} {s[2][2]:.6f} {s[1][2]:.6f} {s[0][2]:.6f} {s[0][1]:.6f}"

            if 'dft_energy' in config.keys():
                atm.info['dft_energy'] = config['dft_energy']
            if 'dft_forces' in config.keys():
                atm.set_calculator(SinglePointCalculator(atm, forces=config['dft_forces']))
            
            with io.StringIO() as buf, redirect_stdout(buf):
                write('-', atm, format='extxyz', write_results=True, write_info=True)
                dataset_txt += buf.getvalue()


        return dataset_txt