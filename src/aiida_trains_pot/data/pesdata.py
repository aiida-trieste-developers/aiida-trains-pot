from aiida.orm import Data
import tempfile
import os
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
import io
from contextlib import redirect_stdout
from ase.io import write
import warnings
import h5py

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
    
    def __add__(self, other):
        if not isinstance(other, PESData):
            raise TypeError(f"Cannot add {type(other)} to PESData")
        return self.__iadd__(other)


    def __iadd__(self, other):
        """Support the += operation for combining two PESData objects by creating a new node."""
        if not isinstance(other, PESData):
            raise TypeError(f"Cannot add {type(other)} to PESData")

        return PESData(data = self.get_list() + other.get_list())


    def __next__(self):
        """Return the next item from the dataset list."""
        if self._index < len(self._data):
            result = self._data[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
        
    def __len__(self):
        """Return the number of configurations in the dataset."""
        if self.base.attributes.get('dataset_size'):
            return self.base.attributes.get('dataset_size')
        else:
            return len(self.get_list())

    def get_list(self):
        """Return the contents of this node as a list."""
        try:
            # Open the HDF5 file and load its contents
            with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                with h5py.File(hdf_file, 'r') as hdf:
                    data = []
                    for group in hdf.values():
                        config = {}
                        for key, value in group.items():
                            # Convert datasets to lists or native Python types
                            if isinstance(value, h5py.Dataset):
                                config[key] = value[()].tolist() if hasattr(value[()], 'tolist') else value[()]
                            else:
                                config[key] = value[()]
                        # Add attributes to the config
                        for attr_key, attr_value in group.attrs.items():
                            config[attr_key] = attr_value
                        data.append(config)

            # Ensure symbols are decoded properly (if they were byte strings)
            for config in data:
                if 'symbols' in config:
                    config['symbols'] = [str(symbol, 'utf-8') if isinstance(symbol, bytes) else str(symbol) for symbol in config['symbols']]

            return data

        except FileNotFoundError as e:
            print(f"File '{self._list_key}' not found: {e}")
            return []
        except Exception as e:
            print(f"An error occurred while reading '{self._list_key}': {e}")
            return []

    def set_list(self, data):
        """Set the contents of this node by saving a list as an HDF5 file."""
        # Ensure data is a list
        if not isinstance(data, list):
            raise TypeError("Input data must be a list.")
        num_labelled_frames = 0
        num_unlabelled_frames = 0
        for item in data:
            if "dft_forces" in item.keys() and "dft_energy" in item.keys():
                num_labelled_frames += 1
            else:
                num_unlabelled_frames += 1

        save_data = []
        for item in data:
            if 'pbc' not in item:
                item['pbc'] = [True, True, True]
                warnings.warn("Periodic boundary conditions not found in the dataset. Assuming PBC = [True, True, True].", UserWarning)
            if 'cell' not in item:
                raise ValueError("Cell vectors not found in the dataset.")
            if 'symbols' not in item:
                raise ValueError("Atomic symbols not found in the dataset.")
            if 'positions' not in item:
                raise ValueError("Atomic positions not found in the dataset.")
        for item in data:
            # Ensure that symbols are a list of strings
            item['symbols'] = [str(symbol) for symbol in item['symbols']]
            save_data.append({key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()})

        try:
            # Create a temporary directory to save the file
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_temp_file = os.path.join(temp_dir, f"{self._list_key}.h5")
                # Save the data using h5py
                with h5py.File(dataset_temp_file, 'w') as hdf:
                    for idx, item in enumerate(save_data):
                        group = hdf.create_group(f"item_{idx}")
                        for key, value in item.items():
                            if isinstance(value, list) or isinstance(value, np.ndarray):
                                group.create_dataset(key, data=value)
                            else:
                                group.attrs[key] = value

                # Store the file in the AiiDA repository
                self.base.repository.put_object_from_file(dataset_temp_file, self._list_key)
            # Store metadata as attributes
            self.base.attributes.set('dataset_size', len(data))
            self.base.attributes.set('num_labelled_frames', num_labelled_frames)
            self.base.attributes.set('num_unlabelled_frames', num_unlabelled_frames)
        except Exception as e:
            print(f"An error occurred while saving '{self._list_key}': {e}")

    def get_ase_list(self):
        """Convert dataset list to an ASE list."""

        ase_list = []
        dataset_list = self.get_list()
        for config in dataset_list:
            ase_list.append(Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'], pbc=config['pbc']))
            if 'dft_stress' in config.keys():
                stress = config['dft_stress']
            if 'dft_energy' in config.keys() and 'dft_forces' in config.keys():
                ase_list[-1].set_calculator(SinglePointCalculator(ase_list[-1], energy=config['dft_energy'], forces=config['dft_forces'], stress=stress))
        
        return ase_list
    
    def get_txt(self, write_params=False, key_prefix=''):
        """Convert dataset list to xyz file."""

        dataset_txt = ''
        if not key_prefix.endswith('_') and key_prefix != '':
            key_prefix += '_'
        exclude_params = ['cell', 'symbols', 'positions', 'pbc', 'forces', 'stress', 'energy', 'dft_forces', 'dft_stress', 'dft_energy', 'md_forces', 'md_stress', 'md_energy']
        for config in self.get_list():
            params = [key for key in config.keys() if key not in exclude_params]
            atm = Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'], pbc=config['pbc'])
            atm.info = {}
            if write_params:
                for key in params:
                    atm.info[key] = config[key]
            if len(atm.get_chemical_symbols()) == 1:
                atm.info["config_type"] = "IsolatedAtom"
            if 'dft_stress' in config.keys():
                s = config['dft_stress']
                atm.info[f'{key_prefix}stress'] = f"{s[0][0]:.6f} {s[0][1]:.6f} {s[0][2]:.6f} {s[1][0]:.6f} {s[1][1]:.6f} {s[1][2]:.6f} {s[2][0]:.6f} {s[2][1]:.6f} {s[2][2]:.6f}"
                # atm.info['dft_stress'] = f"{s[0][0]:.6f} {s[1][1]:.6f} {s[2][2]:.6f} {s[1][2]:.6f} {s[0][2]:.6f} {s[0][1]:.6f}"

            if 'dft_energy' in config.keys():
                atm.info[f'{key_prefix}energy'] = config['dft_energy']
            if 'dft_forces' in config.keys():
                atm.set_calculator(SinglePointCalculator(atm, forces=config['dft_forces']))
            
            with io.StringIO() as buf, redirect_stdout(buf):
                write('-', atm, format='extxyz', write_results=True, write_info=True)
                dataset_txt += buf.getvalue()
        if key_prefix != '':
            dataset_txt = dataset_txt.replace(':forces', f':{key_prefix}forces')
        return dataset_txt
    
        
    def get_unlabelled(self):
        """Return a PESData object with only unlabelled frames."""
        unlabelled_data = [config for config in self.get_list() if 'dft_forces' not in config.keys() or 'dft_energy' not in config.keys()]
        return PESData(data=unlabelled_data)

    def get_labelled(self):
        """Return a PESData object with only labelled frames."""
        labelled_data = [config for config in self.get_list() if 'dft_forces' in config.keys() and 'dft_energy' in config.keys()]
        return PESData(data=labelled_data)
    
    @property
    def len_unlabelled(self):
        """Return the number of unlabelled configurations in the dataset."""
        if "num_unlabelled_frames" in self.base.attributes.keys():
            return self.base.attributes.get('num_unlabelled_frames')
        else:
            return len(self.get_unlabelled())
    
    @property
    def len_labelled(self):
        """Return the number of labelled configurations in the dataset."""
        if "num_labelled_frames" in self.base.attributes.keys():
            return self.base.attributes.get('num_labelled_frames')
        else:
            return len(self.get_labelled())