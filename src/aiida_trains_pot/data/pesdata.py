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
import re

def convert_stress(stress):
    if len(np.shape(stress)) == 1:
        if len(stress) == 6:
            stress = np.array([[stress[0], stress[5], stress[4]],
                               [stress[5], stress[1], stress[3]],
                               [stress[4], stress[3], stress[2]]])
        else:
            stress = np.array([[stress[0], stress[1], stress[2]],
                               [stress[3], stress[4], stress[5]],
                               [stress[6], stress[7], stress[8]]])
    else:
        stress = stress
    return stress


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
            if isinstance(data[0], Atoms):
                self.set_ase(data)
            else:
                self.set_list(data)

    def __iter__(self):
        """Return an iterator over the dataset."""
        self._index = 0
        self._max_index = self.base.attributes.get('dataset_size', 0)
        if self._max_index == 0:
            # Count the number of items in the HDF5 file
            try:
                with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                    with h5py.File(hdf_file, 'r') as hdf:
                        self._max_index = len(hdf)
            except (FileNotFoundError, KeyError):
                self._max_index = 0
        return self

    def __next__(self):
        """Return the next item from the dataset."""
        if self._index < self._max_index:
            result = self.get_item(self._index)
            self._index += 1
            return result
        else:
            raise StopIteration
    
    def __add__(self, other):
        if not isinstance(other, PESData):
            raise TypeError(f"Cannot add {type(other)} to PESData")
        return self.__iadd__(other)


    def __iadd__(self, other):
        """Support the += operation for combining two PESData objects by creating a new node."""
        if not isinstance(other, PESData):
            raise TypeError(f"Cannot add {type(other)} to PESData")

        return PESData(data = self.get_list() + other.get_list())

        
    def __len__(self):
        """Return the number of configurations in the dataset."""
        try:
            if self.base.attributes.get('dataset_size'):
                return self.base.attributes.get('dataset_size')
            else:
                return len(self.get_list())
        except:
            return 0

    def _extract_config_from_group(self, group):
        """
        Helper method to extract configuration data from an HDF5 group.
        
        :param group: HDF5 group object containing configuration data
        :return: Dictionary with the configuration data
        """
        config = {}
        # Extract datasets
        for key, value in group.items():
            # Convert datasets to lists or native Python types
            if isinstance(value, h5py.Dataset):
                config[key] = value[()].tolist() if hasattr(value[()], 'tolist') else value[()]
            else:
                config[key] = value[()]
        
        # Add attributes to the config
        for attr_key, attr_value in group.attrs.items():
            config[attr_key] = attr_value
        
        # Ensure symbols are decoded properly
        if 'symbols' in config:
            config['symbols'] = [str(symbol, 'utf-8') if isinstance(symbol, bytes) else str(symbol) for symbol in config['symbols']]
        
        return config

    def get_atomic_species(self):
        """Return the list of atomic species in the dataset."""
        try:
            return self.base.attributes.get('atomic_species', [])
        except Exception as e:
            print(f"An error occurred while retrieving atomic species: {e}")
            return []

    def get_item(self, index):
        """Return a specific item from the dataset by index."""
        try:
            with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                with h5py.File(hdf_file, 'r') as hdf:
                    group_key = f"item_{index}"
                    if group_key not in hdf:
                        raise IndexError(f"Index {index} out of range")
                    
                    return self._extract_config_from_group(hdf[group_key])
                    
        except FileNotFoundError as e:
            print(f"File '{self._list_key}' not found: {e}")
            raise IndexError(f"Index {index} out of range") from e
        except Exception as e:
            print(f"An error occurred while reading '{self._list_key}': {e}")
            raise

    def iter_items(self):
        """Generator function to iterate through items without loading all into memory."""
        try:
            with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                with h5py.File(hdf_file, 'r') as hdf:
                    for group_key in sorted(hdf.keys(), key=lambda k: int(k.split('_')[1])):
                        yield self._extract_config_from_group(hdf[group_key])
        except FileNotFoundError as e:
            print(f"File '{self._list_key}' not found: {e}")
        except Exception as e:
            print(f"An error occurred while reading '{self._list_key}': {e}")

    def get_list(self, max_items=None, warn_threshold=1000):
        """
        Return the contents of this node as a list.
        
        :param max_items: Optional limit to the number of items to load
        :param warn_threshold: Show warning if loading more than this many items
        """
        n_items = self.base.attributes.get('dataset_size', 0)
        if n_items == 0:
            # Count the number of items in the HDF5 file
            try:
                with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                    with h5py.File(hdf_file, 'r') as hdf:
                        n_items = len(hdf)
            except (FileNotFoundError, KeyError):
                n_items = 0
        
        if max_items is None:
            max_items = n_items
        else:
            max_items = min(max_items, n_items)
        
        if max_items > warn_threshold:
            warnings.warn(f"Loading {max_items} items into memory. This may consume a significant amount of RAM. "
                         f"Consider using iter_items() for memory-efficient iteration.", UserWarning)
        
        # Use the iterator to build the list, which is more consistent and reuses code
        data = []
        for i, config in enumerate(self.iter_items()):
            if i >= max_items:
                break
            data.append(config)
        
        return data

    def set_ase(self, data):
        """
        Set the contents of this node by saving a list of ASE Atoms objects as an HDF5 file.

        :param data: A list of ASE Atoms objects to save.
        """
        from ase.calculators.singlepoint import SinglePointDFTCalculator as dft_calc
        # Ensure data is a list of Atoms objects
        if not isinstance(data, list):
            raise TypeError("Input data must be a list of ase.atoms.Atoms.")
        else:
            for item in data:
                if not isinstance(item, Atoms):
                    raise TypeError("Input data must be a list of ase.atoms.Atoms.")
        
        num_labelled_frames = 0
        num_unlabelled_frames = 0
        symb = set()
        
        save_data = []
        for atm in data:
            if isinstance(atm.calc, dft_calc):
                num_labelled_frames += 1
                
                save_data.append({'cell': atm.cell, 'symbols': atm.get_chemical_symbols(), 'positions': atm.get_positions(), 'pbc': atm.pbc, 'dft_energy': atm.calc.results['energy'], 'dft_forces': atm.calc.results['forces']})
                try:
                    stress = atm.get_stress(voigt=False)
                    save_data[-1]['dft_stress'] = stress
                except:
                    continue
            else:
                num_unlabelled_frames += 1
                save_data.append({'cell': atm.cell, 'symbols': atm.get_chemical_symbols(), 'positions': atm.get_positions(), 'pbc': atm.pbc})

            symb = symb.union(set(save_data[-1]['symbols']))

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
            self.base.attributes.set('atomic_species', list(symb))
        except Exception as e:
            print(f"An error occurred while saving '{self._list_key}': {e}")          




    def set_list(self, data):
        """Set the contents of this node by saving a list as an HDF5 file."""
        # Ensure data is a list
        if not isinstance(data, list):
            raise TypeError("Input data must be a list.")
        num_labelled_frames = 0
        num_unlabelled_frames = 0
        symb = set()
        for item in data:
            if "dft_forces" in item.keys() and "dft_energy" in item.keys():
                num_labelled_frames += 1
            else:
                num_unlabelled_frames += 1
            symb = symb.union(set(item['symbols']))

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
            self.base.attributes.set('atomic_species', list(symb))
        except Exception as e:
            print(f"An error occurred while saving '{self._list_key}': {e}")
    
    def _config_to_ase(self, config):
        """
        Helper method to convert a configuration dictionary to an ASE Atoms object.
        
        :param config: Dictionary containing atomic configuration data
        :return: ASE Atoms object
        """
        atoms = Atoms(
            symbols=config['symbols'], 
            positions=config['positions'], 
            cell=config['cell'], 
            pbc=config['pbc']
        )
        
        # Add calculator with DFT data if available
        if 'dft_energy' in config and 'dft_forces' in config:
            calc_kwargs = {
                'energy': config['dft_energy'],
                'forces': config['dft_forces']
            }
            if 'dft_stress' in config:
                calc_kwargs['stress'] = convert_stress(config['dft_stress'])
            
            atoms.set_calculator(SinglePointCalculator(atoms, **calc_kwargs))
        
        return atoms

    def get_ase_item(self, index):
        """
        Get a specific configuration as an ASE Atoms object.
        
        :param index: Index of the configuration to retrieve
        :return: ASE Atoms object
        """
        config = self.get_item(index)
        return self._config_to_ase(config)
    
    def get_ase(self, index=None):
        if index is not None:
            return self.get_ase_item(index)
        else:
            return self.get_ase_list()

    def get_ase_list(self, max_items=None, warn_threshold=1000):
        """
        Convert dataset to a list of ASE Atoms objects.
        
        :param max_items: Optional limit to the number of items to load
        :param warn_threshold: Show warning if loading more than this many items
        :return: List of ASE Atoms objects
        """
        n_items = self.base.attributes.get('dataset_size', 0)
        if n_items == 0:
            # Count the number of items in the HDF5 file
            try:
                with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                    with h5py.File(hdf_file, 'r') as hdf:
                        n_items = len(hdf)
            except (FileNotFoundError, KeyError):
                n_items = 0
        
        if max_items is None:
            max_items = n_items
        else:
            max_items = min(max_items, n_items)
        
        if max_items > warn_threshold:
            warnings.warn(f"Loading {max_items} items into memory. This may consume a significant amount of RAM.", 
                          UserWarning)
        
        ase_list = []
        for i, config in enumerate(self.iter_items()):
            if i >= max_items:
                break
            ase_list.append(self._config_to_ase(config))
        
        return ase_list

    def get_txt(self, write_params=False, key_prefix='', max_items=None, warn_threshold=1000):
        return self.get_xyz(write_params, key_prefix, max_items, warn_threshold)

    def get_xyz(self, write_params=False, key_prefix='', max_items=None, warn_threshold=1000):
        """
        Convert dataset to XYZ format text.
        
        :param write_params: Whether to include additional parameters in the output
        :param key_prefix: Prefix to add to property keys (energy, forces, stress)
        :param max_items: Optional limit to the number of items to process
        :param warn_threshold: Show warning if processing more than this many items
        :return: Text in XYZ format
        """
        n_items = self.base.attributes.get('dataset_size', 0)
        if n_items == 0:
            # Count the number of items in the HDF5 file
            try:
                with self.base.repository.open(self._list_key, 'rb') as hdf_file:
                    with h5py.File(hdf_file, 'r') as hdf:
                        n_items = len(hdf)
            except (FileNotFoundError, KeyError):
                n_items = 0
        
        if max_items is None:
            max_items = n_items
        else:
            max_items = min(max_items, n_items)
        
        if max_items > warn_threshold:
            warnings.warn(f"Processing {max_items} items. This may take some time and consume memory.", 
                          UserWarning)
        
        dataset_txt = ''
        if not key_prefix.endswith('_') and key_prefix != '':
            key_prefix += '_'
        
        exclude_params = ['cell', 'symbols', 'positions', 'pbc', 'forces', 'stress', 'energy', 
                          'dft_forces', 'dft_stress', 'dft_energy', 'md_forces', 'md_stress', 'md_energy']
        exclude_pattern = re.compile(r'pot_\d+_(energy|forces|stress)')
        
        for i, config in enumerate(self.iter_items()):
            if i >= max_items:
                break
            
            params = [key for key in config.keys() if key not in exclude_params and not exclude_pattern.match(key)]
            atm = Atoms(symbols=config['symbols'], positions=config['positions'], cell=config['cell'], pbc=config['pbc'])
            atm.pbc = [True, True, True]
            atm.info = {}
            
            if write_params:
                for key in params:
                    atm.info[key] = config[key]
                    
            if len(atm.get_chemical_symbols()) == 1:
                atm.info["config_type"] = "IsolatedAtom"
                
            if 'dft_stress' in config:
                s = convert_stress(config['dft_stress'])
                atm.info[f'{key_prefix}stress'] = f"{s[0][0]:.6f} {s[0][1]:.6f} {s[0][2]:.6f} {s[1][0]:.6f} {s[1][1]:.6f} {s[1][2]:.6f} {s[2][0]:.6f} {s[2][1]:.6f} {s[2][2]:.6f}"
            
            if 'dft_energy' in config:
                atm.info[f'{key_prefix}energy'] = config['dft_energy']
                
            if 'dft_forces' in config:
                atm.set_calculator(SinglePointCalculator(atm, forces=config['dft_forces']))
            
            with io.StringIO() as buf, redirect_stdout(buf):
                write('-', atm, format='extxyz', write_results=True, write_info=True)
                dataset_txt += buf.getvalue()
                
        if key_prefix != '':
            dataset_txt = dataset_txt.replace(':forces', f':{key_prefix}forces')
            
        return dataset_txt
    
        
    def get_unlabelled(self):
        """Return a PESData object with only unlabelled frames."""
        if self.base.attributes.get('num_labelled_frames') == 0:
            return self
        unlabelled_data = []
        for config in self.iter_items():
            if 'dft_forces' not in config.keys() or 'dft_energy' not in config.keys():
                unlabelled_data.append(config)
        return PESData(data=unlabelled_data)

    def get_labelled(self):
        """Return a PESData object with only labelled frames."""
        if self.base.attributes.get('num_unlabelled_frames') == 0:
            return self
        labelled_data = []
        for config in self.iter_items():
            if 'dft_forces' in config.keys() and 'dft_energy' in config.keys():
                labelled_data.append(config)
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

    def iter_ase(self, max_items=None):
        """
        Generator function to iterate through ASE Atoms objects without loading all into memory.
        
        :param max_items: Optional limit to the number of items to process
        :yield: ASE Atoms objects one at a time
        """
        if max_items is not None:
            counter = 0
            for config in self.iter_items():
                if counter >= max_items:
                    break
                yield self._config_to_ase(config)
                counter += 1
        else:
            for config in self.iter_items():
                yield self._config_to_ase(config)