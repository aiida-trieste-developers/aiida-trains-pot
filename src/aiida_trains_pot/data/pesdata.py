from aiida.orm import List
import tempfile
import os
import numpy as np

class PESData(List):
    
    @property
    def _list_key(self):
        """Generate a unique filename for the list based on the node's UUID."""
        return f"psedata_{self.uuid}.npz"  # Unique filename with the node's UUID

    def get_list(self):
        """Return the contents of this node as a list."""
        try:            
            # Open the file and load its contents
            with self.base.repository.as_path(self._list_key) as f:
                data = np.load(f, allow_pickle=True)
                return [item for _, val in data.items() for item in (val.tolist() if isinstance(val, np.ndarray) else val)]

                #return [val.tolist() for _, val in data.items()]
            
                #return data

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