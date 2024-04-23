import pandas as pd
import matplotlib.pyplot as plt

# Assuming the file is space-separated; adjust the delimiter as needed
file_path = './00-System.ATOMS'

# Load the data
data = pd.read_csv(file_path, delim_whitespace=True, header=None)

# Extract the columns for carbon and hydrogen atoms
carbon_atoms = data[1]
hydrogen_atoms = data[2]

# Calculate summary statistics
carbon_summary = carbon_atoms.describe()
hydrogen_summary = hydrogen_atoms.describe()

# Print summary statistics
print("Carbon Atoms Summary:")
print(carbon_summary)
print("\nHydrogen Atoms Summary:")
print(hydrogen_summary)

# Plot the normalized distributions with a bin size of 1 atom, including mean lines
plt.figure(figsize=(14, 7))

# Carbon atoms distribution
plt.subplot(1, 2, 1)
n, bins, patches = plt.hist(carbon_atoms, bins=range(int(carbon_atoms.min()), int(carbon_atoms.max()) + 2), 
                            color='blue', alpha=0.7, density=True)
plt.axvline(carbon_atoms.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(carbon_atoms.mean(), max(n) * 0.95, ' Mean: {:.2f}'.format(carbon_atoms.mean()), 
         color='black', ha='center')
plt.title('Normalized Distribution of Carbon Atoms')
plt.xlabel('Number of Carbon Atoms')
plt.ylabel('Normalized Frequency')

# Hydrogen atoms distribution
plt.subplot(1, 2, 2)
n, bins, patches = plt.hist(hydrogen_atoms, bins=range(int(hydrogen_atoms.min()), int(hydrogen_atoms.max()) + 2), 
                            color='green', alpha=0.7, density=True)
plt.axvline(hydrogen_atoms.mean(), color='k', linestyle='dashed', linewidth=1)
plt.text(hydrogen_atoms.mean(), max(n) * 0.95, ' Mean: {:.2f}'.format(hydrogen_atoms.mean()), 
         color='black', ha='center')
plt.title('Normalized Distribution of Hydrogen Atoms')
plt.xlabel('Number of Hydrogen Atoms')
plt.ylabel('Normalized Frequency')

plt.tight_layout()
plt.show()
