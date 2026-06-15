"""Plot parity plots and error calibration plots for trained potentials."""

# %%
from aiida import load_profile
from aiida.orm import load_node

from aiida_trains_pot.utils.plotting import plot_error_calibration, plot_parity

load_profile()

# %%

plot_parity(load_node(1069011))  # tutti i cicli completati
# plot_parity(load_node(1069011), cycles=[1, 3])     # solo cicli 1 e 3
# plot_parity(load_node(1069011), show=False, save_path="parity.png")

# %%
plot_error_calibration(load_node(1069011))  # tutti i cicli
# plot_error_calibration(load_node(1069011), cycles=[2])  # solo ciclo 2
# plot_error_calibration(load_node(1069011), sets=["training", "test"])
