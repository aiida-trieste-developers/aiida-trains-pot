# AiiDA-TrainsPot

Welcome to the AiiDA-TrainsPot, the AiiDA workflow that Trains a Potential for you.

## Remote machine requirements

AiiDA-TrainsPot requires to have installed in the remote machine:

- Quantum ESPRESSO (at least pw.x executable) - [quantum-espresso.org](https://www.quantum-espresso.org/)
- MACE and PyYAML (better if inside a python environment) - [github.com/ACEsuit/mace](https://github.com/ACEsuit/mace), [github.com/yaml/pyyaml](https://github.com/yaml/pyyaml)
- LAMMPS with MACE extension - [mace-docs.readthedocs.io/en/latest/guide/lammps.html](https://mace-docs.readthedocs.io/en/latest/guide/lammps.html)

## Installation

1. Clone and install aiida-trains-pot repository

```
git clone git@github.com:aiida-trieste-developers/aiida-trains-pot.git
cd aiida-trains-pot
pip install .
```

2. Clone and install aiida-lammps (last release of aiida-lammps was not compatible with MACE)

```
git clone git@github.com:aiidaplugins/aiida-lammps.git
cd aiida-lammps
pip install .
```

3. Install codes for Quantum ESPRESSO, MACE (pre-process, train and post-process), LAMMPS. Examples of configuration yaml file can be found in examples/setup_codes.

1. Install `PortableCode` for committee evalution:

```
portable_codes_installation
```

If needed specify in in prepend command the activation command for the python environment where MACE was installed

## Contributing

We welcome contributions from everyone. Before you start contributing, please make sure you have read and understood our Contributor License Agreement (CLA). By contributing to this project, you agree to the terms and conditions outlined in our [CLA](CLA.md).

Please follow our [contributing guidelines](CONTRIBUTING.md) to get started.

## How to Cite

If you use this plugin in your research, please cite:

**D. Bidoggia, N. Manko, M. Peressi and A. Marrazzo,**  *Automated training of neural-network interatomic potentials*  [_arXiv:2509.11703_ (physics.comp-ph), 2025](https://doi.org/10.48550/arXiv.2509.11703)

## Acknowledgments

This project was supported by:

- **[Università degli Studi di Trieste](https://portale.units.it/en)**
- **[Scuola Internazionale Superiore di Studi Avanzati](https://www.sissa.it/it)**
- **[Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing](https://www.supercomputing-icsc.it/en/icsc-home/)**
- **[EU Centre of Excellence "MaX – Materials Design at the Exascale"](https://www.max-centre.eu/)**

<p align="center">
  <img src="docs/Università degli Studi di Trieste Logo.jpeg" alt="Università degli Studi di Trieste Logo" height="110" style="filter: invert(1);"/>
  <img src="docs/sissa-logo.png" alt="SISSA Logo" width="100"/>
  <img src="docs/MAX.png" alt="MaX Centre Logo" height="110"/>
  <img src="docs/icsc-logo.png" alt="Centro Nazionale di Ricerca Logo" height="110"/>
</p>
