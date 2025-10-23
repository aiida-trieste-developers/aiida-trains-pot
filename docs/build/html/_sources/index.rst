
###############
AiiDA-TrainsPot
###############

Welcome to AiiDA-TrainsPot, the AiiDA_ workflow that Trains a Potential for you.

.. rst-class:: center

    |aiida_logo| |qe_logo| |lammps_logo|  |MACE_logo|

.. |aiida_logo| image:: images/AiiDA_transparent_logo.png
    :width: 25%

.. |qe_logo| image:: images/qe_logo.jpg
    :width: 25%

.. |lammps_logo| image:: images/Lammps-logo.png
    :width: 25%

.. |MACE_logo| image:: images/MACE_logo.png
    :width: 10%

Remote Machine Requirements
===========================

AiiDA-TrainsPot requires the following software installed on the remote machine:

- |Quantum ESPRESSO|_ (at least `pw.x` executable)
- MACE_ and PyYAML_ (preferably within a Python environment)
- LAMMPS_ with MACE extension


Installation
============

1. To clone and install the aiida-trains-pot repository:

.. code-block:: bash

   git clone git@github.com:aiida-trieste-developers/aiida-trains-pot.git
   cd aiida-trains-pot
   pip install -e .

Note that this command will also install the ``aiida-core``, ``aiida-quantumespresso``, ``aiida-lammps`` packages as its dependencies.
For more information on how to install AiiDA and the required services in different environments, we refer to the |aiida-core|_ documentation.

2. To clone and install aiida-lammps (the last release of aiida-lammps was not compatible with MACE):

.. code-block:: bash

   git clone git@github.com:aiidaplugins/aiida-lammps.git
   cd aiida-lammps
   pip install .

3. Install codes for Quantum ESPRESSO, MACE (pre-process, train, and post-process), and LAMMPS. Examples of configuration YAML files can be found in `examples/setup_codes`.

4. Install `PortableCode` for committee evaluation:

.. code-block:: bash

   portable_codes_installation

If needed, specify in the prepend command the activation command for the Python environment where MACE was installed.

Contributing
=============

We welcome contributions from everyone. Before you start contributing, please make sure you have read and understood our Contributor License Agreement (CLA). By contributing to this project, you agree to the terms and conditions outlined in our CLA.md_.

Please follow our CONTRIBUTING.md_ to get started.

Contents
========

.. toctree::
   :maxdepth: 2

   user_guide/index
   module_guide/index

Acknowledgments
===============

This project was supported by:

- |Università degli Studi di Trieste|_.
- |Scuola Internazionale Superiore di Studi Avanzati|_.
- |Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing|_.
- |EU Centre of Excellence "MaX – Materials Design at the Exascale"|_

.. raw:: html

   <p align="center">
     <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTgTBDFRADTwpIJqho2NDfWrdCgIMTxFnlHBA&s" alt="Università degli Studi di Trieste Logo" width="250" style="filter: invert(1);"/>
     <img src="https://www.sissa.it/themes/custom/sissa/images/logo-type.svg" alt="SISSA Logo" width="100"/>
     <img src="https://www.max-centre.eu/sites/default/files/styles/news_responsive/public/MaX_900x600.jpg" alt="MaX Centre Logo" width="250"/>
     <img src="https://www.supercomputing-icsc.it/wp-content/uploads/2022/10/logoxweb.svg" alt="Centro Nazionale di Ricerca Logo" width="250"/>
   </p>



.. |aiida-core| replace:: ``aiida-core``
.. _aiida-core: https://aiida.readthedocs.io/projects/aiida-core/en/latest/intro/get_started.html

.. |aiida-quantumespresso documentation| replace:: ``aiida-quantumespresso`` documentation
.. _aiida-quantumespresso documentation: https://aiida.readthedocs.io/projects/aiida-quantumespresso/en/latest/intro/get_started.html


.. _AiiDA Quantum ESPRESSO tutorial: https://aiida-tutorials.readthedocs.io/en/tutorial-qe-short/

.. _AiiDA: http://aiida.net
.. |Quantum ESPRESSO| replace:: Quantum ESPRESSO
.. _Quantum ESPRESSO: https://www.quantum-espresso.org
.. _MACE: https://github.com/ACEsuit/mace
.. _PyYAML: https://github.com/yaml/pyyaml
.. _LAMMPS: https://mace-docs.readthedocs.io/en/latest/guide/lammps.html
.. _CLA.md: https://github.com/aiida-trieste-developers/aiida-trains-pot/blob/main/CLA.md
.. _CONTRIBUTING.md: https://github.com/aiida-trieste-developers/aiida-trains-pot/blob/main/CONTRIBUTING.md
.. |Università degli Studi di Trieste| replace:: Università degli Studi di Trieste
.. _Università degli Studi di Trieste: https://portale.units.it/en
.. |Scuola Internazionale Superiore di Studi Avanzati| replace:: Scuola Internazionale Superiore di Studi Avanzati
.. _Scuola Internazionale Superiore di Studi Avanzati: https://www.sissa.it/it
.. |Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing| replace:: Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing
.. _Centro Nazionale di Ricerca HPC, Big Data e Quantum Computing: https://www.supercomputing-icsc.it/en/icsc-home/
.. |EU Centre of Excellence "MaX – Materials Design at the Exascale"| replace:: EU Centre of Excellence "MaX – Materials Design at the Exascale"
.. _EU Centre of Excellence "MaX – Materials Design at the Exascale": https://www.max-centre.eu/
.. _wannier90: http://www.wannier.org
.. _Quantum Mobile: https://quantum-mobile.readthedocs.io/en/latest/index.html

.. |AiiDA main paper| replace:: *AiiDA 1.0, a scalable computational infrastructure for automated reproducible workflows and data provenance*
.. _AiiDA main paper: https://doi.org/10.1038/s41597-020-00638-4

.. |AiiDA engine paper| replace:: *Workflows in AiiDA: Engineering a high-throughput, event-based engine for robust and modular computational workflows*
.. _AiiDA engine paper: https://doi.org/10.1016/j.commatsci.2020.110086

.. _NCCR MARVEL: http://nccr-marvel.ch/
.. _MaX – Materials Design at the Exascale: http://www.max-centre.eu/
.. _`swissuniversities P-5 project "Materials Cloud"`: https://www.materialscloud.org/swissuniversities
