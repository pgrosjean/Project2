# Project 2 - Clustering
## Author: Parker Grosjean
## BMI 203 UCSF

![BuildStatus](https://github.com/ucsf-bmi-203-2021/Project2/workflows/HW2/badge.svg?event=push)

## This Repository

This repo holds all of the information needed to run the Partitioning Clustering (KMeans) and Hierarchical Clustering algorithms for clustering molecular fingerprints.
This repository is set up with three main sections locations for the (1) Docs (2) clusters Module and (3) the pytest functionalities.

### Docs
To view the API docs for the clusters module and specifically the classes contained within align.algs please see the [API doc](https://github.com/pgrosjean/Project2/blob/main/docs/build/html/index.html) by running "open index.html" in the location of the file pointed to by the API doc link.

### Clusters Module
The [Clusters Module](https://github.com/pgrosjean/Project1/tree/main/align) holds the algs submodule that contains both the base class BaseMetric and the two Clustering Algorithm classes HierarchicalClustering and PartitionClustering that inheret from it. This submodule also holds a class to discribe Ligands for use in docking campaigns and structural clustering. See the [API doc](https://github.com/pgrosjean/Project2/blob/main/docs/build/html/index.html) for more information regarding the use of this module.

### Pytest Functionalities
All of the necessary files for testing can be found in [this folder](https://github.com/pgrosjean/Project2/tree/main/files_test) and the actual unit tests are implemented [here](https://github.com/pgrosjean/Project2/blob/main/test/test_clusters.py).

### Additional Information About Assignment
The data used in this analsis comes from [Smith and Smith, 2020](https://chemrxiv.org/articles/preprint/Repurposing_Therapeutics_for_the_Wuhan_Coronavirus_nCov-2019_Supercomputer-Based_Docking_to_the_Viral_S_Protein_and_Human_ACE2_Interface/11871402). In this study, the authors generated 6 Spike-Ace2 interface poses using MD simulations. They then docked ~10k small molecules against each protein conformation. Provided for you is the top (#1) pose for each ligand docked against one Spike-ACE2 interface conformation, as well as the corresponding SMILES string, AutoDock Vina score, and the “On” bits in the Extended Connectivity Fingerprint for that compound. These can all be found in ligand\_information.csv.


### Testing
Testing is as simple as running
```
python -m pytest
```
from the root directory of this project.
