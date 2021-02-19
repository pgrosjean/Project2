
import pytest
import numpy as np
import pandas as pd
from clusters import algs

def test_partitioning():
	# Since the partition clustering is not deterministic
	# I am checking that the number of clusters are correct
	# I am also checking that a basic small clustering test
	# example clusters correctly.
	test_array = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 1, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 1, 0],
				[1, 0, 1, 1, 0, 0, 0, 0, 0],
				[0, 1, 1, 0, 0, 0, 0, 0, 0],
				[1, 1, 0, 0, 0, 0, 0, 0, 0]])
	p_cluster = algs.PartitionClustering(num_clusters=2, max_iter=500)
	p_labels = list(p_cluster.cluster(test_array))
	# Checking it creates the proper number of clusters
	assert len(np.unique(p_labels)) == 2
	# Checking that id does correct clustering
	assert p_labels == [0, 1, 1, 1, 0, 0, 0] or p_labels == [1, 0, 0, 0, 1, 1, 1]

def test_hierarchical():
	# Since this clustering is deterministic then I will
	test_array = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 1, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 1, 0],
				[1, 0, 1, 1, 0, 0, 0, 0, 0],
				[0, 1, 1, 0, 0, 0, 0, 0, 0],
				[1, 1, 0, 0, 0, 0, 0, 0, 0]])
	h_cluster = algs.HierarchicalClustering(num_clusters=2, linkage='ward')
	h_labels = list(h_cluster.cluster(test_array))
	# Checking it creates the proper number of clusters
	assert len(np.unique(h_labels)) == 2
	# Checking that id does correct clustering
	assert h_labels == [0, 1, 1, 1, 0, 0, 0] or h_labels == [1, 0, 0, 0, 1, 1, 1]

def test_quality_metric():
	# Testing quality metric with pre-computed silhouette score
	test_array = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 1, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 0, 0],
				[0, 0, 0, 0, 1, 1, 0, 1, 0],
				[1, 0, 1, 1, 0, 0, 0, 0, 0],
				[0, 1, 1, 0, 0, 0, 0, 0, 0],
				[1, 1, 0, 0, 0, 0, 0, 0, 0]])
	h_cluster = algs.HierarchicalClustering(num_clusters=2, linkage='ward')
	h_labels = list(h_cluster.cluster(test_array))
	cq = h_cluster.cluster_quality(test_array, h_labels)
	assert np.around(cq, decimals=2) == 0.51

def test_similarity_index():
	# generating identical clusters (index of 1)
	cluster_1 = np.array([0, 0, 1, 1])
	cluster_2 = np.array([0, 0, 1, 1])
	# generating completely different clusters
	cluster_3 = np.array([0, 1, 1, 0])
	cluster_4 = np.array([1, 1, 1, 0])
	assert algs.cluster_similarity(cluster_1, cluster_2, index='jaccard') == 1
	assert algs.cluster_similarity(cluster_1, cluster_3, index='jaccard') == 0
	assert algs.cluster_similarity(cluster_4, cluster_3, index='jaccard') == 0.25

def test_data_preprocessing():
	# reading in test csv
	ligand_list = algs.preprocess_data('./files_test/test_csv.csv')
	ligand_1 = ligand_list[0]
	# asserting correct class attributes
	assert ligand_1.ligand_id == 1
	assert ligand_1.dock_score == -1.7
	assert ligand_1.smiles == 'O=N'
	assert ligand_1._on_bits == '53,623,650'
	ligand_2 = ligand_list[1]
	assert ligand_2.ligand_id == 2008
	assert ligand_2.dock_score == -2.9
	assert ligand_2.smiles == 'O=C(OCCC[N+@@H]1[C@@H](C)CCCC1)c1ccccc1'
	assert ligand_2._on_bits == '91,140,147,457,807,844'

def test_ligand_class():
	df = pd.read_csv('./files_test/test_csv.csv')
	ligand_id, dock_score, smiles, on_bits = df.iloc[0].values
	ligand_1 = algs.Ligand(ligand_id, dock_score, smiles, on_bits)
	assert np.count_nonzero(~np.equal(ligand_1.fingerprint, ligand_1._generate_fp_vec())) == 0
	assert ligand_1.ligand_id == 1
	assert ligand_1.dock_score == -1.7
	assert ligand_1.smiles == 'O=N'
	assert ligand_1._on_bits == '53,623,650'

def test_build_distance_matrix():
	dist_arr = np.around(np.array([[np.inf, 1.0, 0.33333333],
	[1.0, np.inf, 0.66666667],
	[0.33333333, 0.66666667, np.inf]]), decimals=2)
	print(dist_arr)
	# calculating distance arrat from arr
	arr = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
	bm = algs.BaseMetric(metric='Tanimoto')
	da = np.around(bm.generate_dist_arr(arr), decimals=2)
	print(da)
	assert np.count_nonzero(~np.equal(dist_arr, da)) == 0
