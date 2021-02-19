import numpy as np
import pandas as pd


############################################################
#################### Defining Functions ####################
############################################################
def preprocess_data(csv_filename, fingerprint_length=1024):
	'''
	This function preprocess_data converts the data read in from
	a csv function with ligand information into a list of
	ligand class instances.

	Args:
		csv_filename (str): Name of csv file for reading in.
		fingerprint_length (int, default=1024): length of fingerprint. Defaults to 1024.

	Returns:
		ligand_list (list): Set of Ligand class instances.
	'''
	df = pd.read_csv(csv_filename)
	ligand_list = []
	for idx in range(len(df)):
		ligand_info = df.iloc[idx].values
		ligand_id, dock_score, smiles, on_bits = ligand_info
		ligand_list.append(Ligand(ligand_id, dock_score, smiles, on_bits, fingerprint_length))
	return ligand_list


def generate_fingerprint_array(ligand_list):
	'''
	This function converts a list of Ligand class instances to a numpy
	array the fingerprints.

	Args:
		ligand_list (list): List of Ligan class instances.

	Returns:
		fp_arr (array-like): Numpy array of full-bit fingerprints.
	'''
	fp_arr = [ligand.fingerprint for ligand in ligand_list]
	fp_arr = np.array(fp_arr)
	return fp_arr


def cluster_similarity(c1_labels, c2_labels, index='jaccard'):
	'''
	This function measures the similarity of cluster 1 and cluster 2
	based on the arr using either the 'jaccard' or 'rand' index.

	Args:
		arr (array-like): input array for clustering
		c1_labels (array-like): 1D array of cluster labels.
		c2_labels (array-like): 1D array of cluster labels.
		index (str, default='jaccard'): Index to use must be either 'jaccard'
			or 'rand'

	Returns:
		sim_index (float): Similarity index value.
	'''
	index = index.lower()
	assert index in ['jaccard', 'rand'], 'Index must be jaccard or rand'
	assert c1_labels.shape[0] == c2_labels.shape[0]
	f_00 = 0
	f_10 = 0
	f_01 = 0
	f_11 = 0
	for i in range(c1_labels.shape[0]):
		for j in range(c2_labels.shape[0])[i+1:]:
			c1 = c1_labels[i] == c1_labels[j]
			c2 = c2_labels[i] == c2_labels[j]
			if c1 == False and c2 == False:
				f_00 += 1
			elif c1 == True and c2 == False:
				f_10 += 1
			elif c1 == False and c2 == True:
				f_01 += 1
			elif c1 == True and c2 == True:
				f_11 += 1
	if index == 'jaccard':
		sim_index = f_11/(f_11 + f_10 + f_01)
		return sim_index
	elif index == 'rand':
		sim_index = (f_11 + f_00)/(f_00 + f_01 + f_10 + f_11)
		return sim_index


############################################################
##################### Defining Classes #####################
############################################################
class Ligand:
	'''
	This class describes a small molecule, by storing info about its identity.

	Attributes:
		ligand_id (str): The ligand id.
		dock_score (float): The docking energy score.
		smiles (str): The smiles identifier string.
		fingerprint (array-like): ECFP Fingerprint of Molecule.

	'''
	def __init__(self, ligand_id, dock_score, smiles, on_bits, fp_length=1024):
		'''
		Args:
			ligand_id (str): The ligand id
			dock_score (float): The docking energy score.
			smiles (str): The smiles identifier string.
			on_bits (str): string containing on bits in fingerprint seperated
				by commas.
			fp_length (int, default=1024): length of fingerprint.
				Defaults to 1024.
		'''
		self.ligand_id = ligand_id
		self._fp_length = fp_length
		self.dock_score = float(dock_score)
		self.smiles = smiles
		self._on_bits = on_bits
		self.fingerprint = self._generate_fp_vec()

	def _generate_fp_vec(self):
		'''
		This function generates the array-like fingerprint from the sparse
		string representation.
		'''
		on_bits = [int(idx) for idx in self._on_bits.split(',')] # list of on bits
		fp = np.zeros(self._fp_length) # defining base fp
		fp[on_bits] = 1 # turning bits on
		return fp


class AglomerativeCluster:
	'''
	This class is used to define a cluster for use in Aglomerative Clustering.
	'''
	def __init__(self, ids):
		'''
		Args:
			ids (list): list of ids or row_labels from arr in cluster.
		'''
		if type(ids) != list:
			ids = [ids]
		self._ids = ids

	@property
	def ids(self):
		'''list: list of ids in the cluster that correspond to
			row names in the array being clustered.'''
		return self._ids

	@ids.setter
	def ids(self, value):
		# ensuring no repeats in the cluster
		assert len(np.unique(np.array(value))) == len(value)
		self._ids = value


class KMeansCluster:
	'''
	This class is used to define a cluster for use in PartitionClustering.
	'''
	def __init__(self, initial_cc):
		'''
		Args:
			initital_cc (arrray-like): Initial cluster center.
		'''
		self._cc = initial_cc
		self._data_inds = np.array([]).astype('int')
		self._distances = np.array([]).astype('float')

	@property
	def cc(self):
		'''
		array-like: 1D array of size [1, num_features] describing the location
			of this clusters cluster center.
		'''
		return self._cc

	@cc.setter
	def cc(self, value):
		if type(value) != np.ndarray:
			if type(value) == list:
				value = np.array(value)
			else:
				assert type(value) == np.ndarray, 'cc must be list or array-like'
		self._cc = value

	@property
	def data_inds(self):
		'''
		array_like: Array of size [1, num_points] conatining the indices of the
			data in the input arry that belong to this cluster.
		'''
		return self._data_inds

	@data_inds.setter
	def data_inds(self, value):
		if type(value) != np.ndarray:
			if type(value) == list:
				value = np.array(value)
			else:
				assert type(value) == np.ndarray, 'cluster_arr must be array-like'
		self._data_inds = value


class BaseMetric:
	'''
	This class acts as a base for all clustering algorithms and includes
	metric functions.
	'''
	def __init__(self, metric):
		'''
		Args:
			metric (str): metric to use for clustering when inhereting.
		'''
		metric = metric.lower()
		assert metric in ['tanimoto', 'manhattan', 'hamming', 'euclidean', 'chebyshev']
		metric = metric.lower()
		self._metric = metric

	def _tanimoto(self, a, b):
		'''
		Calculates the tanimoto distance.

		Args:
			a (array-like): first array.
			b (array-like): second array.

		Returns:
			distance (float): Calculated distance.
		'''
		set_a = set(list(a.nonzero()[0]))
		# set_a.update()
		set_b = set(list(b.nonzero()[0]))
		# set_b.update(list(b.nonzero()))
		intersect = len(set_a & set_b)
		union = len(set_a | set_b)
		distance = 1 - (intersect / union)
		return distance

	def _minowski(self, a, b, p=2):
		'''
		Calculates the p-norm of two vectors.

		Args:
			a (array-like): first array.
			b (array-like): second array.
			p (int): value norm to take.

		Returns:
			distance (float): Calculated distance.
		'''
		distance = float(np.sum(np.abs(a-b)**p)**(1/p))
		return distance

	def _euclidean(self, a, b):
		'''
		Calculates the euclidean distance (2-norm) of two vectors.

		Args:
			a (array-like): first array.
			b (array-like): second array.

		Returns:
			distance (float): Calculated distance.
		'''
		distance = self._minowski(a, b, p=2)
		return distance

	def _manhattan(self, a, b):
		'''
		Calculates the manhattan distance (1-norm) of two vectors.

		Args:
			a (array-like): first array.
			b (array-like): second array.

		Returns:
			distance (float): Calculated distance.
		'''
		distance = self._minowski(a, b, p=1)
		return distance

	def _chebyshev(self, a, b):
		'''
		Calculates the chebyshev (infinity-norm) of two vectors.

		Args:
			a (array-like): first array.
			b (array-like): second array.

		Returns:
			distance (float): Calculated distance.
		'''
		distance = float(np.max(np.abs(a-b)))
		return distance

	def _hamming(self, a, b):
		'''
		Calculates the hamming distance of two vectors.

		Args:
			a (array-like): first array.
			b (array-like): second array.

		Returns:
			distance (float): Calculated distance.
		'''
		distance = float(np.count_nonzero(np.equals(a, b)))
		return distance

	def _metric_func(self, a, b):
		'''
		Calculates the metric function defined at instance initialization.

		Args:
			a (array-like): first array.
			b (array-like): second array.
			metric

		Returns:

		'''
		metric = self._metric
		if metric == 'tanimoto':
			return self._tanimoto(a, b)
		elif metric == 'hamming':
			return self._hamming(a, b)
		elif metric == 'manhattan':
			return self._manhattan(a, b)
		elif metric == 'euclidean':
			return self._euclidean(a, b)
		elif metric == 'chebyshev':
			return self._chebyshev(a,b)

	def generate_dist_arr(self, arr):
		'''
		This function generates the initial distance matrix for use
		in the HierarchicalClustering scheme.

		Args:
			arr (array-like): Input array of dimensions elements x features
				for clustering.

		Returns:
			dist_arr (array-like): Initial distance array used in clustering.
		'''
		# initailizing distance array for updatin throughout clustering
		dist_arr = np.zeros((arr.shape[0], arr.shape[0]))
		# calculating all distances between individual elements fow upper triangle
		for i in np.arange(arr.shape[0]):
			for j in np.arange(arr.shape[0])[i+1:]:
				distance = self._metric_func(arr[i, :], arr[j,:])
				dist_arr[i, j] = distance
		# making the matrix symmetric
		i_lower = np.tril_indices(dist_arr.shape[0], 0)
		dist_arr[i_lower] = dist_arr.T[i_lower]
		# filling in the diaganol with infs
		np.fill_diagonal(dist_arr, np.inf)
		return dist_arr

	def cluster_quality(self, arr, cluster_labels):
		'''
		This function calculates the silhouette quality metric after clustering
		has been performed.

		Args:
			arr (array-like): Array that has been clustered.
			cluster_labels (array-like): Array of cluster labels from clustering.

		Returns:
			quality_score (float): The calculated silhouette quality metric.
		'''
		# calculating distance matrix
		dist_arr = self.generate_dist_arr(arr)
		cluster_list = [] # list that contains indices of points in dist_arr
		for c_num in np.unique(cluster_labels):
			# establishing indices to all in-cluster points
			c_inds = (cluster_labels == c_num).nonzero()[0]
			cluster_list.append(c_inds)
		s_vec = []
		for base_c_idx in range(len(cluster_list)):
			base_c = cluster_list[base_c_idx]
			other_cs = [x for i,x in enumerate(cluster_list) if i != base_c_idx]
			# calculating inter cluster distances (a vals)
			a_vec = []
			for i in base_c:
				dists = []
				for j in [j for j in base_c[base_c != i]]:
					dists.append(dist_arr[i, j])
				if len(dists) == 0:
					dists = [0]
				a_vec.append(np.mean(np.array(dists)))
			a_vec = np.array(a_vec)
			# calculating intra cluster distances
			exp_b_vec = []
			# iterating through each other cluster and calculating distances (b vals)
			for other_c in other_cs:
				temp_b_vec = []
				for i in base_c:
					si_list = []
					for j in other_c:
						si_list.append(dist_arr[i, j])
					temp_b_vec.append(np.mean(np.array(si_list)))
				exp_b_vec.append(np.array(temp_b_vec))
			exp_b_vec = np.array(exp_b_vec)
			b_vec = np.min(np.array(exp_b_vec), axis=0)
			assert np.count_nonzero(np.max(np.array([a_vec, b_vec]), axis=0) == 0) == 0, 'Repeated data point clustered to different clusters, too many number of clusters chosen.'
			s_i = (b_vec-a_vec)/np.max(np.array([a_vec, b_vec]), axis=0)
			s_i = np.expand_dims(s_i, axis=0)
			s_vec.append(s_i)
		# calculating silhouette coefficient
		s_vec = np.hstack(s_vec)
		quality_score = np.mean(s_vec)
		return quality_score



class HierarchicalClustering(BaseMetric):
	'''
	This class implements Hierarchical Clustering and inherets from
	the BaseMetric class. This class is used for generating clusters
	using alglomerative clustering.
	'''
	def __init__(self, num_clusters=1, metric='tanimoto', linkage='ward'):
		'''
		Args:
			num_clusters (int, default=1): Number of clusters calculate.
				Defaults to 1.
			metric (str, default='tanimoto'): Metric function used for clustering.
				Defaults to Tanimoto (Jaccard). Must be one of
				'tanimoto', 'manhattan', 'hamming', 'euclidean', 'chebyshev'.
			linkage (str, default='ward'): Linkage used for clustering.
				Defaults to Ward linkage. Must be one of 'single', 'upgma',
				'wpgma', 'complete', 'ward'.
		'''
		super().__init__(metric)
		linkage = linkage.lower()
		self._linkage = linkage
		assert linkage in ['single', 'upgma', 'wpgma', 'complete', 'ward']
		self._num_clusters = num_clusters

	def _single_linkage(self, c1_idx, c2_idx, temp_idx, dist_arr):
		'''
		Calculates single linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		c1_dist = dist_arr[c1_idx, temp_idx]
		c2_dist = dist_arr[c1_idx, temp_idx]
		distance = min([c1_dist, c2_dist])
		return distance

	def _complete_linkage(self, c1_idx, c2_idx, temp_idx, dist_arr):
		'''
		Calculates complete linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		c1_dist = dist_arr[c1_idx, temp_idx]
		c2_dist = dist_arr[c1_idx, temp_idx]
		distance = max([c1_dist, c2_dist])
		return distance

	def _weighted_average_linkage(self, c1_idx, c2_idx, temp_idx, dist_arr, arr_labels):
		'''
		Calculates WPGMA linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.
			arr_labels (list): List of AglomerativeCluster instances corresponding
				to each column/row in the distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		c1_dist = dist_arr[c1_idx, temp_idx]
		c2_dist = dist_arr[c1_idx, temp_idx]
		distance = (c1_dist + c2_dist)/2
		return distance

	def _unweighted_average_linkage(self, c1_idx, c2_idx, temp_idx, dist_arr, arr_labels):
		'''
		Calculates UPGMA linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.
			arr_labels (list): List of AglomerativeCluster instances corresponding
				to each column/row in the distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		c1_dist = dist_arr[c1_idx, temp_idx]
		c2_dist = dist_arr[c1_idx, temp_idx]
		prop_c1_dist = c1_dist*len(arr_labels[c1_idx].ids)
		prop_c2_dist = c2_dist*len(arr_labels[c2_idx].ids)
		c1_c2_size = len(arr_labels[c1_idx].ids) + len(arr_labels[c2_idx].ids)
		distance = (prop_c1_dist + prop_c2_dist)/c1_c2_size
		return distance

	def _ward_linkage(self, c1_idx, c2_idx, temp_idx, dist_arr, arr_labels):
		'''
		Calculates Ward linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.
			arr_labels (list): List of AglomerativeCluster instances corresponding
				to each column/row in the distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		c1_dist = dist_arr[c1_idx, temp_idx]
		c2_dist = dist_arr[c1_idx, temp_idx]
		n_c1 = len(arr_labels[c1_idx].ids)
		n_c2 = len(arr_labels[c2_idx].ids)
		n_ct = len(arr_labels[temp_idx].ids)
		n_tot = n_c1 + n_c2 + n_ct
		alpha_c1 = (n_c1 + n_ct)/n_tot
		alpha_c2 = (n_c2 + n_ct)/n_tot
		beta = -(n_ct)/n_tot
		distance = alpha_c1*c1_dist + alpha_c2*c2_dist + beta*dist_arr[c1_idx, c2_idx]
		return distance

	def _linkage_func(self, c1_idx, c2_idx, temp_idx, dist_arr, arr_labels):
		'''
		Calculates the linkage from distance array of d(C_i U C_j, C_k)
		where C_i and C_j are being combined and the distance is being
		calculated to the independent cluster C_k by calling the linkage
		function defined during the initialization of the HierarchicalClustering
		class instance.

		Args:
			c1_idx (int): Index of cluster 1 in distance array ie C_i.
			c2_idx (int): Index of cluster 2 in distance array ie C_j.
			temp_idx (int): Index of cluster for distance calc ie C_k.
			dist_arr (array-like): Distance matrix.
			arr_labels (list): List of AglomerativeCluster instances corresponding
				to each column/row in the distance matrix.

		Returns:
			distance (float): Distance to new cluster C_i U C_j from C_k.
		'''
		linkage = self._linkage
		if linkage == 'ward':
			return self._ward_linkage(c1_idx, c2_idx, temp_idx, dist_arr, arr_labels)
		elif linkage == 'upgma':
			return self._unweighted_average_linkage(c1_idx, c2_idx, temp_idx, dist_arr, arr_labels)
		elif linkage == 'wpgma':
			return self._weighted_average_linkage(c1_idx, c2_idx, temp_idx, dist_arr, arr_labels)
		elif linkage == 'single':
			return self._single_linkage(c1_idx, c2_idx, temp_idx, dist_arr)
		elif linkage == 'complete':
			return self._complete_linkage(c1_idx, c2_idx, temp_idx, dist_arr)

	def cluster(self, arr):
		'''
		This function runs the Hierarchical Clustering algorithm
		and returns labels for each cluster.

		Args:
			arr (array-like): array of dimensions elements x features.

		Returns:
			cluster_labels (array_like): Labels corresponding to each
				input row in arr.
		'''
		# defining distance array columns/row labels
		row_labels = range(arr.shape[0])
		arr_labels = [AglomerativeCluster(x) for x in row_labels]
		# generating distance matrix
		dist_arr = self.generate_dist_arr(arr)
		# Defining conditional for stopping the clustering
		num_clusters = self._num_clusters
		assert num_clusters >= 1, 'Method requires at least one cluster'
		# Clustering
		while dist_arr.shape[0] > num_clusters:
			# finding minimum distance in distance matrix
			c1_idx, c2_idx = np.unravel_index(np.argmin(dist_arr, axis=None), dist_arr.shape)
			# merging clusters
			c1 = arr_labels[c1_idx]
			c2 = arr_labels[c2_idx]
			new_c = AglomerativeCluster(c1.ids + c2.ids)
			# Updating distance matrix
			temp_idx_list = list(np.arange(dist_arr.shape[1]))
			temp_idx_list.pop(c1_idx)
			temp_idx_list.pop(c2_idx-1)
			assert c1_idx not in temp_idx_list
			assert c2_idx not in temp_idx_list
			# new column for distance matrix
			new_col = np.zeros((len(temp_idx_list), 1))
			for i, temp_idx in enumerate(temp_idx_list):
				new_col[i] = self._linkage_func(c1_idx, c2_idx, temp_idx, dist_arr, arr_labels)
			# removing old clusters from distance matrix
			dist_arr = np.delete(dist_arr, [c1_idx, c2_idx], axis=0)
			dist_arr = np.delete(dist_arr, [c1_idx, c2_idx], axis=1)
			# Adding new column and row
			dist_arr = np.hstack([dist_arr, new_col])
			new_row = np.append(new_col.T, [[np.inf]], 1)
			dist_arr = np.vstack([dist_arr, new_row])
			# updating cluster label list
			arr_labels.append(new_c)
			arr_labels.pop(c1_idx)
			arr_labels.pop(c2_idx-1)
		# generating vector of labels
		cluster_labels = np.zeros((arr.shape[0]))
		for idx in range(len(arr_labels)):
			cluster_ids = arr_labels[idx].ids
			cluster_labels[cluster_ids] = idx
		# returning cluster labels
		return cluster_labels



class PartitionClustering(BaseMetric):
	'''
	This class implements Partition Clustering and inherets from
	the BaseMetric class. This class is used for generating clusters
	using the K-means algorithm.

	Attributes:
		initial_cluster_centers (array-like): Array of initial cluster centers.
	'''
	def __init__(self, num_clusters=1, metric='tanimoto', max_iter=1000):
		'''
		Args:
			num_clusters (int, default=1): Number of clusters calculate.
				Defaults to 1.
			metric (str, default='tanimoto'): Metric function used for clustering.
				Defaults to Tanimoto (Jaccard). Must be one of
				'tanimoto', 'manhattan', 'hamming', 'euclidean', 'chebyshev'.
			max_iter (int, default=1000): The maximum number of times to iterate
				the k_means algorithm if it does not converge first.
		'''
		super().__init__(metric)
		self._num_clusters = num_clusters
		self._max_iter = max_iter
		self._final_clusters = None
		self.initial_cluster_centers = None
		# seeding random processes in this class
		np.random.seed(14)

	@property
	def final_clusters_(self):
		'''
		list: final_clusters_ is a list of KMeansCluster class instances
		'''
		return self._final_clusters

	def _initialize_cluster_centers(self, arr):
		'''
		Initializes cluster center using the kmeans ++ algorithm.

		Args:
			arr (array-like): Input array of dimensions elements by features.

		Returns:
			initial_ccs (array-like): Array of inititial cluster centers.
		'''
		num_clusters = self._num_clusters
		initial_ccs = []
		possible_inds = list(np.arange(arr.shape[0]))
		p = np.ones(len(possible_inds))/len(possible_inds)
		choices = []
		for _ in range(num_clusters):
			# randomly choose point based on p
			rand_choice = np.random.choice(possible_inds, 1, p=p).astype('int')[0]
			choices.append(rand_choice)
			# remove random choice from possible indices
			possible_inds.remove(rand_choice)
			# add cluster center
			initial_ccs.append(arr[rand_choice, :])
			# updating probabilities p of choosing new cluster center
			p_dists = []
			for point_idx in possible_inds:
				dists = []
				for cc_idx in choices:
					dists.append(self._metric_func(arr[point_idx, :], arr[cc_idx, :]))
				p_dists.append(min(dists))
			p_dist = np.array(p_dists)
			p = p_dist / np.sum(p_dist)
		initial_ccs = np.array(initial_ccs)
		assert initial_ccs.shape[0] == num_clusters
		return initial_ccs

	def cluster(self, arr):
		'''
		This function runs the Partition Clustering K-means algorithm
		and returns labels for each cluster.

		Args:
			arr (array-like): array of dimensions elements x features.

		Returns:
			cluster_labels (array_like): Labels corresponding to each
				input row in arr.
		'''
		# generating inital cluster centers using KMeans++ algorithm
		initial_ccs = self._initialize_cluster_centers(arr)
		self.initial_cluster_centers = initial_ccs
		# Calculating distance matrix for use in finding the medoid of clusters
		dist_arr = self.generate_dist_arr(arr)
		# defining initial empty clusters
		clusters = [KMeansCluster(initial_ccs[x, :]) for x in range(initial_ccs.shape[0])]
		# generating first round of clusters
		for row_idx in range(arr.shape[0]):
			cluster_dist = []
			for cluster in clusters:
				cluster_dist.append(self._metric_func(arr[row_idx, :], cluster.cc))
			top_clust = np.argmin(cluster_dist)
			top_clust_dist = np.amin(cluster_dist)
			clusters[top_clust].data_inds = np.hstack((clusters[top_clust].data_inds, row_idx))
			clusters[top_clust]._distances = np.hstack((clusters[top_clust]._distances, top_clust_dist))
		# iterating until convergence or max_iter is reached
		for iteration in range(self._max_iter):
			updated_clusters = []
			# updating cluster centers
			for cluster in clusters:
				new_cc_average = np.mean(arr[cluster.data_inds, :], axis=0)
				# setting the medoid as the new cluster center
				medoid_idx = np.argmin(np.sum(dist_arr[cluster.data_inds, cluster.data_inds], axis=0))
				new_cc = arr[cluster.data_inds[medoid_idx], :]
				updated_clusters.append(KMeansCluster(new_cc))
			# calculating distance for each point to new cluster centers
			for row_idx in range(arr.shape[0]):
				cluster_dist = []
				for cluster in updated_clusters:
					cluster_dist.append(self._metric_func(arr[row_idx, :], cluster.cc))
				top_clust = np.argmin(cluster_dist)
				updated_clusters[top_clust].data_inds = np.hstack((updated_clusters[top_clust].data_inds, row_idx))

			# # handeling any empty clusters if there are any
			# # finding the futhest centroid from the centroid of the largest cluster
			# clust_sizes = np.array([c.data_inds.shape[0] for c in updated_clusters])
			# print([len(x.cc) for x in updated_clusters])
			# clust_sizes_ordered = np.argsort(np.argsort(clust_sizes))[(clust_sizes > 0)]
			# # calculating the number of clusters with 0
			# num_empty_clusters = np.sum(np.array(clust_sizes) == 0)
			# if num_empty_clusters > 0:
			# 	empty_clusters_inds = np.arange(len(updated_clusters))[clust_sizes == 0]
			# 	clust_sizes_ordered = list(clust_sizes_ordered)
			# 	# Generating new cluster center for all clusters with no points
			# 	largest_cluster = updated_clusters[clust_sizes_ordered.pop(np.argmax(np.array(clust_sizes_ordered)))].cc
			# 	dists = []
			# 	# findinng the furthest away cluster ceneter
			# 	for cluster_idx in clust_sizes_ordered:
			# 		dists.append(self._metric_func(updated_clusters[cluster_idx].cc, largest_cluster))
			# 	for ec_idx in empty_clusters_inds:
			# 		furthest_cluster = updated_clusters[clust_sizes_ordered.pop(np.argmax(np.array(dists)))]
			# 		print(furthest_cluster.cc.shape)
			# 		dists.pop(np.argmax(np.array(dists)))
			# 		pnt_dists = []
			# 		for point_idx in range(furthest_cluster.data_inds.shape[0]):
			# 			pnt_dists.append(self._metric_func(arr[furthest_cluster.data_inds[point_idx]], largest_cluster))
			# 		# 	print(np.array(pnt_dists))
			# 		# print('transition')
			# 		# print(np.array(pnt_dists))
			# 		new_cc = arr[furthest_cluster.data_inds[np.argmax(np.array(pnt_dists))], :]
			# 		updated_clusters[ec_idx].cc = new_cc
			# # print(num_empty_clusters, np.array(clust_sizes) == 0)

			# calculating difference between previous clusters and new clusters
			tot_diff = 0
			for c1, c2 in zip(clusters, updated_clusters):
				c1_set = set()
				c1_set.update(c1.data_inds)
				c2_set = set()
				c2_set.update(c2.data_inds)
				temp_diff = len(c2_set - c1_set)
				tot_diff += temp_diff
			# stopping iteration if the clusters converge
			if tot_diff == 0:
				clusters = updated_clusters
				print(f'Converged after {iteration} iterations.')
				break
			else:
				clusters = updated_clusters
				if iteration == self._max_iter - 1:
					print(f'Reached maximum ({self._max_iter}) number of iterations')
		self._final_clusters = clusters
		# generating cluster labels
		cluster_labels = np.zeros(arr.shape[0])
		for cluster_idx in range(len(clusters)):
			cluster_labels[clusters[cluster_idx].data_inds] = cluster_idx
		# returning cluster labels
		return cluster_labels
