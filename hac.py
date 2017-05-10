from sys import argv
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
import math
import heapq
import scipy.spatial.distance
from scipy.linalg import norm
#from sklearn.metrics.pairwise import cosine_similarity

f = open(argv[1])

data = f.readlines()
docs, words, w = data[0:3]
row = int(words) + 1
col = int(docs) + 1
del data[0:3]
df = {}

# get df
for d in data:
	d = d.split(" ")
	if d[1] in df:
		df[d[1]] += 1
	else:
		df[d[1]] = 1

# compute tf * idf, idf = log((N+1)/(df+1))
val = []
for d in data:
	d = d.split(" ")
	d[2] = float(d[2]) * math.log((float(docs) + 1) / float(df[d[1]] + 1), 2) 
	val.append(d)

def computeLength(list):
	power = 0
	count = 0
	for e in list:
		if e != 0:
			power += e ** 2
	return math.sqrt(float(power))

def computeCentroid(indices):
	centroid = np.zeros(len(mat[:,0]))
	if type(indices) is list:
		size = len(indices)
		for i in indices:
			centroid = np.add(centroid, mat[:,i])
		return centroid/size
	else:
		return mat[:,indices]

mat = np.zeros((row, col))
for v in val:
	mat[int(v[1]), int(v[0])] = v[2]

# Euclidean length
elength = []
for i in range(len(mat[0])):
	elength.append(computeLength(mat[:,i]))

# unit vector
for i in range(len(mat[0])):
	#leni = computeLength(mat[:,i])
	for j in range(len(mat[:,0])):
		if mat[j,i] != 0:
			mat[j,i] = mat[j,i] / elength[i]

#sp = sparse.csc_matrix(mat)

def cosine(col1, col2):
	#product = 0
	#for i in range(len(col1)):
	#	if col1[i] != 0 and col2[i] != 0:
	#		product += col1[i] * col2[i]
	product = np.sum(np.multiply(col1, col2))

	return 1 - product / (scipy.linalg.norm(col1) * scipy.linalg.norm(col2))



# compute cosine similarity
dist = scipy.spatial.distance.pdist(mat.T, 'cosine')
res = scipy.spatial.distance.squareform(dist)

def addNewDist(indices):
	cen = computeCentroid(indices)
	for cluster in clusters:
		if cluster != indices:
			l = []
			cluster_cen = centroids[str(cluster)]
			l.append(cosine(cen, cluster_cen))
			l.append([cluster, indices])
			heapq.heappush(h, l)

def isvaild(indices):
	for index in indices:
		if index not in clusters:
			return False
			break
	return True


def refreshCluster(list):
	new_cluster = []
	if len(list) == 1:
		for l in list[0]:
			new_cluster.append(l)
			clusters.remove(l)
		clusters.append(new_cluster)
	else:
		for li in list:
			if type(li) is int:
				new_cluster.append(li)
				clusters.remove(li)
			else:
				for l in li:
					new_cluster.append(l)
				clusters.remove(li)
		clusters.append(new_cluster)
	centroids[str(new_cluster)] = computeCentroid(new_cluster)
	return clusters

def merge(list):
	new_cluster = []
	if len(list) == 1:
		for l in list[0]:
			new_cluster.append(l)
	else:
		for li in list:
			if type(li) is int:
				new_cluster.append(li)
			else:
				for l in li:
					new_cluster.append(l)
	return new_cluster


centroids = {}


# heapify
h = []
c = 0
for i in range(1, len(res)):
	for j in range(i + 1, len(res)):
		a = []
		a.append(res[i][j])
		a.append([i,j])
		h.append(a)
heapq.heapify(h)

k = int(argv[2])
clusters = []
for i in range(1, col):
	clusters.append(i)

# centroid of each cluster
#centroids = {}
#for cluster in clusters:
#	centroids[cluster] = computeCentroid(cluster)

centroids = {}
for i in clusters:
	centroids[str(i)] = mat[:,i]


current_clusters = clusters

while len(current_clusters) > k:
	#n_cluster = []
	pop = heapq.heappop(h)
	if isvaild(pop[1]):
		#n_cluster.append(pop[1])
		current_clusters = refreshCluster(pop[1])
		addNewDist(merge(pop[1]))
		current_clusters = clusters

for c in current_clusters:
	if type(c) is list:
		print ",".join(str(x) for x in c)
	else:
		print c



