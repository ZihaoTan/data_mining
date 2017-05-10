from sys import argv
import numpy
argv[2:] = [int(x) for x in argv[2:]]
n, m, f, k = argv[2:]

mat = numpy.zeros((n,m))
#for i in range(n):
#	for j in range(m):
#		mat[i,j] = 9999999

input_file = open(argv[1])
lines = input_file.readlines()
for line in lines:
	line = line.split(',')
	mat[int(line[0]) - 1, int(line[1]) - 1] = line[2]

# intitial U, V matrix with 1's
U = numpy.ones((n,f))
V = numpy.ones((f,m))

def rmse(prediction, target):
	count = 0
	err = 0
	n, m = target.shape
	for i in range(n):
		for j in range(m):
			if target[i,j] != 0:
				err += (target[i,j] - prediction[i,j]) ** 2
				count += 1
	return(err/count) ** 0.5
	#diff = prediction - target
	#reutrn numpy.sqrt(numpy.sum(numpy.power(diff,2)) / (n * m))
	#return numpy.sqrt(numpy.mean((prediction - target) ** 2))



for step in range(k):

	for r in range(n):
		for s in range(f):
			sum_num = 0
			sum_frac = 0
			for j in range(m):
				sum_min = 0
				for b in range(f):
					if b != s:
						sum_min += U[r,b] * V[b,j]
				if mat[r,j] != 0:
					sum_num += V[s,j] * (mat[r,j] - sum_min)
					sum_frac += V[s,j] * V[s,j]
			U[r,s] = sum_num/sum_frac

	for s in range(m):
		for r in range(f):
			sum_num = 0
			sum_frac = 0
			for i in range(n):
				sum_min = 0
				for b in range(f):
					if b != r:
						sum_min += U[i,b] * V[b,s]
				if mat[i,s] != 0:
					sum_num += U[i,r] * (mat[i,s] - sum_min)
					sum_frac += U[i,r] * U[i,r]
			V[r,s] = sum_num/sum_frac

	
	err = float(rmse(numpy.dot(U,V), mat))
	print "%.4f" % err



			