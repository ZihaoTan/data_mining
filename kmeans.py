#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys
import scipy
import numpy as np
from pyspark.sql import SparkSession
from pyspark import SparkContext
from scipy import sparse
from scipy.sparse import csc_matrix
import scipy.linalg
from numpy import linalg
import math
from scipy.spatial.distance import *

def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])

def tagNum(iterator):
    count = 0
    for i in iterator:
        count += 1
        return([count, i])

def toArray(lines):
    return np.array(lines)

def closestPoint(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def closestCos(p, centers):
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = cosine(p, centers[i])
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

def computeLength(col):
    return scipy.linalg.norm(col)

#def cos(v1, v2):
#    product = np.sum(np.multiply(v1, v2))
#    return 1 - (product / linalg.norm(v1) * linalg.norm(v2))


if __name__ == "__main__":

    #if len(sys.argv) != 4:
    #    print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
    #    exit(-1)


    #spark = SparkSession\
    #    .builder\
    #    .appName("PythonKMeans")\
    #    .getOrCreate()
    
    sc = SparkContext(appName="PythonKMeans")
    #lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    #data = lines.map(parseVector).cache()
    
    # read data
    input_file = open(sys.argv[1])
    data = input_file.readlines()
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

    # compute tf * idf, idf = log(N+1)/(df+1)
    val = []
    for d in data:
        d = d.split(" ")
        d[2] = float(d[2]) * math.log((float(docs) + 1) / float(df[d[1]] + 1), 2)
        val.append(d)

    # build a matrix
    mat = np.zeros((row, col))
    for v in val:
        mat[int(v[1]), int(v[0])] = v[2]

    elength = []
    for i in range(len(mat[0])):
        elength.append(computeLength(mat[:,i]))

    # normalize
    for i in range(len(mat[0])):
        for j in range(len(mat[:,0])):
            if mat[j,i] != 0:
                mat[j,i] = mat[j,i] / elength[i]
    
    
    #for i in range(len(mat[:,0])):
    #    mat[i,0] = i
    mat = mat[1:,1:].T
    
    #lines = mat.map(parseVectorFromMat).collect()
    lines = mat.tolist()
    data = sc.parallelize(lines).map(toArray)

    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])
    f = open(sys.argv[4], 'w')
    
    kPoints = data.repartition(1).takeSample(False, K, 1)
    tempDist = 1.0
    
    while tempDist > convergeDist:
        closest = data.map(lambda p: (closestCos(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
        tempDist = sum(np.sqrt(np.sum((kPoints[iK] - p) ** 2)) for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    #print("Final centers: " + str(kPoints))
    
    res = []

    for k in kPoints:
        res.append(np.count_nonzero(k))

    #f = open(sys.argv[4], 'w')
    for r in res:
        f.write(str(r))
        f.write('\n')
    sc.stop()
