from sys import argv
from pyspark import SparkContext
sc = SparkContext(appName = "inf553")

D = open(argv[1])
data = D.readlines()
minSup = float(argv[2])	# min support ratio
d1 = []
for b in data:
	b = b.strip('\n').strip('\r').split(',')
	d1.append(b)	            # get  the original baskets
lend1 = len(d1)
data = sc.parallelize(d1,2)
writeFile = open(argv[3], 'w')

def apriori(partition):
    C1 = {}
    pBaskets = []
    count = 0
    for basket in partition:
	count += 1
	pBaskets.append(basket)
	for i in basket:
	    if i in C1:
		C1[i] += 1
	    else:
		C1[i] = 1

	keys1 = []
	keys = C1.keys()
    for k in keys:
	if C1[k] * 1.0 / len(pBaskets) >= minSup:
	    keys1.append(k)

	res = keys1
	frequent = keys1

	passNo = 2
	maxPass = len(frequent)
    while passNo <= maxPass:
	passtmp = combinations(frequent, passNo)
	tmp = []
	passNo += 1
	passN = []
	for p in passtmp:
	    passN.append(p)
	for p in passN:
	    if(countNum(pBaskets, p) * 1.0 / count) >= minSup:
		res.append(p)
		tmp.append(p)
	frequent = computeFrequent(tmp)		
    
    return res

def countNum(D, pair):
    c = 0
    for basket in D:
	have = True
	for item in pair:
	    if item not in basket:
		have = False
	if have:
		c += 1
    return c


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def computeFrequent(pairs):
    frequent = []
    for pair in pairs:
        for item in pair:
	    if item not in frequent:
		frequent.append(item)
    return frequent

def secondPass(iterator):
    list = []
    for v in iterator:
        if (countNum(d1, v) * 1.0 / len(d1)) >= minSup:
            list.append(v)
    return list

tmp  = data.mapPartitions(apriori).distinct()
res = tmp.mapPartitions(secondPass).collect() 
for r in res:
    if len(r) == 1:
        writeFile.write( str(r[0]) + '\n')
    else:
        s = ",".join(str(a) for a in r)
        writeFile.write(s + '\n')


