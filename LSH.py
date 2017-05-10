from sys import argv
from pyspark import SparkContext
sc = SparkContext(appName = "inf553")

D = open(argv[1])
data = D.readlines()
d1 = []
for d in data:
    d = d.strip('\n').strip('\r').split(',')
    d1.append(d)
writeFile = open(argv[2],'w')

def computeSig(partition):
    sig = []
    for user in partition:
        item = user[1:]
        l = [102] * 20
        for mv in item:
            for i in range(20):
                hashi = (3 * int(mv) + 13 * i) % 100
                if hashi < l[i]:
                    l[i] = hashi
            l.append(user[0])
        sig.append(l)
    return sig

def chunk(partition):
    for user in partition:
        for i in range(0, 19, 4):
            yield (tuple(user[i:i + 4]), user[20], (i + 4) / 4 )

def jaccard(iterator):
    res = []
    k = iterator[0]
    Vs = iterator[1]
    index1 = int(k[1:]) - 1
    list1 = d1[index1][1:]
    for V in Vs:
        index2 = int(V[1:]) - 1
        list2 = d1[index2][1:]
        similar = [val for val in list1 if val in list2]
        com = list(set(list1).union(set(list2)))
        jac = len(similar) * 1.0 / len(com) * 1.0
        if jac != 0:
            res.append((k,(jac,V)))
    return res

def gen(list):
    l = []
    for i in list:
        l.append(i)
    return l

data = sc.parallelize(d1,2)
data1 = data.mapPartitions(computeSig).mapPartitions(chunk)\
        .map(lambda x:((x[2],x[0]),x[1]))\
        .groupByKey().filter(lambda (x,y):len(y)>1).map(lambda (x,y):list(y))
print data1

def mkPairs(iterator):
    res = []
    for a in iterator:
        l = []
        for b in iterator:
            if a!=b:
                l.append(b)
        res.append((a,l))
    return res

def printf(iterator):
    for i in iterator:
        return i

def rmdup(list):
    l = []
    for a in list:
        if a not in l:
            l.append(a)
    return l

def set1(list):
    s = set()
    for i in list:
        s = set(i).union(s)
    return s

def map1(iterator):
    k = iterator[0]
    l = iterator[1]
    L = []
    for i in l:
        L.append((k,i))
    return L

def sort1(iterator):
    return sorted(iterator, key = lambda tup:(-tup[0],int(tup[1][1:])))

def top5(iterator):
    l = []

    if len(iterator) < 5 and len(iterator) > 0:
        for i in range(len(iterator)):
            l.append(iterator[i][1])
    else:
        for i in range(5):
            l.append(iterator[i][1])
    return l

def sortOut(iterator):
    return sorted(iterator, key = lambda x:(int(x[1:])))


data2 = data1.map(mkPairs).flatMap(gen).groupByKey().mapValues(rmdup).mapValues(set1).mapValues(list)\
        .flatMap(jaccard).groupByKey().mapValues(list).mapValues(sort1)\
        .mapValues(top5).mapValues(sortOut).collect()


res1 = sorted(data2, key = lambda x:(int(x[0][1:])))

for r in res1:
    writeFile.write(r[0] + ':')
    s = ",".join(str(a) for a in r[1])
    writeFile.write(s + '\n')
