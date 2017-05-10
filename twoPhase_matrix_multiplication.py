from sys import argv
from operator import add
from pyspark import SparkContext

sc = SparkContext(appName = "inf553")

matA = argv[1]
matB = argv[2]
#content = argv[3]

lineA = sc.textFile(argv[1])
lineB = sc.textFile(argv[2])

mapA = lineA.map(lambda x: x.split(",")).map(lambda x: (x[1],("A", x[0], x[2])))
mapB = lineB.map(lambda x: x.split(",")).map(lambda x: (x[0],("B", x[1], x[2])))

def mul1(iterator):
    listA = []
    listB = []
    listC = []
    for v in iterator:
        if(v[0] == "A"):
            listA.append((v[1],v[2]))
        elif(v[0] == "B"):
            listB.append((v[1],v[2]))
    for a in listA:
        for b in listB:
            listC.append(((a[0],b[0]), int(a[1])* int(b[1])))
    return listC
output = mapA.union(mapB).groupByKey().mapValues(list).mapValues(mul1).flatMap(lambda (x,y): y).reduceByKey(add).collect()
thefile = open(argv[3],'w')
for v in output:
    #print '%s,%s\t%d' % (str(v[0][0]),str(v[0][1]), v[1])
    thefile.write('%s,%s\t%d\n' % (str(v[0][0]),str(v[0][1]),v[1]))
