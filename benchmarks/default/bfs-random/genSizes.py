import math
import sys

def memToVertices(mem, percent):
    return math.ceil((-(16-8*percent)+math.sqrt((16-8*percent)**2+4*8*percent*mem))/(16*percent))

def verticesToEdges(vertices, percent):
    return math.ceil(percent*(vertices*(vertices-1)/2))

def verticesEqualEdges(mem):
    return math.ceil(mem/80)

if __name__ == "__main__":
    percent = float(sys.argv[1])
    mem = float(sys.argv[2])
    #byteSizes = [i*2**30 for i in range(10,101,10)]
    byteSize = int(mem*2**30)
    vertices = memToVertices(byteSize, percent)
    edges = verticesToEdges(vertices, percent)
    #vertices = [verticesEqualEdges(s) for s in byteSizes]
    print(edges, vertices);
