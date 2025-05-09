import sys 
import math 

def memToMatSize(mem):
    return int(math.sqrt(mem * 1024. * 1024. * 1024. / 12))

if __name__ == "__main__":
    mem = float(sys.argv[1])
    matsize = memToMatSize(mem)
    print(f"wA={matsize} wB={matsize} hA={matsize} hB={matsize}")
