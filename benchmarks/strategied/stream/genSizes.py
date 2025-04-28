import sys

def GB_to_arr_size(GB):
    nbytes = GB * 1024. * 1024. * 1024. 
    arr_size = int(nbytes / 3 / 4) #3 arrays, sizeof float
    arr_size = arr_size - (arr_size%1024)
    return int(arr_size)

GB = float(sys.argv[1])
print(GB_to_arr_size(GB))
