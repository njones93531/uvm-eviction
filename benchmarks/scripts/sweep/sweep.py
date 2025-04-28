import os
import re
import sh
import subprocess
import sys
import time
import config

benchmark_dir = "../../sweep"
#photo rep msize = [10, 12, 14, 16, 18. 20. 22, 24]
#photo rep strides = [1, 4, 16, 64]
strides = [1, 2, 30, 32, 34, 60, 64, 68, 120, 128, 136, 252, 256, 260, 500, 512, 524, 1000, 1024, 1048]#, 65537, 2097152] #stride should be >= 1
stride = [128]#[strides[int(sys.argv[1])]]
#msizes = [6, 9, 12, 15, 18, 21, 24]
msizes = [15]
access_type = ["r", "w"] #r, w, rw
array_loc = [0, 1] 
write_val = 1.3
reuse_stride = 8
reuses = [21, 22, 23]
reuse = [reuses[int(sys.argv[1])]]

oldpwd = os.getcwd()
os.chdir(benchmark_dir)
for i in array_loc:
    for j in access_type: 
        read = 0
        write = 0
        if j=="r" or j=="rw":
            read = 1 
        if j=="w" or j=="rw":
            write = 1
        for k in stride:
            for l in msizes:
                sh.make("clean")
                defs =f"-DSTRIDE={k} -DMEM={l}lu -DARRAY_LOC={i} -DREAD={read} -DWRITE={write} -DWRITE_VAL={write_val} -DREUSE={reuse} -DREUSE_STRIDE={reuse_stride}"
                sh.make(f'DEFS={defs}')
                nvprof_cmd = ['nvprof', '--profile-api-trace', 'none',  '--csv', './sweep']
                output = f'sweep_mem{l}_stride{k}_access{j}_arrayloc{i}_reuse{reuse}_rs{reuse_stride}.log'
                logfile = open(output, 'w')
                print(output)
                p = subprocess.Popen(nvprof_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                for line2 in p.stderr:
                    print(line2.decode('utf-8'), end=' ')
                    logfile.write(line2.decode('utf-8'))
                for line3 in p.stdout:
                    logfile.write(line3.decode('utf-8'))
                print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")
                p.communicate()
 

