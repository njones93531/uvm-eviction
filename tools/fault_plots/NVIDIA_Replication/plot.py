import matplotlib.pyplot as plt
import numpy as np

linesToSkip = 28
onLine = 0

page_fault_rw = []
zero_copy_rw = []
stripe_rw = []

page_fault_streaming = []
zero_copy_streaming = []
stripe_streaming = []

page_fault_block = []
zero_copy_block = []
stripe_block = []

x = []

for i in range(8, 31):
    x.append(i/10)


with open("zero_copy_data.txt", "r") as file1:
        for line in file1:
                if(onLine > linesToSkip):
                        tokens = re.split(',| ', line)
                        if(tokens[1] == "Page_Fault"):
                                page_fault_rw.append(float(tokens[8]))
                        if(tokens[1] == "Zero_copy"):
                                zero_copy_rw.append(float(tokens[8]))
                        if(tokens[1] == "stripe_gpu_cpu"):
                                stripe_rw.append(float(tokens[8]))

                onLine+=1

onLine = 0

with open("exampleData1.txt", "r") as file2:
        for line in file2:
                if(onLine > linesToSkip):
                        tokens = re.split(',| ', line)
                        if(tokens[2] == "streaming"):
                            if(tokens[1] == "Page_Fault"):
                                page_fault_streaming.append(float(tokens[8]))
                            if(tokens[1] == "Zero_copy"):
                                zero_copy_streaming.append(float(tokens[8]))
                            if(tokens[1] == "stripe_gpu_cpu"):
                                stripe_streaming.append(float(tokens[8]))
                        if(tokens[2] == "block_streaming"):
                            if(tokens[1] == "Page_Fault"):
                                page_fault_block.append(float(tokens[8]))
                            if(tokens[1] == "Zero_copy"):
                                zero_copy_block.append(float(tokens[8]))
                            if(tokens[1] == "stripe_gpu_cpu"):
                                stripe_block.append(float(tokens[8]))

                onLine+=1


plt.xlabel("Oversubscription Factor")
plt.ylabel("Bandwidth (GB/s)")
plt.legend()
plt.tight_layout()
plt.plot(x, page_fault_rw, label='RandomWarp')
plt.plot(x, page_fault_streaming, label='GridStride')
plt.plot(x, page_fault_block, label='BlockStride')
plt.show()
