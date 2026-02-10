usage:
	#for preview
	bash conv.sh input.cu <string of policies (hhd)> 
	#to file
	bash conv.sh input.cu <string of policies (hhd)> > output.cu

m = do nothing (migrate)

d = Pin to device 
```
cudaMallocManaged((void **) &pointer, size, cudaMemAttachGlobal);

== 

void * devptr;
cudaMalloc(&devptr, size);
CHECK_CUDA_ERROR();
cudaMemcpy(devptr, *a, size, cudaMemcpyHostToDevice);
CHECK_CUDA_ERROR();
CUDA_CHECK(cudaFree(*a));
CHECK_CUDA_ERROR();
*a = devptr;
break;

||

cudaMemAdvise(*a, size, cudaMemAdviseSetPreferredLocation, 0);
cudaMemPrefetchAsync(*a, size, 0, 0);
```

h = Pin to host
```
cudaMallocManaged((void **) &pointer, size, cudaMemAttachGlobal); 

==

cudaMallocManaged((void **) &pointer, size, cudaMemAttachGlobal); 
cudaMemAdvise(pointer, size, cudaMemAdviseSetPrefferedLocation, cudaCpuDeviceId);
cudaMemAdvise(pointer, size, cudaMemAdviseSetAccessedBy, 0);
```
