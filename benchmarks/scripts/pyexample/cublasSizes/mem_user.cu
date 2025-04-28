#include <stdio.h>
#include <unistd.h>

//__global__
//void saxpy(int n, float a, float *x, float *y)
//{
//  int i = blockIdx.x*blockDim.x + threadIdx.x;
//  //pass -1 as n so never true 
//  if (i < n) y[i] = a*x[i] + y[i];
//}

int main(int  argc, char ** argv)
{
  if(argc < 3){
	  printf("Usage: ./mem_user <amount in GiB> <time (s)>\n");
	  exit(0);
  }
  //Allocate N gigs of device memory
  size_t N = (size_t)atoi(argv[1]) * (size_t)1024 * (size_t)1024 * (size_t)1024;
  float *d_x;
  cudaMalloc((void**)&d_x, N); 
  sleep(atoi(argv[2]));
}
