#include <cuda_runtime.h>
#include <stdio.h>
#include<unistd.h>
#define GIGS_TO_BYTES 1024. * 1024. * 1024.

void usage()
{
    printf("Usage: ./memblock <size in GB> <number of seconds>\n");
}

int main(int argc, char** argv)
{
    if(argc != 3) usage();
    int * memblock;
    float gigs = atof(argv[1]);
    size_t time = atoi(argv[2]);
    cudaMalloc(&memblock, (size_t) gigs*GIGS_TO_BYTES);
    sleep(time);
    return 0;
}
