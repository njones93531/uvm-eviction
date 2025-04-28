CC=nvcc
CFLAGS=-arch=sm_90

all: pager

pager: pager.cu
	$(CC) $(CFLAGS) -o pager pager.cu

clean:
	rm -f pager

