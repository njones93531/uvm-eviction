INCLUDES=-I/home/tnallen/dev/uvm-eviction/drivers/x86_64-535.104.05/syscall/kernel/nvidia/ -I/home/tnallen/dev/uvm-eviction/drivers/x86_64-535.104.05/syscall/kernel/nvidia-uvm/ -I/home/tnallen/dev/uvm-eviction/drivers/x86_64-535.104.05/syscall/kernel/common/inc -I/home/tnallen/dev/uvm-eviction/drivers/x86_64-535.104.05/syscall/kernel/

all: test

test: test.cu ioctl_common_nvswitch.h Makefile
	nvcc -Xcompiler "-Wall -Wextra -O0 -g" test.cu -o test $(INCLUDES)
