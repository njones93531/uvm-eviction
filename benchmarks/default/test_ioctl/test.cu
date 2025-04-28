#include <sys/ioctl.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include "uvm_ioctl.h"
#include "uvm_ioctl_scalab.h"
#include "uvm_linux_ioctl.h"
#include "ioctl_common_nvswitch.h"

typedef enum
{
    UVM_FD_UNINITIALIZED,
    UVM_FD_INITIALIZING,
    UVM_FD_VA_SPACE,
    UVM_FD_MM,
    UVM_FD_COUNT
} uvm_fd_type_t;

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

#define UVM_IOCTL_TESTP UVM_IOCTL_BASE(5000)
int test()
{
    UVM_IOCTL_TESTP_PARAMS test_params;
    UVM_INITIALIZE_PARAMS params;
    test_params.myarg = "aloo from user space\n";
    params.flags = UVM_FD_UNINITIALIZED;
    

    int fd = open("/dev/nvidia-uvm", O_RDWR | FD_CLOEXEC);
    if (fd < 0)
    {
        perror("Failed to open the device");
        return errno;
    }
    printf("Trying UVM_IOCTL_INIT: %d\n", UVM_IOCTL_TESTP);
    printf("init: %d\n", ioctl(fd, UVM_INITIALIZE, &params));
    printf("Trying UVM_IOCTL_TESTP: %d\n", UVM_IOCTL_TESTP);
    int ret = ioctl(fd, UVM_IOCTL_TESTP, &test_params);
    close(fd);
    return ret;
}

int main(void)
{
    int* foo;
    cudaMallocManaged(&foo, 5000);
    printf("foo: %ld\n", (long int) foo);
    foo[15] = 1204;
    printf("foo[%d] = %d\n", 15, foo[15]);
    int ret = test();
    printf("ret: %d\n", ret);
    cudaFree(foo);
    return ret;
}
