#ifndef UVM_IOCTL_SCALAB_H
#define UVM_IOCTL_SCALAB_H
#include "uvm_ioctl.h"

#define UVM_IOCTL_TESTP UVM_IOCTL_BASE(5000)
typedef struct
{
    char* myarg; // IN
    NV_STATUS rmStatus; // OUT
} UVM_IOCTL_TESTP_PARAMS;

#endif
