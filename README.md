# uvm-eviction
eviction methods for uvm

# Navigation
## drivers
- top level is driver version to match kernel 
- `vanilla` driver is the unmodified uvm driver for reference, reinstall
- `fault-tracking` logs faults, prefetches, eviction to syslog; reference driver for access counters
## tools
- `syslogger` contains the tool for parsing data out of the system log
- `plotv2` contains .sh scripts that operate the .py scripts for reproducing plots and analysis from the paper if data is available

# Overview
TODO

## Swapping the driver
To enable logging information, the driver with added logging features must be installed. Additionally, it may
need certain parameters to function correctly. This requires root permission in most cases.\
There are two drivers provided. One collects produces "batch" experiment data from the paper, and the other produces
"fault" experiment data from the paper (access patterns). These drivers are at:

../drivers/x86\_64-460.74.27.04/vanilla/kernel/

Notes: The following assumes that the appropriate NVIDIA driver is already installed. These steps will replace the 
existing UVM driver for convenience over reboot and multiple experiments. The base UVM driver will need to 
be reinstalled to revert to the original system status. Full system specifications are at the end of this file.

### Build
`cd drivers/x86_64-460.74.27.04/*/kernel/`\
`make modules`\
`sudo make modules_install`

### Load
`sudo rmmod nvidia_uvm`\
`sudo modprobe nvidia_uvm`\
Note: turn on/off prefetching with 1/0 and/or maximum batch size using arguments as below:\
`sudo modprobe nvidia-uvm uvm_perf_prefetch_enable={1,0}`\
`sudo modprobe nvidia-uvm uvm_perf_fault_batch_count=${batch_size}`

## Applications
TODO

## Tools
Several tools are used to assist with data collection, as all data is logged to the system logger. dmesg is too slow in the event of page fault collection

### Syslogger
Move data from system log to file while application is running. This is used in place of `dmesg` because 
`dmesg` is too slow for prefetch-enabled fault tracking and most likely for full-pattern access too. Usage:\
`tools/syslogger/log "$logfile"&`\
`pid=$!`\
`<path>/app`\
`kill $pid`

### log2csv
Process syslogger output into CSV:\
`tools/sys2csv/log2csv.sh $logfile`

## Data and Formatting
TODO

## Plotting
TODO

## Experimental System Specs
TODO: update this for new cuda version?
Relevant hardware details: Titan V GPU, MD Epyc 7551P 32-Core CPU, 128GB DDR4\
Operating systems and versions: Fedora 33 running 5.9.16-200.fc33.x86\_64, CUDA 11.2, and NVIDIA Driver version 460.27.04\
Compilers and versions: GCC 10.2.1 and NVCC cuda\_11.2.r11.2/compiler.29373293\_0\
Libraries and versions: CUBLAS 11.2\
Note: The provided UVM drivers require the compatible NVIDIA Driver on the system: 460.74.27.04

