This file contains notes about the options for the nvidia_uvm driver.

# modinfo
parm:           uvm_perf_access_counter_mimc_migration_enable:Whether MIMC access counters will trigger migrations.Valid values: <= -1 (default policy), 0 (off), >= 1 (on) (int)
parm:           uvm_perf_access_counter_momc_migration_enable:Whether MOMC access counters will trigger migrations.Valid values: <= -1 (default policy), 0 (off), >= 1 (on) (int)
parm:           uvm_perf_access_counter_batch_count:uint
parm:           uvm_perf_access_counter_granularity:Size of the physical memory region tracked by each counter. Valid values asof Volta: 64k, 2m, 16m, 16g (charp)
parm:           uvm_perf_access_counter_threshold:Number of remote accesses on a region required to trigger a notification.Valid values: [1, 65535] (uint)

uvm_perf_access_counter_batch_count= "256"
uvm_perf_access_counter_granularity= "2m"
uvm_perf_access_counter_mimc_migration_enable= "-1"
uvm_perf_access_counter_momc_migration_enable= "-1"
uvm_perf_access_counter_threshold= "256"


// Whether access counter migrations are enabled or not. The policy is as
// follows:
// - MIMC migrations are enabled by default on P9 systems with ATS support
// - MOMC migrations are disabled by default on all systems
// - Users can override this policy by specifying on/off

// Whether this counter refers to outbound accesses to remote GPUs or
// sysmem (MIMC), or it refers to inbound accesses from CPU or a non-peer
// GPU (whose accesses are routed through the CPU, too) to vidmem (MOMC)
