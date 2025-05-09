/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2020 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


#ifndef _RM_GPU_OPS_H_
#define _RM_GPU_OPS_H_



#include <nvtypes.h>
#include <nvCpuUuid.h>
#include <stdarg.h>
#include <nv-caps.h>
#include <nv-ioctl.h>
#include <nvmisc.h>

NV_STATUS  NV_API_CALL  rm_gpu_ops_create_session (nvidia_stack_t *, nvgpuSessionHandle_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_destroy_session (nvidia_stack_t *, nvgpuSessionHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_device_create (nvidia_stack_t *, nvgpuSessionHandle_t, const nvgpuInfo_t *, const NvProcessorUuid *, nvgpuDeviceHandle_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_device_destroy (nvidia_stack_t *, nvgpuDeviceHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_create(nvidia_stack_t *, nvgpuDeviceHandle_t, unsigned long long, unsigned long long, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_address_space(nvidia_stack_t *, nvgpuDeviceHandle_t, NvHandle, NvHandle, nvgpuAddressSpaceHandle_t *, nvgpuAddressSpaceInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_address_space_destroy(nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_fb(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_alloc_pages(nvidia_stack_t *, void *, NvLength, NvU32 , nvgpuPmaAllocationOptions_t, NvU64 *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_free_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_pin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_unpin_pages(nvidia_stack_t *, void *, NvU64 *, NvLength , NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_get_pma_object(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, void **, const nvgpuPmaStatistics_t *);
NV_STATUS  NV_API_CALL  rm_gpu_ops_pma_register_callbacks(nvidia_stack_t *sp, void *, nvPmaEvictPagesCallback, nvPmaEvictRangeCallback, void *);
void       NV_API_CALL  rm_gpu_ops_pma_unregister_callbacks(nvidia_stack_t *sp, void *);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_alloc_sys(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvLength, NvU64 *, nvgpuAllocInfo_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_get_p2p_caps(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, nvgpuP2PCapsParams_t);

NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_map(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, NvLength, void **, NvU32);
NV_STATUS  NV_API_CALL  rm_gpu_ops_memory_cpu_ummap(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, void*);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_allocate(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, const nvgpuChannelAllocParams_t *, nvgpuChannelHandle_t *, nvgpuChannelInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_channel_destroy(nvidia_stack_t *, nvgpuChannelHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_memory_free(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64);
NV_STATUS  NV_API_CALL rm_gpu_ops_query_caps(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuCaps_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_query_ces_caps(nvidia_stack_t *sp, nvgpuAddressSpaceHandle_t, nvgpuCesCaps_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_gpu_info(nvidia_stack_t *, const NvProcessorUuid *pUuid, const nvgpuClientInfo_t *, nvgpuInfo_t *);
NV_STATUS  NV_API_CALL rm_gpu_ops_service_device_interrupts_rm(nvidia_stack_t *, nvgpuDeviceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_dup_allocation(nvidia_stack_t *, NvHandle, nvgpuAddressSpaceHandle_t, NvU64, nvgpuAddressSpaceHandle_t, NvU64*, NvBool);

NV_STATUS  NV_API_CALL  rm_gpu_ops_dup_memory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, NvHandle *, nvgpuMemoryInfo_t);

NV_STATUS  NV_API_CALL rm_gpu_ops_free_duped_handle(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_fb_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuFbInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_ecc_info(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuEccInfo_t);
NV_STATUS NV_API_CALL rm_gpu_ops_own_page_fault_intr(nvidia_stack_t *, nvgpuDeviceHandle_t, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_fault_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_fault_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuFaultInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, void *, NvU32 *);
NV_STATUS  NV_API_CALL rm_gpu_ops_has_pending_non_replayable_faults(nvidia_stack_t *, nvgpuFaultInfo_t, NvBool *);
NV_STATUS  NV_API_CALL rm_gpu_ops_init_access_cntr_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_destroy_access_cntr_info(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_own_access_cntr_intr(nvidia_stack_t *, nvgpuSessionHandle_t, nvgpuAccessCntrInfo_t, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_enable_access_cntr(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t, nvgpuAccessCntrConfig_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_disable_access_cntr(nvidia_stack_t *, nvgpuDeviceHandle_t, nvgpuAccessCntrInfo_t);
NV_STATUS  NV_API_CALL  rm_gpu_ops_set_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvU64, unsigned, NvBool);
NV_STATUS  NV_API_CALL  rm_gpu_ops_unset_page_directory (nvidia_stack_t *, nvgpuAddressSpaceHandle_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_p2p_object_create(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, nvgpuAddressSpaceHandle_t, NvHandle *);
void       NV_API_CALL rm_gpu_ops_p2p_object_destroy(nvidia_stack_t *, nvgpuSessionHandle_t, NvHandle);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_external_alloc_ptes(nvidia_stack_t*, nvgpuAddressSpaceHandle_t, NvHandle, NvU64, NvU64, nvgpuExternalMappingInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_retain_channel(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvHandle, NvHandle, void **, nvgpuChannelInstanceInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_bind_channel_resources(nvidia_stack_t *, void *, nvgpuChannelResourceBindParams_t);
void       NV_API_CALL rm_gpu_ops_release_channel(nvidia_stack_t *, void *);
void       NV_API_CALL rm_gpu_ops_stop_channel(nvidia_stack_t *, void *, NvBool);
NV_STATUS  NV_API_CALL rm_gpu_ops_get_channel_resource_ptes(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, NvP64, NvU64, NvU64, nvgpuExternalMappingInfo_t);
NV_STATUS  NV_API_CALL rm_gpu_ops_report_non_replayable_fault(nvidia_stack_t *, nvgpuAddressSpaceHandle_t, const void *);

#endif
