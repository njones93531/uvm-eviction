/* _NVRM_COPYRIGHT_BEGIN_
 *
 * Copyright 1999-2020 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 *
 * _NVRM_COPYRIGHT_END_
 */


/*
 * Os interface definitions needed by os-interface.c
 */

#ifndef _OS_INTERFACE_H_
#define _OS_INTERFACE_H_

/******************* Operating System Interface Routines *******************\
*                                                                           *
* Module: os-interface.h                                                    *
*       Included by os.h                                                    *
*       Operating system wrapper functions used to abstract the OS.         *
*                                                                           *
\***************************************************************************/

#include <stdarg.h>
#include <nv-kernel-interface-api.h>
#include <os/nv_memory_type.h>



typedef struct
{
    NvU32  os_major_version;
    NvU32  os_minor_version;
    NvU32  os_build_number;
    const char * os_build_version_str;
    const char * os_build_date_plus_str;
}os_version_info;

/* Each OS defines its own version of this opaque type */
struct os_work_queue;

/*
 * ---------------------------------------------------------------------------
 *
 * Function prototypes for OS interface.
 *
 * ---------------------------------------------------------------------------
 */

NvU64       NV_API_CALL  os_get_num_phys_pages       (void);
NV_STATUS   NV_API_CALL  os_alloc_mem                (void **, NvU64);
void        NV_API_CALL  os_free_mem                 (void *);
NV_STATUS   NV_API_CALL  os_get_current_time         (NvU32 *, NvU32 *);
NvU64       NV_API_CALL  os_get_current_tick         (void);
NvU64       NV_API_CALL  os_get_current_tick_hr      (void);
NvU64       NV_API_CALL  os_get_tick_resolution      (void);
NV_STATUS   NV_API_CALL  os_delay                    (NvU32);
NV_STATUS   NV_API_CALL  os_delay_us                 (NvU32);
NvU64       NV_API_CALL  os_get_cpu_frequency        (void);
NvU32       NV_API_CALL  os_get_current_process      (void);
void        NV_API_CALL  os_get_current_process_name (char *, NvU32);
NvU32       NV_API_CALL  os_get_current_pasid        (void);
NV_STATUS   NV_API_CALL  os_get_current_thread       (NvU64 *);
char*       NV_API_CALL  os_string_copy              (char *, const char *);
NvU32       NV_API_CALL  os_string_length            (const char *);
NvU32       NV_API_CALL  os_strtoul                  (const char *, char **, NvU32);
NvS32       NV_API_CALL  os_string_compare           (const char *, const char *);
NvS32       NV_API_CALL  os_snprintf                 (char *, NvU32, const char *, ...);
NvS32       NV_API_CALL  os_vsnprintf                (char *, NvU32, const char *, va_list);
void        NV_API_CALL  os_log_error                (const char *, va_list);
void*       NV_API_CALL  os_mem_copy                 (void *, const void *, NvU32);
NV_STATUS   NV_API_CALL  os_memcpy_from_user         (void *, const void *, NvU32);
NV_STATUS   NV_API_CALL  os_memcpy_to_user           (void *, const void *, NvU32);
void*       NV_API_CALL  os_mem_set                  (void *, NvU8, NvU32);
NvS32       NV_API_CALL  os_mem_cmp                  (const NvU8 *, const NvU8 *, NvU32);
void*       NV_API_CALL  os_pci_init_handle          (NvU32, NvU8, NvU8, NvU8, NvU16 *, NvU16 *);
NV_STATUS   NV_API_CALL  os_pci_read_byte            (void *, NvU32, NvU8 *);
NV_STATUS   NV_API_CALL  os_pci_read_word            (void *, NvU32, NvU16 *);
NV_STATUS   NV_API_CALL  os_pci_read_dword           (void *, NvU32, NvU32 *);
NV_STATUS   NV_API_CALL  os_pci_write_byte           (void *, NvU32, NvU8);
NV_STATUS   NV_API_CALL  os_pci_write_word           (void *, NvU32, NvU16);
NV_STATUS   NV_API_CALL  os_pci_write_dword          (void *, NvU32, NvU32);
NvBool      NV_API_CALL  os_pci_remove_supported     (void);
void        NV_API_CALL  os_pci_remove               (void *);
void*       NV_API_CALL  os_map_kernel_space         (NvU64, NvU64, NvU32, NvU32);
void        NV_API_CALL  os_unmap_kernel_space       (void *, NvU64);
void*       NV_API_CALL  os_map_user_space           (NvU64, NvU64, NvU32, NvU32, void **);
void        NV_API_CALL  os_unmap_user_space         (void *, NvU64, void *);
NV_STATUS   NV_API_CALL  os_flush_cpu_cache          (void);
NV_STATUS   NV_API_CALL  os_flush_cpu_cache_all      (void);
NV_STATUS   NV_API_CALL  os_flush_user_cache         (NvU64, NvU64, NvU64, NvU64);
void        NV_API_CALL  os_flush_cpu_write_combine_buffer(void);
NvU8        NV_API_CALL  os_io_read_byte             (NvU32);
NvU16       NV_API_CALL  os_io_read_word             (NvU32);
NvU32       NV_API_CALL  os_io_read_dword            (NvU32);
void        NV_API_CALL  os_io_write_byte            (NvU32, NvU8);
void        NV_API_CALL  os_io_write_word            (NvU32, NvU16);
void        NV_API_CALL  os_io_write_dword           (NvU32, NvU32);
NvBool      NV_API_CALL  os_is_administrator         (void);
NvBool      NV_API_CALL  os_allow_priority_override  (void);
void        NV_API_CALL  os_dbg_init                 (void);
void        NV_API_CALL  os_dbg_breakpoint           (void);
void        NV_API_CALL  os_dbg_set_level            (NvU32);
NvU32       NV_API_CALL  os_get_cpu_count            (void);
NvU32       NV_API_CALL  os_get_cpu_number           (void);
NV_STATUS   NV_API_CALL  os_disable_console_access   (void);
NV_STATUS   NV_API_CALL  os_enable_console_access    (void);
NV_STATUS   NV_API_CALL  os_registry_init            (void);
NV_STATUS   NV_API_CALL  os_schedule                 (void);
NV_STATUS   NV_API_CALL  os_alloc_spinlock           (void **);
void        NV_API_CALL  os_free_spinlock            (void *);
NvU64       NV_API_CALL  os_acquire_spinlock         (void *);
void        NV_API_CALL  os_release_spinlock         (void *, NvU64);
NV_STATUS   NV_API_CALL  os_get_address_space_info   (NvU64 *, NvU64 *, NvU64 *, NvU64 *);
NV_STATUS   NV_API_CALL  os_queue_work_item          (struct os_work_queue *, void *);
NV_STATUS   NV_API_CALL  os_flush_work_queue         (struct os_work_queue *);
NV_STATUS   NV_API_CALL  os_alloc_mutex              (void **);
void        NV_API_CALL  os_free_mutex               (void *);
NV_STATUS   NV_API_CALL  os_acquire_mutex            (void *);
NV_STATUS   NV_API_CALL  os_cond_acquire_mutex       (void *);
void        NV_API_CALL  os_release_mutex            (void *);
void*       NV_API_CALL  os_alloc_semaphore          (NvU32);
void        NV_API_CALL  os_free_semaphore           (void *);
NV_STATUS   NV_API_CALL  os_acquire_semaphore        (void *);
NV_STATUS   NV_API_CALL  os_cond_acquire_semaphore   (void *);
NV_STATUS   NV_API_CALL  os_release_semaphore        (void *);
NvBool      NV_API_CALL  os_semaphore_may_sleep      (void);
NV_STATUS   NV_API_CALL  os_get_version_info         (os_version_info*);
NvBool      NV_API_CALL  os_is_isr                   (void);
NvBool      NV_API_CALL  os_pat_supported            (void);
void        NV_API_CALL  os_dump_stack               (void);
NvBool      NV_API_CALL  os_is_efi_enabled           (void);
NvBool      NV_API_CALL  os_is_xen_dom0              (void);
NvBool      NV_API_CALL  os_is_vgx_hyper             (void);
NV_STATUS   NV_API_CALL  os_inject_vgx_msi           (NvU16, NvU64, NvU32);
NvBool      NV_API_CALL  os_is_grid_supported        (void);
NvU32       NV_API_CALL  os_get_grid_csp_support     (void);
void        NV_API_CALL  os_get_screen_info          (NvU64 *, NvU16 *, NvU16 *, NvU16 *, NvU16 *);
void        NV_API_CALL  os_bug_check                (NvU32, const char *);
NV_STATUS   NV_API_CALL  os_lock_user_pages          (void *, NvU64, void **, NvU32);
NV_STATUS   NV_API_CALL  os_lookup_user_io_memory    (void *, NvU64, NvU64 **, void**);
NV_STATUS   NV_API_CALL  os_unlock_user_pages        (NvU64, void *);
NV_STATUS   NV_API_CALL  os_match_mmap_offset        (void *, NvU64, NvU64 *);
NV_STATUS   NV_API_CALL  os_get_euid                 (NvU32 *);
NV_STATUS   NV_API_CALL  os_get_smbios_header        (NvU64 *pSmbsAddr);
NV_STATUS   NV_API_CALL  os_get_acpi_rsdp_from_uefi  (NvU32 *);
void        NV_API_CALL  os_add_record_for_crashLog  (void *, NvU32);
void        NV_API_CALL  os_delete_record_for_crashLog (void *);
NV_STATUS   NV_API_CALL  os_call_vgpu_vfio           (void *, NvU32);
NV_STATUS   NV_API_CALL  os_numa_memblock_size       (NvU64 *);
NV_STATUS   NV_API_CALL  os_alloc_pages_node         (NvS32, NvU32, NvU32, NvU64 *);
NV_STATUS   NV_API_CALL  os_get_page                 (NvU64 address);
NV_STATUS   NV_API_CALL  os_put_page                 (NvU64 address);
NvU32       NV_API_CALL  os_get_page_refcount        (NvU64 address);
NvU32       NV_API_CALL  os_count_tail_pages         (NvU64 address);
void        NV_API_CALL  os_free_pages_phys          (NvU64, NvU32);
NV_STATUS   NV_API_CALL  os_call_nv_vmbus            (NvU32, void *);
NV_STATUS   NV_API_CALL  os_open_temporary_file      (void **);
void        NV_API_CALL  os_close_file               (void *);
NV_STATUS   NV_API_CALL  os_write_file               (void *, NvU8 *, NvU64, NvU64);
NV_STATUS   NV_API_CALL  os_read_file                (void *, NvU8 *, NvU64, NvU64);
NV_STATUS   NV_API_CALL  os_open_readonly_file       (const char *, void **);
NV_STATUS   NV_API_CALL  os_open_and_read_file       (const char *, NvU8 *, NvU64);












extern NvU32 os_page_size;
extern NvU64 os_page_mask;
extern NvU8  os_page_shift;

/*
 * ---------------------------------------------------------------------------
 *
 * Debug macros.
 *
 * ---------------------------------------------------------------------------
 */

#define NV_DBG_INFO       0x0
#define NV_DBG_SETUP      0x1
#define NV_DBG_USERERRORS 0x2
#define NV_DBG_WARNINGS   0x3
#define NV_DBG_ERRORS     0x4


void NV_API_CALL  out_string(const char *str);
int  NV_API_CALL  nv_printf(NvU32 debuglevel, const char *printf_format, ...);

#define NV_DEV_PRINTF(debuglevel, nv, format, ... ) \
        nv_printf(debuglevel, "NVRM: GPU " NV_PCI_DEV_FMT ": " format, NV_PCI_DEV_FMT_ARGS(nv), ## __VA_ARGS__)

#define NV_DEV_PRINTF_STATUS(debuglevel, nv, status, format, ... ) \
        nv_printf(debuglevel, "NVRM: GPU " NV_PCI_DEV_FMT ": " format " (0x%x)\n", NV_PCI_DEV_FMT_ARGS(nv), ## __VA_ARGS__, status)

/*
 * Fields for os_lock_user_pages flags parameter
 */
#define NV_LOCK_USER_PAGES_FLAGS_WRITE                     0:0
#define NV_LOCK_USER_PAGES_FLAGS_WRITE_NO                  0x00000000
#define NV_LOCK_USER_PAGES_FLAGS_WRITE_YES                 0x00000001








#endif /* _OS_INTERFACE_H_ */
