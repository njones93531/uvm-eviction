/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nv-pci-table.h"
#include "nv-pci-types.h"
#include "nv-pci.h"
#include "nv-msi.h"
#include "nv-hypervisor.h"
#include "nv-reg.h"

#if defined(NV_VGPU_KVM_BUILD)
#include "nv-vgpu-vfio-interface.h"
#endif
#include <linux/iommu.h>

#if defined(NV_SEQ_READ_ITER_PRESENT)
#include <linux/seq_file.h>
#include <linux/kernfs.h>
#endif

#include "detect-self-hosted.h"

#if !defined(NV_BUS_TYPE_HAS_IOMMU_OPS)
#include <linux/iommu.h>
#endif
#if NV_IS_EXPORT_SYMBOL_GPL_pci_ats_supported
#include <linux/pci-ats.h>
#endif

extern int NVreg_GrdmaPciTopoCheckOverride;

static void
nv_check_and_exclude_gpu(
    nvidia_stack_t *sp,
    nv_state_t *nv
)
{
    char *uuid_str;

    uuid_str = rm_get_gpu_uuid(sp, nv);
    if (uuid_str == NULL)
    {
        NV_DEV_PRINTF(NV_DBG_INFO, nv, "Unable to read UUID");
        return;
    }

    if (nv_is_uuid_in_gpu_exclusion_list(uuid_str))
    {
        NV_STATUS rm_status = rm_exclude_adapter(sp, nv);
        if (rm_status != NV_OK)
        {
            NV_DEV_PRINTF_STATUS(NV_DBG_ERRORS, nv, rm_status,
                          "Failed to exclude GPU %s", uuid_str);
            goto done;
        }
        nv->flags |= NV_FLAG_EXCLUDE;
        NV_DEV_PRINTF(NV_DBG_INFO, nv, "Excluded GPU %s successfully\n",
                      uuid_str);
    }

done:
    os_free_mem(uuid_str);
}

static NvBool nv_treat_missing_irq_as_error(void)
{
#if defined(NV_LINUX_PCIE_MSI_SUPPORTED)
    return (nv_get_hypervisor_type() != OS_HYPERVISOR_HYPERV);
#else
    return NV_TRUE;
#endif
}

static void nv_get_pci_sysfs_config
(
    struct pci_dev *pci_dev,
    nv_linux_state_t *nvl
)
{
#if NV_FILESYSTEM_ACCESS_AVAILABLE
    char filename[50];
    int ret;

    ret = snprintf(filename, sizeof(filename),
                   "/sys/bus/pci/devices/%04x:%02x:%02x.0/config",
                   NV_PCI_DOMAIN_NUMBER(pci_dev),
                   NV_PCI_BUS_NUMBER(pci_dev),
                   NV_PCI_SLOT_NUMBER(pci_dev));
    if (ret > 0 && ret < sizeof(filename))
    {
        struct file *file = filp_open(filename, O_RDONLY, 0);
        if (!IS_ERR(file))
        {
#if defined(NV_SEQ_READ_ITER_PRESENT)
            /*
             * Sanity check for confirming if file path is mounted over
             * sysfs file system.
             */
            if ((file->f_inode != NULL) && (file->f_inode->i_sb != NULL) &&
                (strcmp(file->f_inode->i_sb->s_id, "sysfs") == 0))
            {
                struct seq_file *sf = file->private_data;

                /*
                 * Sanity check for confirming if 'file->private_data'
                 * actually points to 'struct seq_file'.
                 */
                if ((sf != NULL) && (sf->file == file) && (sf->op == NULL))
                {
                    struct kernfs_open_file *of = sf->private;

                    /*
                     * Sanity check for confirming if 'sf->private'
                     * actually points to 'struct kernfs_open_file'.
                     */
                    if ((of != NULL) && (of->file == file) &&
                        (of->seq_file == sf))
                    {
                        nvl->sysfs_config_file = file;
                    }
                }
            }

            if (nvl->sysfs_config_file == NULL)
            {
                filp_close(file, NULL);
            }
#else
            nvl->sysfs_config_file = file;
#endif
        }
    }
#endif
}

static int nv_resize_pcie_bars(struct pci_dev *pci_dev) {
    nv_printf(NV_DBG_INFO, "NVRM: Resizable BAR is not supported on this driver version.\n");
    return 0;
}

static void
nv_init_coherent_link_info
(
    nv_state_t *nv
)
{
    return;
}

static void nv_init_dynamic_power_management
(
    nvidia_stack_t *sp,
    struct pci_dev *pci_dev
)
{
    nv_linux_state_t *nvl = pci_get_drvdata(pci_dev);
    nv_state_t *nv = NV_STATE_PTR(nvl);
    NvBool pr3_acpi_method_present = NV_FALSE;

    nvl->sysfs_config_file = NULL;

    nv_get_pci_sysfs_config(pci_dev, nvl);

    if (nv_get_hypervisor_type() != OS_HYPERVISOR_UNKNOWN)
    {
        pr3_acpi_method_present = nv_acpi_power_resource_method_present(pci_dev);
    }
    else if (pci_dev->bus && pci_dev->bus->self)
    {
        pr3_acpi_method_present = nv_acpi_power_resource_method_present(pci_dev->bus->self);
    }

    // Support dynamic power management if device is a tegra PCI iGPU
    rm_init_tegra_dynamic_power_management(sp, nv);

    rm_init_dynamic_power_management(sp, nv, pr3_acpi_method_present);
}

static NvBool
nv_pci_validate_bars(const struct pci_dev *pci_dev, NvBool only_bar0)
{
    unsigned int i, j;
    NvBool last_bar_64bit = NV_FALSE;

    for (i = 0, j = 0; i < NVRM_PCICFG_NUM_BARS && j < NV_GPU_NUM_BARS; i++)
    {
        if (NV_PCI_RESOURCE_VALID(pci_dev, i))
        {
            if ((NV_PCI_RESOURCE_FLAGS(pci_dev, i) & PCI_BASE_ADDRESS_MEM_TYPE_64) &&
                (NV_PCI_RESOURCE_FLAGS(pci_dev, i) & PCI_BASE_ADDRESS_MEM_PREFETCH))
            {
                last_bar_64bit = NV_TRUE;
            }

            //
            // If we are here, then we have found a valid BAR -- 32 or 64-bit.
            //
            j++;

            if (only_bar0)
                return NV_TRUE;

            continue;
        }

        //
        // If last_bar_64bit is "true" then, we are looking at the 2nd (upper)
        // half of the 64-bit BAR. This is typically all 0s which looks invalid
        // but it's normal and not a problem and we can ignore it and continue.
        //
        if (last_bar_64bit)
        {
            last_bar_64bit = NV_FALSE;
            continue;
        }

        // Invalid 32 or 64-bit BAR.
        nv_printf(NV_DBG_ERRORS,
            "NVRM: This PCI I/O region assigned to your NVIDIA device is invalid:\n"
            "NVRM: BAR%d is %" NvU64_fmtu "M @ 0x%" NvU64_fmtx " (PCI:%04x:%02x:%02x.%x)\n", i,
            (NvU64)(NV_PCI_RESOURCE_SIZE(pci_dev, i) >> 20),
            (NvU64)NV_PCI_RESOURCE_START(pci_dev, i),
            NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
            NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));

        return NV_FALSE;
    }

    return NV_TRUE;
}

/* find nvidia devices and set initial state */
static int
nv_pci_probe
(
    struct pci_dev *pci_dev,
    const struct pci_device_id *id_table
)
{
    nv_state_t *nv = NULL;
    nv_linux_state_t *nvl = NULL;
    unsigned int i, j;
    int flags = 0;
    nvidia_stack_t *sp = NULL;
    NvBool prev_nv_ats_supported = nv_ats_supported;
    NV_STATUS status;
    NvU8 regs_bar_index = nv_bar_index_to_os_bar_index(pci_dev,
                                                       NV_GPU_BAR_INDEX_REGS);
    NvBool bar0_requested = NV_FALSE;

    nv_printf(NV_DBG_SETUP, "NVRM: probing 0x%x 0x%x, class 0x%x\n",
        pci_dev->vendor, pci_dev->device, pci_dev->class);

#ifdef NV_PCI_SRIOV_SUPPORT
    if (pci_dev->is_virtfn)
    {
#if defined(NV_VGPU_KVM_BUILD)
#if defined(NV_BUS_TYPE_HAS_IOMMU_OPS)
        if (pci_dev->dev.bus->iommu_ops == NULL) 
#else
        if ((pci_dev->dev.iommu != NULL) && (pci_dev->dev.iommu->iommu_dev != NULL) &&
            (pci_dev->dev.iommu->iommu_dev->ops == NULL))
#endif
        {
            nv_printf(NV_DBG_ERRORS, "NVRM: Aborting probe for VF %04x:%02x:%02x.%x "
                      "since IOMMU is not present on the system.\n",
                       NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                       NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));
            return -1;
        }

        return 0;
#else
        nv_printf(NV_DBG_ERRORS, "NVRM: Ignoring probe for VF %04x:%02x:%02x.%x ",
                  NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                  NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));

        return -1;
#endif /* NV_VGPU_KVM_BUILD */
    }
#endif /* NV_PCI_SRIOV_SUPPORT */

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return -1;
    }

    if (!rm_is_supported_pci_device(
                (pci_dev->class >> 16) & 0xFF,
                (pci_dev->class >> 8) & 0xFF,
                pci_dev->vendor,
                pci_dev->device,
                pci_dev->subsystem_vendor,
                pci_dev->subsystem_device,
                NV_FALSE /* print_legacy_warning */))
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: ignoring the legacy GPU %04x:%02x:%02x.%x\n",
                  NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                  NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));
        goto failed;
    }

    num_probed_nv_devices++;

    if (pci_enable_device(pci_dev) != 0)
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: pci_enable_device failed, aborting\n");
        goto failed;
    }

    if ((pci_dev->irq == 0 && !pci_find_capability(pci_dev, PCI_CAP_ID_MSIX))
        && nv_treat_missing_irq_as_error())
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: Can't find an IRQ for your NVIDIA card!\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: Please check your BIOS settings.\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: [Plug & Play OS] should be set to NO\n");
        nv_printf(NV_DBG_ERRORS, "NVRM: [Assign IRQ to VGA] should be set to YES \n");
        goto failed;
    }

    // Validate if BAR0 is usable
    if (!nv_pci_validate_bars(pci_dev, /* only_bar0 = */ NV_TRUE))
        goto failed;

    if (!request_mem_region(NV_PCI_RESOURCE_START(pci_dev, regs_bar_index),
                            NV_PCI_RESOURCE_SIZE(pci_dev, regs_bar_index),
                            nv_device_name))
    {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: request_mem_region failed for %" NvU64_fmtu "M @ 0x%" NvU64_fmtx ". This can\n"
            "NVRM: occur when a driver such as rivatv is loaded and claims\n"
            "NVRM: ownership of the device's registers.\n",
            (NvU64)(NV_PCI_RESOURCE_SIZE(pci_dev, regs_bar_index) >> 20),
            (NvU64)NV_PCI_RESOURCE_START(pci_dev, regs_bar_index));
        goto failed;
    }

    bar0_requested = NV_TRUE;

    NV_KZALLOC(nvl, sizeof(nv_linux_state_t));
    if (nvl == NULL)
    {
        nv_printf(NV_DBG_ERRORS, "NVRM: failed to allocate memory\n");
        goto err_not_supported;
    }

    nv  = NV_STATE_PTR(nvl);

    for (i = 0; i < NVRM_PCICFG_NUM_BARS; i++)
    {
        if ((NV_PCI_RESOURCE_VALID(pci_dev, i)) &&
            (NV_PCI_RESOURCE_FLAGS(pci_dev, i) & PCI_BASE_ADDRESS_SPACE)
                == PCI_BASE_ADDRESS_SPACE_MEMORY)
        {
            nv->bars[NV_GPU_BAR_INDEX_REGS].offset = NVRM_PCICFG_BAR_OFFSET(i);
            nv->bars[NV_GPU_BAR_INDEX_REGS].cpu_address = NV_PCI_RESOURCE_START(pci_dev, i);
            nv->bars[NV_GPU_BAR_INDEX_REGS].size = NV_PCI_RESOURCE_SIZE(pci_dev, i);

            break;
        }
    }
    nv->regs = &nv->bars[NV_GPU_BAR_INDEX_REGS];

    pci_set_drvdata(pci_dev, (void *)nvl);

    /* default to 32-bit PCI bus address space */
    pci_dev->dma_mask = 0xffffffffULL;

    nvl->dev               = &pci_dev->dev;
    nvl->pci_dev           = pci_dev;
    nvl->dma_dev.dev       = nvl->dev;

    nv->pci_info.vendor_id = pci_dev->vendor;
    nv->pci_info.device_id = pci_dev->device;
    nv->subsystem_id       = pci_dev->subsystem_device;
    nv->subsystem_vendor   = pci_dev->subsystem_vendor;
    nv->os_state           = (void *) nvl;
    nv->dma_dev            = &nvl->dma_dev;
    nv->pci_info.domain    = NV_PCI_DOMAIN_NUMBER(pci_dev);
    nv->pci_info.bus       = NV_PCI_BUS_NUMBER(pci_dev);
    nv->pci_info.slot      = NV_PCI_SLOT_NUMBER(pci_dev);
    nv->handle             = pci_dev;
    nv->flags             |= flags;

    if (!nv_lock_init_locks(sp, nv))
    {
        goto err_not_supported;
    }

    if ((rm_is_supported_device(sp, nv)) != NV_OK)
        goto err_not_supported;

    // Wire RM HAL
    if (!rm_init_private_state(sp, nv))
    {
        NV_DEV_PRINTF(NV_DBG_ERRORS, nv, "rm_init_private_state() failed!\n");
        goto err_zero_dev;
    }

    if (!nv->is_tegra_pci_igpu &&
        !pci_devid_is_self_hosted(pci_dev->device) &&
        !nv_pci_validate_bars(pci_dev, /* only_bar0 = */ NV_FALSE))
        goto err_zero_dev;

    if (nv_resize_pcie_bars(pci_dev)) {
        nv_printf(NV_DBG_ERRORS,
            "NVRM: Fatal Error while attempting to resize PCIe BARs.\n");
        goto err_zero_dev;
    }

    nvl->all_mappings_revoked = NV_TRUE;
    nvl->safe_to_mmap = NV_TRUE;
    nvl->gpu_wakeup_callback_needed = NV_TRUE;
    INIT_LIST_HEAD(&nvl->open_files);

    if (!nv->is_tegra_pci_igpu)
    {
        for (i = 0, j = 0; i < NVRM_PCICFG_NUM_BARS && j < NV_GPU_NUM_BARS; i++)
        {
            if (j == NV_GPU_BAR_INDEX_REGS)
            {
                j++;
                continue;
            }

            if ((NV_PCI_RESOURCE_VALID(pci_dev, i)) &&
                (NV_PCI_RESOURCE_FLAGS(pci_dev, i) & PCI_BASE_ADDRESS_SPACE)
                    == PCI_BASE_ADDRESS_SPACE_MEMORY)
            {
                nv->bars[j].offset = NVRM_PCICFG_BAR_OFFSET(i);
                nv->bars[j].cpu_address = NV_PCI_RESOURCE_START(pci_dev, i);
                nv->bars[j].size = NV_PCI_RESOURCE_SIZE(pci_dev, i);
                j++;
            }
        }
    }

    nv->fb   = &nv->bars[NV_GPU_BAR_INDEX_FB];
    nv->interrupt_line = pci_dev->irq;

    NV_ATOMIC_SET(nvl->numa_info.status, NV_IOCTL_NUMA_STATUS_DISABLED);
    nvl->numa_info.node_id = NUMA_NO_NODE;

    if (pci_devid_is_self_hosted(pci_dev->device))
    {
        nv_init_coherent_link_info(nv);
    }

    nv_ats_supported |= nv->ats_support;

    nv_clk_get_handles(nv);

    pci_set_master(pci_dev);

#if defined(CONFIG_VGA_ARB)
#if defined(VGA_DEFAULT_DEVICE)
#if defined(NV_VGA_TRYGET_PRESENT)
    vga_tryget(VGA_DEFAULT_DEVICE, VGA_RSRC_LEGACY_MASK);
#endif
#endif
    vga_set_legacy_decoding(pci_dev, VGA_RSRC_NONE);
#endif

    status = nv_check_gpu_state(nv);
    if (status == NV_ERR_GPU_IS_LOST)
    {
        NV_DEV_PRINTF(NV_DBG_INFO, nv, "GPU is lost, skipping nv_pci_probe\n");
        goto err_gpu_lost;
    }

    nv->cpu_numa_node_id = dev_to_node(nvl->dev);

    if (nv_linux_init_open_q(nvl) != 0)
    {
        NV_DEV_PRINTF(NV_DBG_ERRORS, nv, "nv_linux_init_open_q() failed!\n");
        goto err_gpu_lost;
    }

    nv_printf(NV_DBG_INFO,
              "NVRM: PCI:%04x:%02x:%02x.%x (%04x:%04x): BAR0 @ 0x%llx (%lluMB)\n",
              nv->pci_info.domain, nv->pci_info.bus, nv->pci_info.slot,
              PCI_FUNC(pci_dev->devfn), nv->pci_info.vendor_id, nv->pci_info.device_id,
              nv->regs->cpu_address, (nv->regs->size >> 20));
    nv_printf(NV_DBG_INFO,
              "NVRM: PCI:%04x:%02x:%02x.%x (%04x:%04x): BAR1 @ 0x%llx (%lluMB)\n",
              nv->pci_info.domain, nv->pci_info.bus, nv->pci_info.slot,
              PCI_FUNC(pci_dev->devfn), nv->pci_info.vendor_id, nv->pci_info.device_id,
              nv->fb->cpu_address, (nv->fb->size >> 20));

    num_nv_devices++;

    /*
     * The newly created nvl object is added to the nv_linux_devices global list
     * only after all the initialization operations for that nvl object are
     * completed, so as to protect against simultaneous lookup operations which
     * may discover a partially initialized nvl object in the list
     */
    LOCK_NV_LINUX_DEVICES();

    if (nv_linux_add_device_locked(nvl) != 0)
    {
        UNLOCK_NV_LINUX_DEVICES();
        goto err_add_device;
    }

    UNLOCK_NV_LINUX_DEVICES();

    pm_vt_switch_required(nvl->dev, NV_TRUE);

    nv_init_dynamic_power_management(sp, pci_dev);


    nv_procfs_add_gpu(nvl);

    /* Parse and set any per-GPU registry keys specified. */
    nv_parse_per_device_option_string(sp);

    rm_set_rm_firmware_requested(sp, nv);

#if defined(NV_VGPU_KVM_BUILD)
    if (nvidia_vgpu_vfio_probe(nvl->pci_dev) != NV_OK)
    {
        NV_DEV_PRINTF(NV_DBG_ERRORS, nv, "Failed to register device to vGPU VFIO module");
        goto err_free_all;
    }
#endif

    nv_check_and_exclude_gpu(sp, nv);

#if defined(DPM_FLAG_NO_DIRECT_COMPLETE)
    dev_pm_set_driver_flags(nvl->dev, DPM_FLAG_NO_DIRECT_COMPLETE);
#elif defined(DPM_FLAG_NEVER_SKIP)
    dev_pm_set_driver_flags(nvl->dev, DPM_FLAG_NEVER_SKIP);
#endif

    /*
     * Dynamic power management should be enabled as the last step.
     * Kernel runtime power management framework can put the device
     * into the suspended state. Hardware register access should not be done
     * after enabling dynamic power management.
     */
    rm_enable_dynamic_power_management(sp, nv);

    /*
     * This must be the last action in nv_pci_probe(). Do not add code after this line.
     */
    rm_notify_gpu_addition(sp, nv);

    nv_kmem_cache_free_stack(sp);

    nvidia_modeset_probe(nvl);

    return 0;

goto err_free_all;
err_free_all:
    nv_procfs_remove_gpu(nvl);
    rm_cleanup_dynamic_power_management(sp, nv);
    pm_vt_switch_unregister(nvl->dev);
    LOCK_NV_LINUX_DEVICES();
    nv_linux_remove_device_locked(nvl);
    UNLOCK_NV_LINUX_DEVICES();
err_add_device:
    nv_linux_stop_open_q(nvl);
err_gpu_lost:
    nv_clk_clear_handles(nv);
    nv_ats_supported = prev_nv_ats_supported;
err_zero_dev:
    rm_free_private_state(sp, nv);
err_not_supported:
    nv_lock_destroy_locks(sp, nv);
failed:
    if (bar0_requested)
    {
        release_mem_region(NV_PCI_RESOURCE_START(pci_dev, regs_bar_index),
                           NV_PCI_RESOURCE_SIZE(pci_dev, regs_bar_index));
    }
    NV_PCI_DISABLE_DEVICE(pci_dev);
    pci_set_drvdata(pci_dev, NULL);
    if (nvl != NULL)
    {
        NV_KFREE(nvl, sizeof(nv_linux_state_t));
    }
    nv_kmem_cache_free_stack(sp);
    return -1;
}

static void
nv_pci_remove(struct pci_dev *pci_dev)
{
    nv_linux_state_t *nvl = NULL;
    nv_state_t *nv;
    nvidia_stack_t *sp = NULL;
    NvU8 regs_bar_index = nv_bar_index_to_os_bar_index(pci_dev,
                                                       NV_GPU_BAR_INDEX_REGS);

    nv_printf(NV_DBG_SETUP, "NVRM: removing GPU %04x:%02x:%02x.%x\n",
              NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
              NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));

#ifdef NV_PCI_SRIOV_SUPPORT
    if (pci_dev->is_virtfn)
    {
#if defined(NV_VGPU_KVM_BUILD)
        /* Arg 2 == NV_TRUE means that the PCI device should be removed */
        nvidia_vgpu_vfio_remove(pci_dev, NV_TRUE);
#endif /* NV_VGPU_KVM_BUILD */
        return;
    }
#endif /* NV_PCI_SRIOV_SUPPORT */

    if (nv_kmem_cache_alloc_stack(&sp) != 0)
    {
        return;
    }

    nvl = pci_get_drvdata(pci_dev);
    if (!nvl || (nvl->pci_dev != pci_dev))
    {
        nv_kmem_cache_free_stack(sp);
        return;
    }

    nv = NV_STATE_PTR(nvl);

    /*
     * Flush and stop open_q before proceeding with removal to ensure nvl
     * outlives all enqueued work items.
     */
    nv_linux_stop_open_q(nvl);

    nvidia_modeset_remove(nv->gpu_id);

    LOCK_NV_LINUX_DEVICES();
    down(&nvl->ldata_lock);
    nv->flags |= NV_FLAG_PCI_REMOVE_IN_PROGRESS;

    rm_notify_gpu_removal(sp, nv);

    /*
     * Sanity check: A removed device shouldn't have a non-zero usage_count.
     * For eGPU, fall off the bus along with clients active is a valid scenario.
     * Hence skipping the sanity check for eGPU.
     */
    if ((atomic64_read(&nvl->usage_count) != 0) && !(nv->is_external_gpu))
    {
        nv_printf(NV_DBG_ERRORS,
                  "NVRM: Attempting to remove device %04x:%02x:%02x.%x with non-zero usage count!\n",
                  NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                  NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));

        /*
         * We can't return from this function without corrupting state, so we wait for
         * the usage count to go to zero.
         */
        while (atomic64_read(&nvl->usage_count) != 0)
        {

            /*
             * While waiting, release the locks so that other threads can make
             * forward progress.
             */
            up(&nvl->ldata_lock);
            UNLOCK_NV_LINUX_DEVICES();

            os_delay(500);

            /* Re-acquire the locks before checking again */
            LOCK_NV_LINUX_DEVICES();
            nvl = pci_get_drvdata(pci_dev);
            if (!nvl)
            {
                /* The device was not found, which should not happen */
                nv_printf(NV_DBG_ERRORS,
                          "NVRM: Failed removal of device %04x:%02x:%02x.%x!\n",
                          NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                          NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));  
                WARN_ON(1);
                goto done;
            }
            nv = NV_STATE_PTR(nvl);
            down(&nvl->ldata_lock);
        }

        nv_printf(NV_DBG_ERRORS,
                  "NVRM: Continuing with GPU removal for device %04x:%02x:%02x.%x\n",
                  NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
                  NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn));
    }

    rm_check_for_gpu_surprise_removal(sp, nv);

    nv_linux_remove_device_locked(nvl);

    /* Remove proc entry for this GPU */
    nv_procfs_remove_gpu(nvl);

    nv_clk_clear_handles(nv);

    rm_cleanup_dynamic_power_management(sp, nv);

    nv->removed = NV_TRUE;

    UNLOCK_NV_LINUX_DEVICES();

    pm_vt_switch_unregister(&pci_dev->dev);

#if defined(NV_VGPU_KVM_BUILD)
    /* Arg 2 == NV_TRUE means that the PCI device should be removed */
    nvidia_vgpu_vfio_remove(pci_dev, NV_TRUE);
#endif

    if ((nv->flags & NV_FLAG_PERSISTENT_SW_STATE) || (nv->flags & NV_FLAG_OPEN))
    {
        nv_acpi_unregister_notifier(nvl);
        if (nv->flags & NV_FLAG_PERSISTENT_SW_STATE)
        {
            rm_disable_gpu_state_persistence(sp, nv);
        }
        nv_shutdown_adapter(sp, nv, nvl);
        nv_dev_free_stacks(nvl);
    }

    if (nvl->sysfs_config_file != NULL)
    {
        filp_close(nvl->sysfs_config_file, NULL);
        nvl->sysfs_config_file = NULL;
    }

    if (atomic64_read(&nvl->usage_count) == 0)
    {
        nv_lock_destroy_locks(sp, nv);
    }

    num_probed_nv_devices--;

    pci_set_drvdata(pci_dev, NULL);

    rm_i2c_remove_adapters(sp, nv);
    rm_free_private_state(sp, nv);
    release_mem_region(NV_PCI_RESOURCE_START(pci_dev, regs_bar_index),
                       NV_PCI_RESOURCE_SIZE(pci_dev, regs_bar_index));

    num_nv_devices--;

    if (atomic64_read(&nvl->usage_count) == 0)
    {
        NV_PCI_DISABLE_DEVICE(pci_dev);
        NV_KFREE(nvl, sizeof(nv_linux_state_t));
    }
    else
    {
        up(&nvl->ldata_lock);
    }

    nv_kmem_cache_free_stack(sp);
    return;

done:
    UNLOCK_NV_LINUX_DEVICES();
    nv_kmem_cache_free_stack(sp);
}

static void
nv_pci_shutdown(struct pci_dev *pci_dev)
{
    nv_linux_state_t *nvl = pci_get_drvdata(pci_dev);

    if (nvl != NULL)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);

        if (nvl->is_forced_shutdown)
        {
            nvl->is_forced_shutdown = NV_FALSE;
            return;
        }

        nvidia_modeset_remove(nv->gpu_id);

        nvl->nv_state.is_shutdown = NV_TRUE;
    }

    /* pci_clear_master is not defined for !CONFIG_PCI */
#ifdef CONFIG_PCI
    pci_clear_master(pci_dev);
#endif

    /* SHH HW mandates 1us delay to realise the effects of
     * Bus Mater Enable(BME) disable. Adding 1us delay for
     * all the chips as the delay is not in the data path
     * and not big. Creating HAL for this would be a overkill.
     */
    udelay(1);
}

/*!
 * @brief This function accepts pci information corresponding to a GPU
 * and returns a reference to the nv_linux_state_t corresponding to that GPU.
 *
 * @param[in] domain            Pci domain number for the GPU to be found.
 * @param[in] bus               Pci bus number for the GPU to be found.
 * @param[in] slot              Pci slot number for the GPU to be found.
 * @param[in] function          Pci function number for the GPU to be found.
 *
 * @return Pointer to nv_linux_state_t for the GPU if it is found, or NULL otherwise.
 */
nv_linux_state_t * find_pci(NvU32 domain, NvU8 bus, NvU8 slot, NvU8 function)
{
    nv_linux_state_t *nvl = NULL;

    LOCK_NV_LINUX_DEVICES();

    for (nvl = nv_linux_devices; nvl != NULL; nvl = nvl->next)
    {
        nv_state_t *nv = NV_STATE_PTR(nvl);

        if (nv->pci_info.domain == domain &&
            nv->pci_info.bus == bus &&
            nv->pci_info.slot == slot &&
            nv->pci_info.function == function)
        {
            break;
        }
    }

    UNLOCK_NV_LINUX_DEVICES();
    return nvl;
}

int nvidia_dev_get_pci_info(const NvU8 *uuid, struct pci_dev **pci_dev_out,
    NvU64 *dma_start, NvU64 *dma_limit)
{
    nv_linux_state_t *nvl;

    /* Takes nvl->ldata_lock */
    nvl = find_uuid(uuid);
    if (!nvl)
        return -ENODEV;

    *pci_dev_out = nvl->pci_dev;
    *dma_start = nvl->dma_dev.addressable_range.start;
    *dma_limit = nvl->dma_dev.addressable_range.limit;

    up(&nvl->ldata_lock);

    return 0;
}

NvU8 nv_find_pci_capability(struct pci_dev *pci_dev, NvU8 capability)
{
    u16 status = 0;
    u8  cap_ptr = 0, cap_id = 0xff;

    pci_read_config_word(pci_dev, PCI_STATUS, &status);
    status &= PCI_STATUS_CAP_LIST;
    if (!status)
        return 0;

    switch (pci_dev->hdr_type) {
        case PCI_HEADER_TYPE_NORMAL:
        case PCI_HEADER_TYPE_BRIDGE:
            pci_read_config_byte(pci_dev, PCI_CAPABILITY_LIST, &cap_ptr);
            break;
        default:
            return 0;
    }

    do {
        cap_ptr &= 0xfc;
        pci_read_config_byte(pci_dev, cap_ptr + PCI_CAP_LIST_ID, &cap_id);
        if (cap_id == capability)
            return cap_ptr;
        pci_read_config_byte(pci_dev, cap_ptr + PCI_CAP_LIST_NEXT, &cap_ptr);
    } while (cap_ptr && cap_id != 0xff);

    return 0;
}

static void check_for_bound_driver(struct pci_dev *pci_dev)
{
    if (pci_dev->dev.driver)
    {
        const char *driver_name = pci_dev->dev.driver->name;

        nv_printf(NV_DBG_WARNINGS, "NVRM: GPU %04x:%02x:%02x.%x is already "
            "bound to %s.\n",
            NV_PCI_DOMAIN_NUMBER(pci_dev), NV_PCI_BUS_NUMBER(pci_dev),
            NV_PCI_SLOT_NUMBER(pci_dev), PCI_FUNC(pci_dev->devfn),
            driver_name ? driver_name : "another driver"
        );
    }
}

/* make sure the pci_driver called probe for all of our devices.
 * we've seen cases where rivafb claims the device first and our driver
 * doesn't get called.
 */
int
nv_pci_count_devices(void)
{
    struct pci_dev *pci_dev;
    int count = 0;

    if (NVreg_RegisterPCIDriver == 0)
    {
        return 0;
    }

    pci_dev = pci_get_class(PCI_CLASS_DISPLAY_VGA << 8, NULL);
    while (pci_dev)
    {
        if (rm_is_supported_pci_device(
                PCI_BASE_CLASS_DISPLAY,
                PCI_CLASS_DISPLAY_VGA & 0xFF,
                pci_dev->vendor,
                pci_dev->device,
                pci_dev->subsystem_vendor,
                pci_dev->subsystem_device,
                NV_TRUE /* print_legacy_warning */))
        {
            check_for_bound_driver(pci_dev);
            count++;
        }
        pci_dev = pci_get_class(PCI_CLASS_DISPLAY_VGA << 8, pci_dev);
    }

    pci_dev = pci_get_class(PCI_CLASS_DISPLAY_3D << 8, NULL);
    while (pci_dev)
    {
        if (rm_is_supported_pci_device(
                (pci_dev->class >> 16) & 0xFF,
                (pci_dev->class >> 8) & 0xFF,
                pci_dev->vendor,
                pci_dev->device,
                pci_dev->subsystem_vendor,
                pci_dev->subsystem_device,
                NV_TRUE /* print_legacy_warning */))
        {
            check_for_bound_driver(pci_dev);
            count++;
        }
        pci_dev = pci_get_class(PCI_CLASS_DISPLAY_3D << 8, pci_dev);
    }

    return count;
}

/*
 * On coherent platforms that support BAR1 mappings for GPUDirect RDMA,
 * dma-buf and nv-p2p subsystems need to ensure the 2 devices belong to
 * the same IOMMU group.
 */
NvBool nv_pci_is_valid_topology_for_direct_pci(
    nv_state_t     *nv,
    struct pci_dev *peer
)
{
    struct pci_dev *pdev0 = to_pci_dev(nv->dma_dev->dev);
    struct pci_dev *pdev1 = peer;

    if (!nv->coherent)
    {
        return NV_FALSE;
    }

    switch (NVreg_GrdmaPciTopoCheckOverride) {
        case NV_REG_GRDMA_PCI_TOPO_CHECK_OVERRIDE_ALLOW_ACCESS:
            return NV_TRUE;
        case NV_REG_GRDMA_PCI_TOPO_CHECK_OVERRIDE_DENY_ACCESS:
            return NV_FALSE;
        default:
           return (pdev0->dev.iommu_group == pdev1->dev.iommu_group);
    }
}

NvBool nv_pci_has_common_pci_switch(
    nv_state_t     *nv,
    struct pci_dev *peer
)
{
    struct pci_dev *pci_dev0, *pci_dev1;

    pci_dev0 = pci_upstream_bridge(to_pci_dev(nv->dma_dev->dev));

    while (pci_dev0 != NULL)
    {
        pci_dev1 = pci_upstream_bridge(peer);

        while (pci_dev1 != NULL)
        {
            if (pci_dev0 == pci_dev1)
                return NV_TRUE;

            pci_dev1 = pci_upstream_bridge(pci_dev1);
        }

        pci_dev0 = pci_upstream_bridge(pci_dev0);
    }

    return NV_FALSE;
}

NvBool NV_API_CALL nv_grdma_pci_topology_supported(
    nv_state_t      *nv,
    nv_dma_device_t *dma_peer
)
{
    //
    // Skip topo check on coherent platforms since
    // NIC can map over C2C anyway and PCIe topology shouldn't matter.
    //
    if (nv->coherent)
    {
        return NV_TRUE;
    }

    switch (NVreg_GrdmaPciTopoCheckOverride)
    {
        case NV_REG_GRDMA_PCI_TOPO_CHECK_OVERRIDE_ALLOW_ACCESS:
            return NV_TRUE;
        case NV_REG_GRDMA_PCI_TOPO_CHECK_OVERRIDE_DENY_ACCESS:
            return NV_FALSE;
        default:
            break;
    }

    // Allow RDMA by default on passthrough VMs.
    if ((nv->flags & NV_FLAG_PASSTHRU) != 0)
        return NV_TRUE;

    //
    // Only allow RDMA on unsupported chipsets if there exists
    // a common PCI switch between the GPU and the other device
    //
    if ((nv->flags & NV_FLAG_PCI_P2P_UNSUPPORTED_CHIPSET) != 0)
        return nv_pci_has_common_pci_switch(nv, to_pci_dev(dma_peer->dev));

    return NV_TRUE;
}

#if defined(CONFIG_PM)
extern struct dev_pm_ops nv_pm_ops;
#endif

struct pci_driver nv_pci_driver = {
    .name      = MODULE_NAME,
    .id_table  = nv_pci_table,
    .probe     = nv_pci_probe,
    .remove    = nv_pci_remove,
    .shutdown  = nv_pci_shutdown,
#if defined(NV_USE_VFIO_PCI_CORE) && \
  defined(NV_PCI_DRIVER_HAS_DRIVER_MANAGED_DMA)
    .driver_managed_dma = NV_TRUE,
#endif
#if defined(CONFIG_PM)
    .driver.pm = &nv_pm_ops,
#endif
};

void nv_pci_unregister_driver(void)
{
    if (NVreg_RegisterPCIDriver == 0)
    {
        return;
    }
    return pci_unregister_driver(&nv_pci_driver);
}

int nv_pci_register_driver(void)
{
    if (NVreg_RegisterPCIDriver == 0)
    {
        return 0;
    }
    return pci_register_driver(&nv_pci_driver);
}
