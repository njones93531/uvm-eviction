/*******************************************************************************
    Copyright (c) 2016-2023 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_hmm.h"

// Support for HMM ( https://docs.kernel.org/mm/hmm.html ):

#ifdef NVCPU_X86_64
static bool uvm_disable_hmm = false;
MODULE_PARM_DESC(uvm_disable_hmm,
                 "Force-disable HMM functionality in the UVM driver. "
                 "Default: false (HMM is enabled if possible). "
                 "However, even with uvm_disable_hmm=false, HMM will not be "
                 "enabled if is not supported in this driver build "
                 "configuration, or if ATS settings conflict with HMM.");
#else
// So far, we've only tested HMM on x86_64, so disable it by default everywhere
// else.
static bool uvm_disable_hmm = true;
MODULE_PARM_DESC(uvm_disable_hmm,
                 "Force-disable HMM functionality in the UVM driver. "
                 "Default: true (HMM is not enabled on this CPU architecture). "
                 "However, even with uvm_disable_hmm=false, HMM will not be "
                 "enabled if is not supported in this driver build "
                 "configuration, or if ATS settings conflict with HMM.");
#endif

module_param(uvm_disable_hmm, bool, 0444);

