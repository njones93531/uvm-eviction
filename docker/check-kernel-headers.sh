#!/bin/bash
set -e

KERNEL_VER=$(uname -r)

echo "Detected kernel version: $KERNEL_VER"

# Verify that the build symlink exists
if [ -e /lib/modules/$KERNEL_VER/build/Makefile ]; then
    echo "Kernel headers already present at /lib/modules/$KERNEL_VER/build"
    exit 0
fi

echo "ERROR: Missing kernel headers for $KERNEL_VER."
echo "Please install kernel headers on the host system before running this setup."
echo ""
echo "Examples:"
echo "  Ubuntu/Debian: sudo apt install linux-headers-$(uname -r)"
echo "  Fedora/RHEL:   sudo dnf install kernel-devel-$(uname -r)"
echo "  CentOS:        sudo yum install kernel-devel-$(uname -r)"
echo "  Arch:          sudo pacman -S linux-headers"
echo "  openSUSE:      sudo zypper install kernel-default-devel"
exit 1

