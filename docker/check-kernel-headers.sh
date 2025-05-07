#!/bin/bash

#!/bin/bash

set -e

KERNEL_VER=$(uname -r)

echo "Detected kernel version: $KERNEL_VER"

# Skip install if headers already exist
if [ -d /lib/modules/$KERNEL_VER/build ]; then
    echo "Kernel headers already present at /lib/modules/$KERNEL_VER/build"
    exit 0
fi

# Function to check if a command exists
has_cmd() {
    command -v "$1" &>/dev/null
}

# Install kernel headers based on detected OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "Detected OS: $ID"

    case "$ID" in
        ubuntu|debian)
            sudo apt update
            sudo apt install -y "linux-headers-$KERNEL_VER"
            ;;
        fedora)
            sudo dnf install -y "kernel-devel-$KERNEL_VER"
            ;;
        centos|rhel)
            sudo yum install -y "kernel-devel-$KERNEL_VER"
            ;;
        arch)
            sudo pacman -Syu --noconfirm
            sudo pacman -S --noconfirm "linux-headers"
            ;;
        opensuse*|sles)
            sudo zypper install -y "kernel-default-devel"
            ;;
        *)
            echo "Unsupported distribution: $ID"
            exit 1
            ;;
    esac
else
    echo "/etc/os-release not found. Unable to detect distribution."
    exit 1
fi

# Confirm installation
if [ -d /lib/modules/$KERNEL_VER/build ]; then
    echo "Kernel headers for $KERNEL_VER installed successfully."
else
    echo "Failed to install kernel headers for $KERNEL_VER."
    exit 1
fi

