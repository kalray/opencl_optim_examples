#
# Copyright (C) 2021 Kalray SA. All rights reserved.
#

arch := kv3-1
cluster-system := cos
use-module := strict-flags opencl_linux opencl-kernel

# Common flags
cflags += -DCL_TARGET_OPENCL_VERSION=200 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_HPP_TARGET_OPENCL_VERSION=200
cflags += -g3 -O2 -fPIC -fno-exceptions

HAVE_FAST_MATH ?= 1
OUTPUT_DISPLAY ?= 1
OUTPUT_IMAGE_TO_DISK ?= 1
PERF_TRACKING ?= 0
INPUT_IMAGE_PNG ?= ./images/valve_original.png

ifeq ($(OUTPUT_IMAGE_TO_DISK),0)
  OUTPUT_DISPLAY := 0
endif

ifneq ($(HAVE_FAST_MATH),0)
  fastmath_cflags := -ffast-math
endif

all:


# MPPA OpenCL pocl kernel
sobel_kernel-name := sobel.cl.pocl
sobel_kernel-srcs := sobel.cl
opencl-kernel-bin += sobel_kernel

# Host acceleration binary
host_app-srcs := ocl_utils.c png_utils.c host.c
host_app-cflags := -std=gnu99 -I. \
                   -DPERF_TRACKING=${PERF_TRACKING} \
                   -DOUTPUT_DISPLAY=$(OUTPUT_DISPLAY) \
                   -DOUTPUT_IMAGE_TO_DISK=$(OUTPUT_IMAGE_TO_DISK) \
                   -DHAVE_FAST_MATH=$(HAVE_FAST_MATH)
host_app-lflags := -lpng
host_app-deps := sobel_kernel
host-bin := host_app


run_hw: all
	timeout 300 $(O)/bin/host_app $(INPUT_IMAGE_PNG)


include $(KALRAY_TOOLCHAIN_DIR)/share/make/Makefile.kalray
