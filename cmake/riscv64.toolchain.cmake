# Copyright (c) 2024 SiFive, Inc. All rights reserved.
# Copyright (c) 2024, Phoebe Chen <phoebe.chen@sifive.com>
# Licensed under the MIT License.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES RISCV_TOOLCHAIN_ROOT)

if(NOT RISCV_TOOLCHAIN_ROOT)
  message(FATAL_ERROR "RISCV_TOOLCHAIN_ROOT is not defined. Please set the RISCV_TOOLCHAIN_ROOT variable.")
endif()

# Detect toolchain triple: SiFive xpacks use "riscv64-unknown-linux-gnu-",
# Debian/Ubuntu and most Linux distros use "riscv64-linux-gnu-".
if(EXISTS "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc")
  set(_RISCV_TRIPLE "riscv64-unknown-linux-gnu")
elseif(EXISTS "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-linux-gnu-gcc")
  set(_RISCV_TRIPLE "riscv64-linux-gnu")
else()
  message(FATAL_ERROR "No riscv64 gcc found under ${RISCV_TOOLCHAIN_ROOT}/bin/ "
                      "(looked for riscv64-unknown-linux-gnu-gcc and riscv64-linux-gnu-gcc)")
endif()

set(CMAKE_C_COMPILER   "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-gcc")
set(CMAKE_ASM_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-gcc")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/${_RISCV_TRIPLE}-g++")

# Sysroot: prefer RISCV_SYSROOT env/cmake var; fall back to legacy "${ROOT}/sysroot" layout.
if(DEFINED ENV{RISCV_SYSROOT} AND NOT RISCV_SYSROOT)
  set(RISCV_SYSROOT "$ENV{RISCV_SYSROOT}")
endif()
if(NOT RISCV_SYSROOT)
  set(RISCV_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")
endif()

set(CMAKE_FIND_ROOT_PATH ${RISCV_SYSROOT} ${RISCV_TOOLCHAIN_ROOT})
set(CMAKE_SYSROOT "${RISCV_SYSROOT}")
set(CMAKE_INCLUDE_PATH "${RISCV_SYSROOT}/usr/include/")
set(CMAKE_LIBRARY_PATH "${RISCV_SYSROOT}/usr/lib/")
set(CMAKE_PROGRAM_PATH "${RISCV_SYSROOT}/usr/bin/")

if(RISCV_QEMU_PATH)
  message(STATUS "RISCV_QEMU_PATH=${RISCV_QEMU_PATH} is defined during compilation.")
  set(CMAKE_CROSSCOMPILING_EMULATOR "${RISCV_QEMU_PATH};-L;${CMAKE_SYSROOT}")
endif()

set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

