/*
 * mlas_riscv.h — MLAS RISC-V Dispatch API
 *
 * Defines the unified dispatch table for injecting hardware-optimized
 * kernel implementations (RVV, IME, etc.) into ORT MLAS at runtime.
 *
 * Architecture: Mesa Gallium-style function pointer table.
 * - External libraries populate MLAS_RISCV_DISPATCH with optimized kernels
 * - Single atomic registration via MlasRiscvSetDispatch()
 * - NULL fields = keep scalar default
 * - Capability detection at library load time
 *
 * This header is self-contained — no ORT internal headers required.
 * Both ORT (consumer) and external kernel libraries (producer) include it.
 *
 * License: MIT
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dispatch table version — bump on incompatible struct layout changes */
#define MLAS_RISCV_DISPATCH_VERSION 1

/* Capability flags for runtime feature detection */
#define MLAS_RISCV_CAP_RVV       (1u << 0)  /* Standard V extension (RVV 1.0) */
#define MLAS_RISCV_CAP_ZVFH      (1u << 1)  /* Vector FP16 (Zvfh/Zvfhmin) */
#define MLAS_RISCV_CAP_ZVBB      (1u << 2)  /* Vector bit manipulation */
#define MLAS_RISCV_CAP_IME       (1u << 3)  /* SpacemiT Integer Matrix Extension (vmadotu) */
/* Future: Zve32x, vendor matrix extensions, XTheadVector, etc. */

/*
 * Unified dispatch table for RISC-V MLAS kernels.
 *
 * Function pointer fields use void* to avoid exposing MLAS internal types.
 * ORT casts to the correct internal type with compile-time size validation.
 * Dispatch struct pointers (INT8/FP16/INT4) point to caller-owned static
 * structs that must remain valid for the process lifetime.
 */
struct MLAS_RISCV_DISPATCH {
    /* ── Header ── */
    uint32_t Version;        /* Must be MLAS_RISCV_DISPATCH_VERSION */
    uint32_t Capabilities;   /* Bitmask of MLAS_RISCV_CAP_* */
    uint32_t Vlen;           /* VLEN in bits (e.g., 256 for K1 X60) */
    uint32_t _pad0;          /* Alignment padding */

    /* ── FP32 SGEMM ── */
    /* Signature: size_t (*)(const float* A, const float* B, float* C,
     *   size_t CountK, size_t CountM, size_t CountN,
     *   size_t lda, size_t ldc, float alpha, bool ZeroMode) */
    void* GemmFloatKernel;

    /* ── INT8 Quantized GEMM dispatch structs ── */
    /* Each points to a MLAS_GEMM_QUANT_DISPATCH struct (defined in qgemm.h) */
    const void* GemmU8S8Dispatch;
    const void* GemmS8S8Dispatch;
    const void* GemmU8U8Dispatch;

    /* ── FP16 Half GEMM ── */
    /* Points to MLAS_HGEMM_DISPATCH (halfgemm.h) */
    const void* HGemmDispatch;

    /* ── INT4 Quantized N-Bit GEMM ── */
    /* Points to MLAS_QNBIT_GEMM_DISPATCH (qnbitgemm.h) */
    const void* QNBitGemmDispatch;

    /* ── Softmax / Reduction kernels ── */
    /* float (*)(const float* Input, size_t N) */
    void* ReduceMaximumF32Kernel;
    /* float (*)(const float* Input, float* Output, size_t N, const float* NegMax) */
    void* ComputeSumExpF32Kernel;
    /* void (*)(float* Output, size_t N, const float* Parameters) */
    void* ComputeSoftmaxOutputF32Kernel;
    /* void (*)(const float* Input, float* Output, size_t N, const float* Parameters) */
    void* ComputeLogSoftmaxOutputF32Kernel;

    /* ── Activation kernels ── */
    /* All: void (*)(const float* Input, float* Output, size_t N) */
    void* ErfKernelRoutine;
    void* ComputeExpF32Kernel;
    void* LogisticKernelRoutine;
    void* TanhKernelRoutine;

    /* ── Quantization ── */
    void* QuantizeLinearS8Kernel;
    void* QuantizeLinearU8Kernel;

    /* ── ABI extension padding ── */
    void* _reserved[6];
};

/*
 * Register a RISC-V dispatch table with ORT MLAS.
 *
 * Called once from the external library's __attribute__((constructor)).
 * NULL fields in the dispatch table are skipped (scalar default preserved).
 *
 * Returns 0 on success, -1 on version mismatch or invalid argument.
 *
 * Thread safety: Must be called before any ORT inference (typically at
 * shared library load time, before main()).
 */
int MlasRiscvSetDispatch(const struct MLAS_RISCV_DISPATCH* Dispatch);

#ifdef __cplusplus
}
#endif
