/*
 * qgemm_kernel_ime.cpp — IME vmadot INT8 GEMM kernel for MLAS
 *
 * ═════════════════════════════════════════════════════════════════════
 * ⚠️  NOT UPSTREAM — SpacemiT proprietary vmadot instruction
 *
 * This file uses `vmadot` (Integer Matrix Extension, 4×4×8 INT8 MAC),
 * a SpacemiT-specific custom RISC-V instruction NOT part of the RVV 1.0
 * standard. It requires SpacemiT GCC/binutils with `-march=rv64gcv_xsmtvdotii`
 * and will fail to assemble on upstream toolchains.
 *
 * This branch (riscv64/mlas-ime-spacemit-NOT-UPSTREAM) exists only for
 * internal K3/K1 benchmarking and product integration. DO NOT submit this
 * file to microsoft/onnxruntime via PR — upstream only accepts standard
 * RVV code (see qgemm_kernel_rvv.cpp in the sibling branch).
 * ═════════════════════════════════════════════════════════════════════
 *
 * Drop-in replacement for qgemm_kernel_rvv.cpp's MlasGemmQuantKernel.
 * Uses SpacemiT IME vmadot instruction (4×4×8 INT8 matrix multiply-accumulate).
 *
 * Compilation: SpacemiT GCC -march=rv64gcv_xsmtvdotii
 *
 * MLAS interface:
 *   A: packed row-major [M, AlignedK] uint8_t
 *   B: packed column-major [N, AlignedK] uint8_t (from CopyPackB)
 *   C: row-major [M, ldc] int32_t output
 *
 * vmadot data layout:
 *   A_tile: (4, 8) row-major contiguous 32 bytes
 *   B_tile: (4, 8) = 4 columns × 8 K elements, column-major 32 bytes
 *            B_tile[n*8+k] for n=0..3, k=0..7
 *
 * Strategy: MLAS packs B as [N, AlignedK]. For each 4-column group,
 * repack 4 columns × 8 K elements into vmadot's B layout on the fly.
 */

#include "mlasi.h"
#include "qgemm.h"
#include <string.h>

/* IME kernel uses same packing/strides as RVV kernel */
struct MLAS_GEMM_QUANT_KERNEL_IME
{
    typedef uint8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef uint8_t OffsetAType;
    typedef uint8_t OffsetBType;
    /* PackedK=4 to match MLAS framework (AlignedK = PackedCountK * 4) */
    static constexpr size_t PackedK = 4;
    /* StrideN=128 processed in groups of 4 by vmadot */
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 4, 128, 256 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 4, 128, 256 };
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_IME::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_IME::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_IME::PackedStrides;

/* Reuse RVV kernel's zero-point fixup and CopyPack routines */
template<>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_IME>(int32_t ZeroPointA, bool AIsSigned) {
    if (AIsSigned) ZeroPointA = (uint8_t)(ZeroPointA ^ 0x80);
    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE constexpr int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_IME>(int32_t ZeroPointB, bool BIsSigned) {
    if (BIsSigned) ZeroPointB = MLAS_GEMM_QUANT_KERNEL_IME::OffsetBType(ZeroPointB ^ 0x80);
    return ZeroPointB;
}

/* Reuse RVV's CopyPackA (with RVV vectorized row sum) */
template<>
void MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_IME>(
    MLAS_GEMM_QUANT_KERNEL_IME::PackedAType* D, const uint8_t* A,
    size_t lda, size_t CountM, size_t CountK, int32_t* RowSumBuffer, bool AIsSigned)
{
    /* Identical to RVV version — delegate */
    extern void MlasGemmQuantCopyPackA_RVV(uint8_t*, const uint8_t*, size_t, size_t, size_t, int32_t*, bool);
    // For now, inline the same logic
    const size_t AlignedCountK = (CountK + 3) & ~3;
    const uint8_t BitFlipValue = (AIsSigned ? 0x80 : 0);
    while (CountM-- > 0) {
        int32_t RowSum = 0;
        for (size_t k = 0; k < CountK; k++) {
            uint8_t v = A[k] ^ BitFlipValue;
            D[k] = v;
            RowSum += v;
        }
        for (size_t k = CountK; k < AlignedCountK; k++) D[k] = 0;
        *RowSumBuffer++ = RowSum;
        A += lda;
        D += AlignedCountK;
    }
}

/*
 * IME CopyPackB: pack B directly into vmadot tile format
 *
 * vmadot B tile: 4 columns × 8 K = 32 bytes, layout B_tile[n_in*8 + k_in]
 * Tiles: [n/4][k/8] × 32 bytes, total size = ceil(N/4) * ceil(K/8) * 32
 *
 * MLAS calls CopyPackB once per N-stripe, then GemmQuantKernel many times.
 * Pre-packing B here eliminates per-kernel memcpy overhead (the key bottleneck).
 */
template<>
void MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_IME>(
    MLAS_GEMM_QUANT_KERNEL_IME::PackedBType* D, const uint8_t* B,
    size_t ldb, size_t CountN, size_t CountK, int32_t* ColumnSumBuffer, bool BIsSigned)
{
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    const size_t K8 = (CountK + 7) & ~7;
    const size_t K_tiles = K8 / 8;

    /* Zero entire output (handles padding) */
    size_t total = ((CountN + 3) / 4) * K_tiles * 32;
    memset(D, 0, total);

    for (size_t n = 0; n < CountN; n++) {
        const uint8_t* b_col = B + n;
        int32_t ColSum = 0;
        size_t n_tile = n / 4;
        size_t n_in = n % 4;

        for (size_t k = 0; k < CountK; k++) {
            uint8_t v = (*b_col) ^ BitFlipValue;
            ColSum += v;

            size_t k_tile = k / 8;
            size_t k_in = k % 8;
            D[n_tile * K_tiles * 32 + k_tile * 32 + n_in * 8 + k_in] = v;
            b_col += ldb;
        }
        ColumnSumBuffer[n] = ColSum;
    }
}

/* ═══ IME vmadot GEMM kernel (zero-copy B) ═══
 *
 * B is already in vmadot tile format from CopyPackB.
 * A still needs per-tile packing (rows at stride AlignedK → contiguous 32 bytes).
 * A packing is cheap: 4 × memcpy(8 bytes) = 32 bytes per tile.
 */

template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_IME>(
    const MLAS_GEMM_QUANT_KERNEL_IME::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_IME::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    const size_t AlignedK = PackedCountK * MLAS_GEMM_QUANT_KERNEL_IME::PackedK;
    const size_t K8 = (AlignedK + 7) & ~7;
    const size_t K_tiles = K8 / 8;
    size_t RowsHandled = (CountM >= 4) ? 4 : 1;

    size_t n = 0;

    /* ═══ 16-column fast path: 4 × vmadot per K tile, share A load ═══
     *
     * Uses v28, v26, v24, v22 as accumulators (all even, required by vmadot).
     * A is loaded once into v0, B columns loaded into v1, v2, v3, v4.
     * 4 vmadot instructions share the same A data → 75% reduction in A packing.
     *
     * Register budget: 4×m2 accumulators = 8 v-regs, 1 A + 4 B = 5 v-regs = 13 total < 32.
     */
    for (; n + 16 <= CountN && RowsHandled == 4; n += 16) {
        int32_t acc0[16], acc1[16], acc2[16], acc3[16];
        for (int m = 0; m < 4; m++)
            for (int j = 0; j < 4; j++) {
                acc0[m*4+j] = RowSumBuffer[m] * (ZeroPointB ? ZeroPointB[n+j] : 0) + ColumnSumBuffer[n+j];
                acc1[m*4+j] = RowSumBuffer[m] * (ZeroPointB ? ZeroPointB[n+4+j] : 0) + ColumnSumBuffer[n+4+j];
                acc2[m*4+j] = RowSumBuffer[m] * (ZeroPointB ? ZeroPointB[n+8+j] : 0) + ColumnSumBuffer[n+8+j];
                acc3[m*4+j] = RowSumBuffer[m] * (ZeroPointB ? ZeroPointB[n+12+j] : 0) + ColumnSumBuffer[n+12+j];
            }

        int32_t vacc0[16] __attribute__((aligned(64)));
        int32_t vacc1[16] __attribute__((aligned(64)));
        int32_t vacc2[16] __attribute__((aligned(64)));
        int32_t vacc3[16] __attribute__((aligned(64)));
        memset(vacc0, 0, 64); memset(vacc1, 0, 64);
        memset(vacc2, 0, 64); memset(vacc3, 0, 64);

        size_t bo0 = (n/4) * K_tiles * 32;
        size_t bo1 = ((n+4)/4) * K_tiles * 32;
        size_t bo2 = ((n+8)/4) * K_tiles * 32;
        size_t bo3 = ((n+12)/4) * K_tiles * 32;

        const uint8_t* a0 = A;
        const uint8_t* a1 = A + AlignedK;
        const uint8_t* a2 = A + 2*AlignedK;
        const uint8_t* a3 = A + 3*AlignedK;

        for (size_t kt = 0; kt < K_tiles; kt++) {
            size_t ks = kt * 8;
            uint8_t a_buf[32] __attribute__((aligned(32)));
            *(uint64_t*)(a_buf)      = *(const uint64_t*)(a0 + ks);
            *(uint64_t*)(a_buf + 8)  = *(const uint64_t*)(a1 + ks);
            *(uint64_t*)(a_buf + 16) = *(const uint64_t*)(a2 + ks);
            *(uint64_t*)(a_buf + 24) = *(const uint64_t*)(a3 + ks);

            const uint8_t* b0p = B + bo0 + kt*32;
            const uint8_t* b1p = B + bo1 + kt*32;
            const uint8_t* b2p = B + bo2 + kt*32;
            const uint8_t* b3p = B + bo3 + kt*32;

            __asm__ volatile(
                /* Load accumulators */
                "vsetvli t0, zero, e32, m2\n\t"
                "vle32.v v28, (%[C0])\n\t"
                "vle32.v v26, (%[C1])\n\t"
                "vle32.v v24, (%[C2])\n\t"
                "vle32.v v22, (%[C3])\n\t"
                /* Load A (shared) and 4 B tiles */
                "vsetvli t0, zero, e8, m1\n\t"
                "vle8.v v0, (%[A])\n\t"
                "vle8.v v1, (%[B0])\n\t"
                "vle8.v v2, (%[B1])\n\t"
                "vle8.v v3, (%[B2])\n\t"
                "vle8.v v4, (%[B3])\n\t"
                /* 4 vmadot: same A, different B columns */
                "vmadot v28, v0, v1\n\t"
                "vmadot v26, v0, v2\n\t"
                "vmadot v24, v0, v3\n\t"
                "vmadot v22, v0, v4\n\t"
                /* Store accumulators */
                "vsetvli t0, zero, e32, m2\n\t"
                "vse32.v v28, (%[C0])\n\t"
                "vse32.v v26, (%[C1])\n\t"
                "vse32.v v24, (%[C2])\n\t"
                "vse32.v v22, (%[C3])\n\t"
                : : [A] "r"(a_buf),
                    [B0] "r"(b0p), [B1] "r"(b1p), [B2] "r"(b2p), [B3] "r"(b3p),
                    [C0] "r"(vacc0), [C1] "r"(vacc1), [C2] "r"(vacc2), [C3] "r"(vacc3)
                : "t0", "memory"
            );
        }

        for (int m = 0; m < 4; m++)
            for (int j = 0; j < 4; j++) {
                int32_t v0 = acc0[m*4+j] + vacc0[m*4+j];
                int32_t v1 = acc1[m*4+j] + vacc1[m*4+j];
                int32_t v2 = acc2[m*4+j] + vacc2[m*4+j];
                int32_t v3 = acc3[m*4+j] + vacc3[m*4+j];
                if (!ZeroMode) {
                    v0 += C[m*ldc+n+j]; v1 += C[m*ldc+n+4+j];
                    v2 += C[m*ldc+n+8+j]; v3 += C[m*ldc+n+12+j];
                }
                C[m*ldc+n+j] = v0; C[m*ldc+n+4+j] = v1;
                C[m*ldc+n+8+j] = v2; C[m*ldc+n+12+j] = v3;
            }
    }

    /* ═══ 8-column path: 2 × vmadot per K tile ═══ */
    for (; n + 8 <= CountN && RowsHandled == 4; n += 8) {
        /* Zero-point + column sum correction for 8 columns */
        int32_t acc0[16], acc1[16];
        for (int m = 0; m < 4; m++) {
            for (int j = 0; j < 4; j++) {
                int32_t zpb0 = ZeroPointB ? ZeroPointB[n+j] : 0;
                int32_t zpb1 = ZeroPointB ? ZeroPointB[n+4+j] : 0;
                acc0[m*4+j] = RowSumBuffer[m] * zpb0 + ColumnSumBuffer[n+j];
                acc1[m*4+j] = RowSumBuffer[m] * zpb1 + ColumnSumBuffer[n+4+j];
            }
        }

        int32_t vacc0[16] __attribute__((aligned(64)));
        int32_t vacc1[16] __attribute__((aligned(64)));
        memset(vacc0, 0, 64);
        memset(vacc1, 0, 64);

        size_t b_off0 = (n / 4) * K_tiles * 32;
        size_t b_off1 = ((n+4) / 4) * K_tiles * 32;

        const uint8_t* a0 = A;
        const uint8_t* a1 = A + AlignedK;
        const uint8_t* a2 = A + 2*AlignedK;
        const uint8_t* a3 = A + 3*AlignedK;

        for (size_t kt = 0; kt < K_tiles; kt++) {
            size_t k_start = kt * 8;

            /* Pack A once, use for both column groups */
            uint8_t a_buf[32] __attribute__((aligned(32)));
            *(uint64_t*)(a_buf)      = *(const uint64_t*)(a0 + k_start);
            *(uint64_t*)(a_buf + 8)  = *(const uint64_t*)(a1 + k_start);
            *(uint64_t*)(a_buf + 16) = *(const uint64_t*)(a2 + k_start);
            *(uint64_t*)(a_buf + 24) = *(const uint64_t*)(a3 + k_start);

            const uint8_t* b0 = B + b_off0 + kt * 32;
            const uint8_t* b1 = B + b_off1 + kt * 32;

            /* 2 vmadot: same A, different B columns */
            /* Use v28 for group 0, v26 for group 1 (both even registers) */
            __asm__ volatile(
                "vsetvli t0, zero, e32, m2\n\t"
                "vle32.v v28, (%[C0])\n\t"
                "vle32.v v26, (%[C1])\n\t"
                "vsetvli t0, zero, e8, m1\n\t"
                "vle8.v v0, (%[A])\n\t"
                "vle8.v v1, (%[B0])\n\t"
                "vle8.v v2, (%[B1])\n\t"
                "vmadot v28, v0, v1\n\t"
                "vmadot v26, v0, v2\n\t"
                "vsetvli t0, zero, e32, m2\n\t"
                "vse32.v v28, (%[C0])\n\t"
                "vse32.v v26, (%[C1])\n\t"
                : : [A] "r"(a_buf), [B0] "r"(b0), [B1] "r"(b1),
                    [C0] "r"(vacc0), [C1] "r"(vacc1)
                : "t0", "memory"
            );
        }

        /* Write results for both groups */
        for (int m = 0; m < 4; m++)
            for (int j = 0; j < 4; j++) {
                int32_t v0 = acc0[m*4+j] + vacc0[m*4+j];
                int32_t v1 = acc1[m*4+j] + vacc1[m*4+j];
                if (!ZeroMode) { v0 += C[m*ldc+n+j]; v1 += C[m*ldc+n+4+j]; }
                C[m*ldc+n+j] = v0;
                C[m*ldc+n+4+j] = v1;
            }
    }

    /* ═══ 4-column path for remaining columns ═══ */
    for (; n + 4 <= CountN; n += 4) {
        int32_t zpb[4];
        for (int j = 0; j < 4; j++)
            zpb[j] = ZeroPointB ? ZeroPointB[n+j] : 0;

        if (RowsHandled == 4) {
            int32_t acc[16];
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < 4; j++)
                    acc[m*4+j] = RowSumBuffer[m] * zpb[j] + ColumnSumBuffer[n+j];

            int32_t vacc[16] __attribute__((aligned(64)));
            memset(vacc, 0, 64);
            size_t b_n_offset = (n / 4) * K_tiles * 32;

            const uint8_t* a0 = A;
            const uint8_t* a1 = A + AlignedK;
            const uint8_t* a2 = A + 2*AlignedK;
            const uint8_t* a3 = A + 3*AlignedK;

            for (size_t kt = 0; kt < K_tiles; kt++) {
                size_t k_start = kt * 8;
                uint8_t a_buf[32] __attribute__((aligned(32)));
                *(uint64_t*)(a_buf)      = *(const uint64_t*)(a0 + k_start);
                *(uint64_t*)(a_buf + 8)  = *(const uint64_t*)(a1 + k_start);
                *(uint64_t*)(a_buf + 16) = *(const uint64_t*)(a2 + k_start);
                *(uint64_t*)(a_buf + 24) = *(const uint64_t*)(a3 + k_start);

                const uint8_t* b_tile = B + b_n_offset + kt * 32;
                __asm__ volatile(
                    "vsetvli t0, zero, e32, m2\n\t"
                    "vle32.v v28, (%[C])\n\t"
                    "vsetvli t0, zero, e8, m1\n\t"
                    "vle8.v v0, (%[A])\n\t"
                    "vle8.v v1, (%[B])\n\t"
                    "vmadot v28, v0, v1\n\t"
                    "vsetvli t0, zero, e32, m2\n\t"
                    "vse32.v v28, (%[C])\n\t"
                    : : [A] "r"(a_buf), [B] "r"(b_tile), [C] "r"(vacc)
                    : "t0", "memory"
                );
            }

            for (int m = 0; m < 4; m++)
                for (int j = 0; j < 4; j++) {
                    int32_t val = acc[m*4+j] + vacc[m*4+j];
                    if (!ZeroMode) val += C[m*ldc + n+j];
                    C[m*ldc + n+j] = val;
                }
        } else {
            /* 1-row fallback: use B in vmadot tile format, extract elements */
            size_t b_n_offset = (n / 4) * K_tiles * 32;
            for (int j = 0; j < 4; j++) {
                int32_t Acc = RowSumBuffer[0] * zpb[j] + ColumnSumBuffer[n+j];
                for (size_t k = 0; k < AlignedK; k++) {
                    size_t kt = k / 8, ki = k % 8;
                    Acc += (int32_t)A[k] * (int32_t)B[b_n_offset + kt*32 + j*8 + ki];
                }
                if (!ZeroMode) Acc += C[n+j];
                C[n+j] = Acc;
            }
        }
    }

    /* Remaining columns (< 4): scalar fallback */
    for (; n < CountN; n++) {
        int32_t zpb = ZeroPointB ? ZeroPointB[n] : 0;
        for (size_t m = 0; m < RowsHandled; m++) {
            int32_t Acc = RowSumBuffer[m] * zpb + ColumnSumBuffer[n];
            const uint8_t* a = A + m*AlignedK;
            const uint8_t* b = B + n*AlignedK;
            for (size_t k = 0; k < AlignedK; k++)
                Acc += (int32_t)a[k] * (int32_t)b[k];
            if (!ZeroMode) Acc += C[m*ldc + n];
            C[m*ldc + n] = Acc;
        }
    }

    return RowsHandled;
}

/* Dispatch table */
const MLAS_GEMM_QUANT_DISPATCH MlasGemmU8S8DispatchIme = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_IME>,
    nullptr,
    nullptr,
    MLAS_GEMM_QUANT_KERNEL_IME::PackedK,
    0,
    MLAS_GEMM_QUANT_KERNEL_IME::Strides.M,
};

extern "C" {
    const MLAS_GEMM_QUANT_DISPATCH* MlasGemmU8S8DispatchImePtr = &MlasGemmU8S8DispatchIme;
}
