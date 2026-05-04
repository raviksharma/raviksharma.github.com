#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>
#include <stdint.h>
#include <time.h>

static inline uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <seed>\n", argv[0]);
        return 1;
    }

    int seed = atoi(argv[1]);
    int iterations = 100000000;

    size_t vl = __riscv_vsetvl_e32m1(8);

    // Both will do (iterations * 8 * vl) total float additions

    // --- Scalar: 8*vl dependent adds per iteration ---
    float sa = (float)seed;
    uint64_t t0 = now_ns();
    for (int i = 0; i < iterations; i++) {
        for (size_t j = 0; j < 8 * vl; j++) {
            sa = sa + 1.0f;
        }
    }
    uint64_t scalar_ns = now_ns() - t0;

    // --- Vector: 8 dependent vector adds per iteration (each does vl adds) ---
    float ones[8] = {1,1,1,1,1,1,1,1};
    float seeds[8] = {seed,seed,seed,seed,seed,seed,seed,seed};

    vfloat32m1_t va = __riscv_vle32_v_f32m1(seeds, vl);
    vfloat32m1_t vone = __riscv_vle32_v_f32m1(ones, vl);

    t0 = now_ns();
    for (int i = 0; i < iterations; i++) {
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
        va = __riscv_vfadd_vv_f32m1(va, vone, vl);
    }
    uint64_t vector_ns = now_ns() - t0;

    float r[8];
    __riscv_vse32_v_f32m1(r, va, vl);

    uint64_t total_adds = (uint64_t)iterations * 8 * vl;
    printf("VLEN: %zu elements (e32m1)\n", vl);
    printf("Total float additions each: %lu\n", total_adds);
    printf("Scalar: %.3f s\n", scalar_ns / 1e9);
    printf("Vector: %.3f s\n", vector_ns / 1e9);
    printf("Speedup: %.1fx\n", (double)scalar_ns / vector_ns);
    printf("(results: scalar=%.1f, vector=%.1f)\n", sa, r[0]);

    return 0;
}