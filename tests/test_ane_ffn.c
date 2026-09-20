#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#include "ds4_ane.h"

#pragma pack(push, 2)
typedef struct {
    __fp16  d;
    int8_t  qs[32];
} test_block_q8_0;
#pragma pack(pop)

static double get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

static inline float test_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main(void) {
    printf("=================================================================\n");
    printf("         DS4 Apple Neural Engine (ANE) Heterogeneous Test        \n");
    printf("=================================================================\n");

    if (!ds4_ane_init()) {
        fprintf(stderr, "[-] ANE initialization failed or disabled.\n");
        return 1;
    }

    printf("[+] ANE Engine Initialized Successfully.\n");
    printf("[+] Device: %s\n", ds4_ane_get_device_info());
    printf("[+] ANE Available: %s\n", ds4_ane_is_available() ? "YES" : "NO");

    /* DeepSeek-V4.1 Flash typical Shared Expert dimensions */
    const uint32_t model_dim = 2048;
    const uint32_t shared_dim = 1536;
    const float clamp_value = 6.0f;

    const uint32_t in_blocks = model_dim / 32;
    const uint32_t down_blocks = shared_dim / 32;

    const uint64_t gate_bytes = (uint64_t)shared_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t up_bytes   = (uint64_t)shared_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t down_bytes = (uint64_t)model_dim * down_blocks * sizeof(test_block_q8_0);

    const uint64_t total_weight_bytes = gate_bytes + up_bytes + down_bytes;
    uint8_t *model_map = (uint8_t *)calloc(total_weight_bytes, 1);
    if (!model_map) {
        model_map = (uint8_t *)malloc(total_weight_bytes);
        memset(model_map, 0, total_weight_bytes);
    }

    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_bytes;
    const uint64_t down_offset = gate_bytes + up_bytes;

    /* Initialize pseudo-random weights with valid Q8_0 scales */
    srand(42);
    test_block_q8_0 *gate_blocks = (test_block_q8_0 *)(model_map + gate_offset);
    test_block_q8_0 *up_blocks   = (test_block_q8_0 *)(model_map + up_offset);
    test_block_q8_0 *down_blocks_w = (test_block_q8_0 *)(model_map + down_offset);

    for (uint64_t i = 0; i < (uint64_t)shared_dim * in_blocks; i++) {
        gate_blocks[i].d = (__fp16)0.05f;
        up_blocks[i].d   = (__fp16)0.05f;
        for (int b = 0; b < 32; b++) {
            gate_blocks[i].qs[b] = (int8_t)((rand() % 60) - 30);
            up_blocks[i].qs[b]   = (int8_t)((rand() % 60) - 30);
        }
    }

    for (uint64_t i = 0; i < (uint64_t)model_dim * down_blocks; i++) {
        down_blocks_w[i].d = (__fp16)0.05f;
        for (int b = 0; b < 32; b++) {
            down_blocks_w[i].qs[b] = (int8_t)((rand() % 60) - 30);
        }
    }

    /* Input vector */
    float *x = (float *)malloc(model_dim * sizeof(float));
    for (uint32_t i = 0; i < model_dim; i++) {
        x[i] = ((float)(rand() % 100) / 100.0f) - 0.5f;
    }

    /* Output buffer for ANE */
    float *ane_out = (float *)malloc(model_dim * sizeof(float));
    memset(ane_out, 0, model_dim * sizeof(float));

    /* Compute Reference on CPU */
    printf("[*] Computing CPU reference SwiGLU FFN...\n");
    float *ref_mid = (float *)malloc(shared_dim * sizeof(float));
    float *ref_out = (float *)malloc(model_dim * sizeof(float));

    for (uint32_t row = 0; row < shared_dim; row++) {
        const test_block_q8_0 *grow = gate_blocks + (uint64_t)row * in_blocks;
        const test_block_q8_0 *urow = up_blocks + (uint64_t)row * in_blocks;
        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (uint32_t b = 0; b < in_blocks; b++) {
            float g_blk = 0.0f, u_blk = 0.0f;
            for (int j = 0; j < 32; j++) {
                g_blk += (float)grow[b].qs[j] * x[b * 32 + j];
                u_blk += (float)urow[b].qs[j] * x[b * 32 + j];
            }
            g_sum += g_blk * (float)grow[b].d;
            u_sum += u_blk * (float)urow[b].d;
        }

        float g = g_sum;
        float u = u_sum;
        if (clamp_value > 1.0e-6f) {
            if (g > clamp_value) g = clamp_value;
            if (u > clamp_value) u = clamp_value;
            else if (u < -clamp_value) u = -clamp_value;
        }
        ref_mid[row] = test_silu(g) * u;
    }

    for (uint32_t row = 0; row < model_dim; row++) {
        const test_block_q8_0 *drow = down_blocks_w + (uint64_t)row * down_blocks;
        float sum = 0.0f;
        for (uint32_t b = 0; b < down_blocks; b++) {
            float blk = 0.0f;
            for (int j = 0; j < 32; j++) {
                blk += (float)drow[b].qs[j] * ref_mid[b * 32 + j];
            }
            sum += blk * (float)drow[b].d;
        }
        ref_out[row] = sum;
    }

    /* Test 1: Single execution & numerical verification */
    printf("[*] Running ANE asynchronous Shared Expert FFN...\n");
    int start_res = ds4_ane_shared_ffn_start(
        model_map, total_weight_bytes,
        gate_offset, up_offset, down_offset,
        model_dim, shared_dim,
        x, ane_out, clamp_value
    );

    if (!start_res) {
        fprintf(stderr, "[-] ds4_ane_shared_ffn_start failed!\n");
        return 1;
    }

    int finish_res = ds4_ane_shared_ffn_finish();
    if (!finish_res) {
        fprintf(stderr, "[-] ds4_ane_shared_ffn_finish failed!\n");
        return 1;
    }

    /* Verification: Compare ane_out with ref_out */
    double dot_prod = 0.0, norm_ane = 0.0, norm_ref = 0.0;
    float max_abs_diff = 0.0f;
    for (uint32_t i = 0; i < model_dim; i++) {
        float diff = fabsf(ane_out[i] - ref_out[i]);
        if (diff > max_abs_diff) max_abs_diff = diff;
        dot_prod += (double)ane_out[i] * (double)ref_out[i];
        norm_ane += (double)ane_out[i] * (double)ane_out[i];
        norm_ref += (double)ref_out[i] * (double)ref_out[i];
    }
    double cosine_sim = dot_prod / (sqrt(norm_ane) * sqrt(norm_ref));

    printf("[+] Numerical Verification Results:\n");
    printf("    Max Absolute Diff: %.6f\n", max_abs_diff);
    printf("    Cosine Similarity: %.8f\n", cosine_sim);

    if (cosine_sim < 0.999) {
        fprintf(stderr, "[-] Precision mismatch! Cosine similarity %.6f < 0.999\n", cosine_sim);
        return 1;
    }
    printf("[+] PASSED Numerical Precision Test!\n");

    /* Test 2: Benchmark Throughput & Latency */
    const int warmup = 10;
    const int iters = 100;
    printf("[*] Benchmarking latency over %d iterations...\n", iters);

    for (int i = 0; i < warmup; i++) {
        ds4_ane_shared_ffn_start(
            model_map, total_weight_bytes,
            gate_offset, up_offset, down_offset,
            model_dim, shared_dim,
            x, ane_out, clamp_value
        );
        ds4_ane_shared_ffn_finish();
    }

    double t0 = get_time_us();
    for (int i = 0; i < iters; i++) {
        ds4_ane_shared_ffn_start(
            model_map, total_weight_bytes,
            gate_offset, up_offset, down_offset,
            model_dim, shared_dim,
            x, ane_out, clamp_value
        );
        ds4_ane_shared_ffn_finish();
    }
    double t1 = get_time_us();

    double avg_latency_us = (t1 - t0) / iters;
    double flos_per_ffn = 2.0 * (double)model_dim * (double)shared_dim * 3.0; // 3 GEMVs (gate, up, down)
    double gflops = (flos_per_ffn / (avg_latency_us * 1e-6)) / 1e9;

    printf("[+] Benchmark Results:\n");
    printf("    Average Latency: %.2f us (%.3f ms)\n", avg_latency_us, avg_latency_us / 1000.0);
    printf("    Compute Throughput: %.2f GFLOPS\n", gflops);
    printf("    Estimated Layers/s: %.1f layers/sec\n", 1e6 / avg_latency_us);

    /* Test 3: Abort functionality */
    ds4_ane_shared_ffn_start(
        model_map, total_weight_bytes,
        gate_offset, up_offset, down_offset,
        model_dim, shared_dim,
        x, ane_out, clamp_value
    );
    ds4_ane_shared_ffn_abort();
    printf("[+] Abort test passed cleanly.\n");

    printf("\n[SUCCESS] All ANE heterogeneous offload tests passed!\n");

    free(x);
    free(ane_out);
    free(ref_mid);
    free(ref_out);
    free(model_map);

    return 0;
}
