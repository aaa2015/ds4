#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
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

static void test_rms_norm(const float *x, const float *w, float *out, uint32_t dim, float eps) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = x[i] * scale * (w ? w[i] : 1.0f);
    }
}

static void test_q8_0_gemv(const test_block_q8_0 *w, uint32_t in_dim, uint32_t out_dim, const float *x, float *out) {
    const uint32_t n_blocks = in_dim / 32;
    for (uint32_t r = 0; r < out_dim; r++) {
        const test_block_q8_0 *row = w + (uint64_t)r * n_blocks;
        float sum = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
            float blk = 0.0f;
            for (int j = 0; j < 32; j++) {
                blk += (float)row[b].qs[j] * x[b * 32 + j];
            }
            sum += blk * (float)row[b].d;
        }
        out[r] = sum;
    }
}

int main(void) {
    printf("=================================================================\n");
    printf("     DS4 Apple Neural Engine (ANE) MTP Draft Speculative Test    \n");
    printf("=================================================================\n");

    if (!ds4_ane_init()) {
        fprintf(stderr, "[-] ANE initialization failed.\n");
        return 1;
    }

    if (!ds4_ane_mtp_is_available()) {
        fprintf(stderr, "[-] ANE MTP is not available.\n");
        return 1;
    }

    printf("[+] ANE Engine Initialized Successfully.\n");
    printf("[+] Device: %s\n", ds4_ane_get_device_info());
    printf("[+] ANE MTP Draft Offload: ENABLED\n");

    const uint32_t model_dim = 2048;
    const uint32_t shared_dim = 1536;
    const uint32_t n_hc = 4;
    const uint32_t n_vocab = 8192;
    const int test_token = 42;

    const uint32_t in_blocks = model_dim / 32;
    const uint32_t down_blocks = shared_dim / 32;

    /* Base model buffers: token_embd (FP16) + head (Q8_0) */
    const uint64_t embd_bytes = (uint64_t)n_vocab * model_dim * sizeof(uint16_t);
    const uint64_t head_row_bytes = in_blocks * sizeof(test_block_q8_0);
    const uint64_t head_bytes = (uint64_t)n_vocab * head_row_bytes;
    const uint64_t base_model_size = embd_bytes + head_bytes;
    uint8_t *base_model_map = (uint8_t *)calloc(base_model_size, 1);

    const uint64_t token_embd_offset = 0;
    const uint64_t head_offset = embd_bytes;

    /* MTP model buffers */
    const uint64_t enorm_bytes = (uint64_t)model_dim * sizeof(float);
    const uint64_t e_proj_bytes = (uint64_t)model_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t hnorm_bytes = (uint64_t)model_dim * sizeof(float);
    const uint64_t h_proj_bytes = (uint64_t)model_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t gate_bytes = (uint64_t)shared_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t up_bytes   = (uint64_t)shared_dim * in_blocks * sizeof(test_block_q8_0);
    const uint64_t down_bytes = (uint64_t)model_dim * down_blocks * sizeof(test_block_q8_0);
    const uint64_t norm_bytes = (uint64_t)model_dim * sizeof(float);

    const uint64_t enorm_offset = 0;
    const uint64_t e_proj_offset = enorm_offset + enorm_bytes;
    const uint64_t hnorm_offset = e_proj_offset + e_proj_bytes;
    const uint64_t h_proj_offset = hnorm_offset + hnorm_bytes;
    const uint64_t ffn_gate_offset = h_proj_offset + h_proj_bytes;
    const uint64_t ffn_up_offset = ffn_gate_offset + gate_bytes;
    const uint64_t ffn_down_offset = ffn_up_offset + up_bytes;
    const uint64_t norm_offset = ffn_down_offset + down_bytes;
    const uint64_t mtp_model_size = norm_offset + norm_bytes;
    uint8_t *mtp_model_map = (uint8_t *)calloc(mtp_model_size, 1);

    /* Initialize pseudo-random weights */
    srand(1234);

    /* Embedding table (FP16) */
    __fp16 *emb_table = (__fp16 *)(base_model_map + token_embd_offset);
    for (uint64_t i = 0; i < (uint64_t)n_vocab * model_dim; i++) {
        emb_table[i] = (__fp16)(((float)(rand() % 100) / 100.0f) - 0.5f);
    }

    /* Head weights (Q8_0) */
    test_block_q8_0 *head_w = (test_block_q8_0 *)(base_model_map + head_offset);
    for (uint64_t i = 0; i < (uint64_t)n_vocab * in_blocks; i++) {
        head_w[i].d = (__fp16)0.05f;
        for (int b = 0; b < 32; b++) {
            head_w[i].qs[b] = (int8_t)((rand() % 50) - 25);
        }
    }

    /* MTP norms (F32) */
    float *enorm_w = (float *)(mtp_model_map + enorm_offset);
    float *hnorm_w = (float *)(mtp_model_map + hnorm_offset);
    float *norm_w  = (float *)(mtp_model_map + norm_offset);
    for (uint32_t i = 0; i < model_dim; i++) {
        enorm_w[i] = 1.0f + 0.1f * ((float)(rand() % 20) / 20.0f);
        hnorm_w[i] = 1.0f + 0.1f * ((float)(rand() % 20) / 20.0f);
        norm_w[i]  = 1.0f + 0.1f * ((float)(rand() % 20) / 20.0f);
    }

    /* MTP projections (Q8_0) */
    test_block_q8_0 *e_proj_w = (test_block_q8_0 *)(mtp_model_map + e_proj_offset);
    test_block_q8_0 *h_proj_w = (test_block_q8_0 *)(mtp_model_map + h_proj_offset);
    for (uint64_t i = 0; i < (uint64_t)model_dim * in_blocks; i++) {
        e_proj_w[i].d = (__fp16)0.02f;
        h_proj_w[i].d = (__fp16)0.02f;
        for (int b = 0; b < 32; b++) {
            e_proj_w[i].qs[b] = (int8_t)((rand() % 40) - 20);
            h_proj_w[i].qs[b] = (int8_t)((rand() % 40) - 20);
        }
    }

    /* MTP FFN weights (Q8_0) */
    test_block_q8_0 *gate_w = (test_block_q8_0 *)(mtp_model_map + ffn_gate_offset);
    test_block_q8_0 *up_w   = (test_block_q8_0 *)(mtp_model_map + ffn_up_offset);
    test_block_q8_0 *down_w = (test_block_q8_0 *)(mtp_model_map + ffn_down_offset);

    for (uint64_t i = 0; i < (uint64_t)shared_dim * in_blocks; i++) {
        gate_w[i].d = (__fp16)0.02f;
        up_w[i].d   = (__fp16)0.02f;
        for (int b = 0; b < 32; b++) {
            gate_w[i].qs[b] = (int8_t)((rand() % 40) - 20);
            up_w[i].qs[b]   = (int8_t)((rand() % 40) - 20);
        }
    }
    for (uint64_t i = 0; i < (uint64_t)model_dim * down_blocks; i++) {
        down_w[i].d = (__fp16)0.02f;
        for (int b = 0; b < 32; b++) {
            down_w[i].qs[b] = (int8_t)((rand() % 40) - 20);
        }
    }

    /* Input prev_hc */
    float *prev_hc = (float *)malloc(n_hc * model_dim * sizeof(float));
    for (uint32_t i = 0; i < n_hc * model_dim; i++) {
        prev_hc[i] = ((float)(rand() % 100) / 100.0f) - 0.5f;
    }

    /* Output buffers */
    float *out_hc_ane = (float *)malloc(n_hc * model_dim * sizeof(float));
    float *logits_ane = (float *)malloc(n_vocab * sizeof(float));
    int top_id_ane = -1;

    /* Compute CPU reference */
    printf("[*] Computing CPU Reference MTP Speculative Draft...\n");
    float *ref_emb = (float *)malloc(model_dim * sizeof(float));
    float *ref_enorm = (float *)malloc(model_dim * sizeof(float));
    float *ref_eproj = (float *)malloc(model_dim * sizeof(float));
    float *ref_hnorm = (float *)malloc(model_dim * sizeof(float));
    float *ref_hproj = (float *)malloc(model_dim * sizeof(float));
    float *ref_out_hc = (float *)malloc(n_hc * model_dim * sizeof(float));
    float *ref_mid = (float *)malloc(shared_dim * sizeof(float));
    float *ref_down = (float *)malloc(model_dim * sizeof(float));
    float *ref_collapsed = (float *)malloc(model_dim * sizeof(float));
    float *ref_final_emb = (float *)malloc(model_dim * sizeof(float));
    float *ref_logits = (float *)malloc(n_vocab * sizeof(float));

    /* 1. Embedding lookup */
    const __fp16 *tok_emb_ptr = emb_table + (uint64_t)test_token * model_dim;
    for (uint32_t i = 0; i < model_dim; i++) {
        ref_emb[i] = (float)tok_emb_ptr[i];
    }

    /* 2. Enorm */
    test_rms_norm(ref_emb, enorm_w, ref_enorm, model_dim, 1e-6f);

    /* 3. Eproj */
    test_q8_0_gemv(e_proj_w, model_dim, model_dim, ref_enorm, ref_eproj);

    /* 4. Hnorm & Hproj & residual */
    for (uint32_t h = 0; h < n_hc; h++) {
        const float *prev_row = prev_hc + (uint64_t)h * model_dim;
        float *out_row = ref_out_hc + (uint64_t)h * model_dim;

        test_rms_norm(prev_row, hnorm_w, ref_hnorm, model_dim, 1e-6f);
        test_q8_0_gemv(h_proj_w, model_dim, model_dim, ref_hnorm, ref_hproj);

        for (uint32_t i = 0; i < model_dim; i++) {
            out_row[i] = ref_eproj[i] + ref_hproj[i];
        }

        /* 5. FFN */
        for (uint32_t r = 0; r < shared_dim; r++) {
            const test_block_q8_0 *grow = gate_w + (uint64_t)r * in_blocks;
            const test_block_q8_0 *urow = up_w   + (uint64_t)r * in_blocks;
            float g = 0.0f, u = 0.0f;
            for (uint32_t b = 0; b < in_blocks; b++) {
                float gb = 0.0f, ub = 0.0f;
                for (int j = 0; j < 32; j++) {
                    gb += (float)grow[b].qs[j] * out_row[b * 32 + j];
                    ub += (float)urow[b].qs[j] * out_row[b * 32 + j];
                }
                g += gb * (float)grow[b].d;
                u += ub * (float)urow[b].d;
            }
            ref_mid[r] = test_silu(g) * u;
        }

        test_q8_0_gemv(down_w, shared_dim, model_dim, ref_mid, ref_down);
        for (uint32_t i = 0; i < model_dim; i++) {
            out_row[i] += ref_down[i];
        }
    }

    /* 6. Collapse HC */
    for (uint32_t i = 0; i < model_dim; i++) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            sum += ref_out_hc[(uint64_t)h * model_dim + i];
        }
        ref_collapsed[i] = sum / (float)n_hc;
    }
    test_rms_norm(ref_collapsed, norm_w, ref_final_emb, model_dim, 1e-6f);

    /* 7. Output logits & Argmax */
    test_q8_0_gemv(head_w, model_dim, n_vocab, ref_final_emb, ref_logits);
    int ref_top_id = 0;
    float ref_max_logit = ref_logits[0];
    for (uint32_t v = 1; v < n_vocab; v++) {
        if (ref_logits[v] > ref_max_logit) {
            ref_max_logit = ref_logits[v];
            ref_top_id = (int)v;
        }
    }

    printf("[+] Reference computation complete: Top Predicted Token ID = %d (logit = %.4f)\n",
           ref_top_id, ref_max_logit);

    /* Run ANE MTP Draft */
    printf("[*] Executing ds4_ane_eval_mtp_draft on ANE...\n");
    int ok = ds4_ane_eval_mtp_draft(
        test_token,
        1,
        base_model_map,
        base_model_size,
        token_embd_offset,
        n_vocab,
        mtp_model_map,
        mtp_model_size,
        enorm_offset,
        e_proj_offset,
        hnorm_offset,
        h_proj_offset,
        ffn_gate_offset,
        ffn_up_offset,
        ffn_down_offset,
        norm_offset,
        head_offset,
        prev_hc,
        out_hc_ane,
        model_dim,
        shared_dim,
        n_hc,
        logits_ane,
        &top_id_ane
    );

    if (!ok) {
        fprintf(stderr, "[-] ds4_ane_eval_mtp_draft failed!\n");
        return 1;
    }

    printf("[+] ANE MTP Draft returned: Top Predicted Token ID = %d (logit = %.4f)\n",
           top_id_ane, logits_ane[top_id_ane]);

    /* Verify correctness */
    if (top_id_ane == ref_top_id) {
        printf("[+] EXACT MATCH: ANE top predicted token ID (%d) matches reference (%d)!\n",
               top_id_ane, ref_top_id);
    } else {
        fprintf(stderr, "[-] MISMATCH: ANE top token %d != reference %d\n", top_id_ane, ref_top_id);
        return 1;
    }

    /* Compute Cosine Similarity of out_hc */
    double dot = 0.0, norm_ane = 0.0, norm_ref = 0.0;
    for (uint32_t i = 0; i < n_hc * model_dim; i++) {
        dot += (double)out_hc_ane[i] * (double)ref_out_hc[i];
        norm_ane += (double)out_hc_ane[i] * (double)out_hc_ane[i];
        norm_ref += (double)ref_out_hc[i] * (double)ref_out_hc[i];
    }
    double sim = dot / (sqrt(norm_ane) * sqrt(norm_ref));
    printf("[+] HC State Cosine Similarity: %.8f\n", sim);
    if (sim < 0.9999) {
        fprintf(stderr, "[-] Cosine similarity below threshold!\n");
        return 1;
    }

    /* Benchmark Latency */
    const int benchmark_iters = 50;
    printf("[*] Benchmarking ANE MTP Draft over %d iterations...\n", benchmark_iters);
    double t0 = get_time_us();
    for (int it = 0; it < benchmark_iters; it++) {
        ds4_ane_eval_mtp_draft(
            test_token,
            (uint32_t)(it + 1),
            base_model_map,
            base_model_size,
            token_embd_offset,
            n_vocab,
            mtp_model_map,
            mtp_model_size,
            enorm_offset,
            e_proj_offset,
            hnorm_offset,
            h_proj_offset,
            ffn_gate_offset,
            ffn_up_offset,
            ffn_down_offset,
            norm_offset,
            head_offset,
            prev_hc,
            out_hc_ane,
            model_dim,
            shared_dim,
            n_hc,
            logits_ane,
            &top_id_ane
        );
    }
    double t1 = get_time_us();
    double avg_ms = (t1 - t0) / (double)benchmark_iters / 1000.0;
    printf("[+] ANE MTP Draft Average Latency: %.3f ms (Drafting TPS potential: %.1f tokens/sec)\n",
           avg_ms, 1000.0 / avg_ms);

    /* Test ds4_ane_rms_norm and ds4_ane_vocab_project_argmax (GLM-5.3-Flash Head offload) */
    printf("\n[*] Testing ds4_ane_rms_norm & ds4_ane_vocab_project_argmax...\n");
    float *ane_normed = (float *)malloc(model_dim * sizeof(float));
    ds4_ane_rms_norm(ref_collapsed, norm_w, ane_normed, model_dim, 1e-6f);
    float max_norm_diff = 0.0f;
    for (uint32_t i = 0; i < model_dim; i++) {
        float diff = fabsf(ane_normed[i] - ref_final_emb[i]);
        if (diff > max_norm_diff) max_norm_diff = diff;
    }
    printf("[+] ds4_ane_rms_norm Max Abs Diff: %.6f\n", max_norm_diff);
    if (max_norm_diff > 1e-4f) {
        fprintf(stderr, "[-] ds4_ane_rms_norm precision error too high!\n");
        free(ane_normed);
        return 1;
    }

    int proj_top_id = -1;
    int proj_ok = ds4_ane_vocab_project_argmax(
        base_model_map,
        base_model_size,
        head_offset,
        model_dim,
        n_vocab,
        ane_normed,
        &proj_top_id
    );
    if (!proj_ok) {
        fprintf(stderr, "[-] ds4_ane_vocab_project_argmax returned failure!\n");
        free(ane_normed);
        return 1;
    }
    printf("[+] ds4_ane_vocab_project_argmax Top Token ID: %d (expected %d)\n",
           proj_top_id, ref_top_id);
    if (proj_top_id != ref_top_id) {
        fprintf(stderr, "[-] Vocab projection argmax mismatch: got %d, expected %d!\n",
                proj_top_id, ref_top_id);
        free(ane_normed);
        return 1;
    }
    printf("[+] EXACT MATCH: ds4_ane_vocab_project_argmax matches reference argmax!\n");

    /* Benchmark Vocab Project Latency */
    const int proj_iters = 100;
    double pt0 = get_time_us();
    for (int it = 0; it < proj_iters; it++) {
        ds4_ane_vocab_project_argmax(
            base_model_map,
            base_model_size,
            head_offset,
            model_dim,
            n_vocab,
            ane_normed,
            &proj_top_id
        );
    }
    double pt1 = get_time_us();
    double proj_avg_us = (pt1 - pt0) / (double)proj_iters;
    printf("[+] ds4_ane_vocab_project_argmax Average Latency: %.2f us (%.3f ms)\n",
           proj_avg_us, proj_avg_us / 1000.0);
    free(ane_normed);

    printf("=================================================================\n");
    printf("    [PASS] ANE MTP Speculative Draft Subsystem Fully Operational!\n");
    printf("=================================================================\n");

    /* Cleanup */
    free(base_model_map);
    free(mtp_model_map);
    free(prev_hc);
    free(out_hc_ane);
    free(logits_ane);
    free(ref_emb);
    free(ref_enorm);
    free(ref_eproj);
    free(ref_hnorm);
    free(ref_hproj);
    free(ref_out_hc);
    free(ref_mid);
    free(ref_down);
    free(ref_collapsed);
    free(ref_final_emb);
    free(ref_logits);

    return 0;
}
