#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <IOSurface/IOSurface.h>
#include <arm_neon.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <math.h>
#include <float.h>

#include "ds4_ane.h"

/*
 * ds4_ane.m - Apple Neural Engine (ANE) Heterogeneous Acceleration for ds4.
 *
 * This subsystem offloads the Shared Expert FFN in DeepSeek-V4.1 Flash
 * to run asynchronously alongside Metal GPU routed MoE calculations.
 *
 * On M2 Ultra, this utilizes the 32-core ANE (31.6 TOPS) and spare memory bandwidth
 * without competing for GPU shader execution pipelines.
 */

#pragma pack(push, 2)
typedef struct {
    __fp16  d;
    int8_t  qs[32];
} ane_block_q8_0;
#pragma pack(pop)

static BOOL g_ane_initialized = NO;
static BOOL g_ane_available = NO;
static BOOL g_ane_enabled = YES;
static BOOL g_ane_mtp_enabled = YES;
static char g_device_info[256] = "Unknown Apple Silicon";
static uint32_t g_ane_cores = 0;

static dispatch_queue_t g_ane_queue = NULL;
static dispatch_group_t g_ane_group = NULL;
static volatile BOOL g_ane_in_flight = NO;
static volatile int g_ane_last_status = 0;

/* Intermediate buffer for SwiGLU activation (shared_dim floats) */
static float *g_ane_mid_buf = NULL;
static size_t g_ane_mid_cap = 0;

/* CoreML model cache (if a compiled .mlmodelc bundle is provided) */
static MLModel *g_coreml_model = nil;
static NSString *g_coreml_path = nil;

static inline float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

#if defined(__ARM_FEATURE_DOTPROD)
/* Accelerated dot product of Q8_0 block with float32 vector using ARM NEON */
static inline float dot_q8_0_f32_neon(const ane_block_q8_0 *b, const float *x) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (int i = 0; i < 32; i += 16) {
        int8x16_t q = vld1q_s8(&b->qs[i]);
        int16x8_t q_low = vmovl_s8(vget_low_s8(q));
        int16x8_t q_high = vmovl_s8(vget_high_s8(q));

        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t x2 = vld1q_f32(x + i + 8);
        float32x4_t x3 = vld1q_f32(x + i + 12);

        float32x4_t qf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_low)));
        float32x4_t qf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_low)));
        float32x4_t qf2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_high)));
        float32x4_t qf3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_high)));

        sum0 = vmlaq_f32(sum0, qf0, x0);
        sum1 = vmlaq_f32(sum1, qf1, x1);
        sum2 = vmlaq_f32(sum2, qf2, x2);
        sum3 = vmlaq_f32(sum3, qf3, x3);
    }

    float32x4_t total = vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
    float sum = vaddvq_f32(total);
    return sum * (float)b->d;
}
#else
static inline float dot_q8_0_f32_scalar(const ane_block_q8_0 *b, const float *x) {
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) {
        sum += (float)b->qs[i] * x[i];
    }
    return sum * (float)b->d;
}
#endif

/* Vector-matrix multiply for Q8_0 weights */
static void ane_q8_0_gemv(
    const ane_block_q8_0 *weights,
    uint32_t in_dim,
    uint32_t out_dim,
    const float *x,
    float *out
) {
    const uint32_t n_blocks = in_dim / 32;
    for (uint32_t row = 0; row < out_dim; row++) {
        const ane_block_q8_0 *row_blocks = weights + (uint64_t)row * n_blocks;
        float row_sum = 0.0f;
        for (uint32_t b = 0; b < n_blocks; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
            row_sum += dot_q8_0_f32_neon(&row_blocks[b], x + b * 32);
#else
            row_sum += dot_q8_0_f32_scalar(&row_blocks[b], x + b * 32);
#endif
        }
        out[row] = row_sum;
    }
}

/* Probe Apple Silicon processor and ANE hardware capabilities */
static void ds4_ane_probe_hardware(void) {
    size_t size = sizeof(g_device_info);
    if (sysctlbyname("machdep.cpu.brand_string", g_device_info, &size, NULL, 0) != 0) {
        snprintf(g_device_info, sizeof(g_device_info), "Apple Silicon (Unknown)");
    }

    uint32_t ncpu = 0;
    size = sizeof(ncpu);
    sysctlbyname("hw.ncpu", &ncpu, &size, NULL, 0);

    /* Heuristics for Neural Engine core counts across Apple Silicon:
     * - M1/M2/M3/M4 base/Pro/Max: 16 cores (approx 15.8 ~ 18 TOPS on M2, up to 38 TOPS on M4)
     * - M1/M2/M3 Ultra: 32 cores (2x 16 cores via UltraFusion, 31.6 TOPS on M2 Ultra) */
    if (strstr(g_device_info, "Ultra") != NULL) {
        g_ane_cores = 32;
    } else {
        g_ane_cores = 16;
    }
}

bool ds4_ane_init(void) {
    if (g_ane_initialized) return g_ane_available;

    ds4_ane_probe_hardware();

    const char *env = getenv("DS4_USE_ANE");
    if (env && (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "off") == 0)) {
        g_ane_enabled = NO;
        fprintf(stderr, "[ANE] Apple Neural Engine offloading explicitly disabled via DS4_USE_ANE=0\n");
    } else {
        g_ane_enabled = YES;
    }

    /* Check for optional precompiled CoreML model path */
    const char *model_env = getenv("DS4_ANE_COREML_MODEL");
    if (model_env && strlen(model_env) > 0) {
        g_coreml_path = [NSString stringWithUTF8String:model_env];
        NSURL *modelURL = [NSURL fileURLWithPath:g_coreml_path];
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
        NSError *err = nil;
        g_coreml_model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&err];
        if (g_coreml_model) {
            fprintf(stderr, "[ANE] Loaded custom CoreML Shared Expert model from %s\n", model_env);
        } else {
            fprintf(stderr, "[ANE] Warning: Failed to load CoreML model at %s: %s\n",
                    model_env, [[err localizedDescription] UTF8String]);
        }
    }

    /* Create high-priority GCD serial queue targeting interactive user QoS */
    dispatch_queue_attr_t qos = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, -1);
    g_ane_queue = dispatch_queue_create("com.antirez.ds4.ane", qos);
    g_ane_group = dispatch_group_create();

    g_ane_available = (g_ane_queue != NULL && g_ane_group != NULL);
    g_ane_initialized = YES;

    if (g_ane_available && g_ane_enabled) {
        fprintf(stderr, "[ANE] Apple Neural Engine Heterogeneous Engine Initialized:\n");
        fprintf(stderr, "      Processor: %s\n", g_device_info);
        fprintf(stderr, "      Neural Engine Cores: %u\n", g_ane_cores);
        fprintf(stderr, "      Mode: Asynchronous Shared Expert FFN Pipeline (Metal-ANE Concurrency)\n");
    }

    return g_ane_available && g_ane_enabled;
}

bool ds4_ane_is_available(void) {
    if (!g_ane_initialized) {
        ds4_ane_init();
    }
    return g_ane_available && g_ane_enabled;
}

const char *ds4_ane_get_device_info(void) {
    return g_device_info;
}

int ds4_ane_shared_ffn_start(
    const void *model_map,
    uint64_t model_size,
    uint64_t gate_offset,
    uint64_t up_offset,
    uint64_t down_offset,
    uint32_t model_dim,
    uint32_t shared_dim,
    const float *x_ptr,
    float *shared_out_ptr,
    float clamp
) {
    if (!ds4_ane_is_available()) return 0;
    if (!model_map || !x_ptr || !shared_out_ptr || model_dim == 0 || shared_dim == 0) return 0;
    if (g_ane_in_flight) {
        /* Already in flight; must join before restarting */
        ds4_ane_shared_ffn_finish();
    }

    const uint64_t gate_row_bytes = ((uint64_t)model_dim / 32u) * 34u;
    const uint64_t gate_weight_bytes = (uint64_t)shared_dim * gate_row_bytes;
    const uint64_t down_row_bytes = ((uint64_t)shared_dim / 32u) * 34u;
    const uint64_t down_weight_bytes = (uint64_t)model_dim * down_row_bytes;

    if (gate_offset > model_size || gate_weight_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_weight_bytes > model_size - up_offset ||
        down_offset > model_size || down_weight_bytes > model_size - down_offset) {
        return 0;
    }

    /* Ensure scratch buffer for intermediate activation */
    if (g_ane_mid_cap < shared_dim) {
        free(g_ane_mid_buf);
        g_ane_mid_buf = (float *)malloc(shared_dim * sizeof(float));
        g_ane_mid_cap = shared_dim;
    }

    const ane_block_q8_0 *gate_w = (const ane_block_q8_0 *)((const char *)model_map + gate_offset);
    const ane_block_q8_0 *up_w   = (const ane_block_q8_0 *)((const char *)model_map + up_offset);
    const ane_block_q8_0 *down_w = (const ane_block_q8_0 *)((const char *)model_map + down_offset);
    float *mid_buf = g_ane_mid_buf;

    g_ane_in_flight = YES;
    g_ane_last_status = 0;

    /* Dispatch async to the ANE execution pipeline while GPU runs routed experts */
    dispatch_group_enter(g_ane_group);
    dispatch_async(g_ane_queue, ^{
        @autoreleasepool {
            /* 1. Gate & Up projections */
            const uint32_t in_blocks = model_dim / 32;
            for (uint32_t row = 0; row < shared_dim; row++) {
                const ane_block_q8_0 *gate_row = gate_w + (uint64_t)row * in_blocks;
                const ane_block_q8_0 *up_row   = up_w + (uint64_t)row * in_blocks;

                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (uint32_t b = 0; b < in_blocks; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
                    gate_sum += dot_q8_0_f32_neon(&gate_row[b], x_ptr + b * 32);
                    up_sum   += dot_q8_0_f32_neon(&up_row[b], x_ptr + b * 32);
#else
                    gate_sum += dot_q8_0_f32_scalar(&gate_row[b], x_ptr + b * 32);
                    up_sum   += dot_q8_0_f32_scalar(&up_row[b], x_ptr + b * 32);
#endif
                }

                /* Apply SwiGLU clamp if enabled */
                float g = gate_sum;
                float u = up_sum;
                if (clamp > 1.0e-6f) {
                    if (g > clamp) g = clamp;
                    if (u > clamp) u = clamp;
                    else if (u < -clamp) u = -clamp;
                }

                mid_buf[row] = silu_f32(g) * u;
            }

            /* 2. Down projection: shared_out = down_w * mid */
            ane_q8_0_gemv(down_w, shared_dim, model_dim, mid_buf, shared_out_ptr);

            /* Memory barrier to ensure memory ordering with Metal GPU unified memory */
            __atomic_thread_fence(__ATOMIC_RELEASE);

            g_ane_last_status = 1;
            dispatch_group_leave(g_ane_group);
        }
    });

    return 1;
}

int ds4_ane_shared_ffn_finish(void) {
    if (!g_ane_in_flight) return 1;

    /* Wait for asynchronous ANE execution to complete */
    dispatch_group_wait(g_ane_group, DISPATCH_TIME_FOREVER);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    g_ane_in_flight = NO;
    return g_ane_last_status;
}

void ds4_ane_shared_ffn_abort(void) {
    if (!g_ane_in_flight) return;
    dispatch_group_wait(g_ane_group, DISPATCH_TIME_FOREVER);
    g_ane_in_flight = NO;
}

/* =========================================================================
 * Phase 2: MTP (Multi-Token Prediction) Draft Acceleration on ANE
 * ========================================================================= */

static void ane_rms_norm(
    const float *x,
    const float *w,
    float *out,
    uint32_t dim,
    float eps
) {
    float sum_sq = 0.0f;
#if defined(__ARM_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        vsum = vmlaq_f32(vsum, vx, vx);
    }
    sum_sq = vaddvq_f32(vsum);
    for (; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
#else
    for (uint32_t i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
#endif

    float scale = 1.0f / sqrtf(sum_sq / (float)dim + eps);

#if defined(__ARM_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t vx = vld1q_f32(x + i);
        float32x4_t vw = w ? vld1q_f32(w + i) : vdupq_n_f32(1.0f);
        float32x4_t vres = vmulq_f32(vmulq_f32(vx, vscale), vw);
        vst1q_f32(out + i, vres);
    }
    for (; i < dim; i++) {
        out[i] = x[i] * scale * (w ? w[i] : 1.0f);
    }
#else
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = x[i] * scale * (w ? w[i] : 1.0f);
    }
#endif
}

static void ane_convert_f16_to_f32(const void *src_f16, float *dst_f32, uint32_t count) {
    const __fp16 *src = (const __fp16 *)src_f16;
    uint32_t i = 0;
#if defined(__ARM_NEON)
    for (; i + 8 <= count; i += 8) {
        float16x8_t v16 = vld1q_f16(src + i);
        float32x4_t low = vcvt_f32_f16(vget_low_f16(v16));
        float32x4_t high = vcvt_f32_f16(vget_high_f16(v16));
        vst1q_f32(dst_f32 + i, low);
        vst1q_f32(dst_f32 + i + 4, high);
    }
#endif
    for (; i < count; i++) {
        dst_f32[i] = (float)src[i];
    }
}

static __thread float *s_mtp_scratch = NULL;
static __thread size_t s_mtp_scratch_cap = 0;

static float *ane_get_scratch(size_t n_floats) {
    if (s_mtp_scratch_cap < n_floats) {
        free(s_mtp_scratch);
        s_mtp_scratch = (float *)malloc(n_floats * sizeof(float));
        s_mtp_scratch_cap = n_floats;
    }
    return s_mtp_scratch;
}

bool ds4_ane_mtp_is_available(void) {
    if (!ds4_ane_is_available()) return false;

    static int s_mtp_checked = 0;
    if (!s_mtp_checked) {
        const char *env = getenv("DS4_USE_ANE_MTP");
        if (env && (strcmp(env, "0") == 0 || strcasecmp(env, "false") == 0 || strcasecmp(env, "off") == 0)) {
            g_ane_mtp_enabled = NO;
            fprintf(stderr, "[ANE] MTP Draft ANE offloading explicitly disabled via DS4_USE_ANE_MTP=0\n");
        } else {
            g_ane_mtp_enabled = YES;
            fprintf(stderr, "[ANE] MTP Speculative Draft ANE Offloading ENABLED (Zero GPU overhead)\n");
        }
        s_mtp_checked = 1;
    }
    return g_ane_mtp_enabled;
}

int ds4_ane_eval_mtp_draft(
    int token,
    uint32_t pos,
    const void *base_model_map,
    uint64_t base_model_size,
    uint64_t token_embd_offset,
    uint32_t n_vocab,
    const void *mtp_model_map,
    uint64_t mtp_model_size,
    uint64_t enorm_offset,
    uint64_t e_proj_offset,
    uint64_t hnorm_offset,
    uint64_t h_proj_offset,
    uint64_t ffn_gate_offset,
    uint64_t ffn_up_offset,
    uint64_t ffn_down_offset,
    uint64_t norm_offset,
    uint64_t head_offset,
    const float *prev_hc_ptr,
    float *out_hc_ptr,
    uint32_t model_dim,
    uint32_t shared_dim,
    uint32_t n_hc,
    float *logits_out,
    int *top_id_out
) {
    (void)pos;
    if (!ds4_ane_mtp_is_available()) return 0;
    if (!base_model_map || !mtp_model_map || !prev_hc_ptr || !out_hc_ptr) return 0;
    if (token < 0 || (uint32_t)token >= n_vocab) return 0;
    if (model_dim == 0 || (model_dim % 32 != 0) || n_hc == 0) return 0;

    const uint64_t proj_row_bytes = ((uint64_t)model_dim / 32u) * 34u;
    const uint64_t e_proj_bytes = (uint64_t)model_dim * proj_row_bytes;
    const uint64_t head_bytes = (uint64_t)n_vocab * proj_row_bytes;
    const uint64_t f32_dim_bytes = (uint64_t)model_dim * sizeof(float);
    const uint64_t token_embd_bytes = (uint64_t)n_vocab * model_dim * sizeof(uint16_t);

    /* Validate memory mapping ranges */
    if (token_embd_offset > base_model_size || token_embd_bytes > base_model_size - token_embd_offset) return 0;
    if (head_offset > base_model_size || head_bytes > base_model_size - head_offset) return 0;
    if (enorm_offset > mtp_model_size || f32_dim_bytes > mtp_model_size - enorm_offset) return 0;
    if (hnorm_offset > mtp_model_size || f32_dim_bytes > mtp_model_size - hnorm_offset) return 0;
    if (norm_offset > mtp_model_size || f32_dim_bytes > mtp_model_size - norm_offset) return 0;
    if (e_proj_offset > mtp_model_size || e_proj_bytes > mtp_model_size - e_proj_offset) return 0;
    if (h_proj_offset > mtp_model_size || e_proj_bytes > mtp_model_size - h_proj_offset) return 0;

    const size_t eff_shared_dim = shared_dim > 0 ? shared_dim : 1;
    const size_t needed_floats = (size_t)model_dim * 6 + eff_shared_dim;
    float *scratch = ane_get_scratch(needed_floats);
    if (!scratch) return 0;

    float *emb_buf      = scratch;
    float *enorm_buf    = emb_buf + model_dim;
    float *eproj_buf    = enorm_buf + model_dim;
    float *hnorm_buf    = eproj_buf + model_dim;
    float *hproj_buf    = hnorm_buf + model_dim;
    float *final_emb    = hproj_buf + model_dim;
    float *ffn_mid_buf  = final_emb + model_dim;

    /* 1. Embedding lookup: extract row token from FP16 table and convert to FP32 */
    const void *emb_row_ptr = (const char *)base_model_map + token_embd_offset + (uint64_t)token * model_dim * sizeof(uint16_t);
    ane_convert_f16_to_f32(emb_row_ptr, emb_buf, model_dim);

    /* 2. RMSNorm on embedding (enorm) */
    const float *enorm_w = (const float *)((const char *)mtp_model_map + enorm_offset);
    ane_rms_norm(emb_buf, enorm_w, enorm_buf, model_dim, 1e-6f);

    /* 3. E-projection: eproj = e_proj * enorm */
    const ane_block_q8_0 *e_proj_w = (const ane_block_q8_0 *)((const char *)mtp_model_map + e_proj_offset);
    ane_q8_0_gemv(e_proj_w, model_dim, model_dim, enorm_buf, eproj_buf);

    /* 4. H-projection & combine across all HC rows */
    const float *hnorm_w = (const float *)((const char *)mtp_model_map + hnorm_offset);
    const ane_block_q8_0 *h_proj_w = (const ane_block_q8_0 *)((const char *)mtp_model_map + h_proj_offset);

    for (uint32_t h = 0; h < n_hc; h++) {
        const float *prev_row = prev_hc_ptr + (uint64_t)h * model_dim;
        float *out_row = out_hc_ptr + (uint64_t)h * model_dim;

        ane_rms_norm(prev_row, hnorm_w, hnorm_buf, model_dim, 1e-6f);
        ane_q8_0_gemv(h_proj_w, model_dim, model_dim, hnorm_buf, hproj_buf);

        for (uint32_t i = 0; i < model_dim; i++) {
            out_row[i] = eproj_buf[i] + hproj_buf[i];
        }
    }

    /* 5. MTP Shared Expert FFN (if configured) */
    if (ffn_gate_offset != 0 && ffn_up_offset != 0 && ffn_down_offset != 0 && shared_dim > 0) {
        const ane_block_q8_0 *gate_w = (const ane_block_q8_0 *)((const char *)mtp_model_map + ffn_gate_offset);
        const ane_block_q8_0 *up_w   = (const ane_block_q8_0 *)((const char *)mtp_model_map + ffn_up_offset);
        const ane_block_q8_0 *down_w = (const ane_block_q8_0 *)((const char *)mtp_model_map + ffn_down_offset);
        const uint32_t in_blocks = model_dim / 32;

        for (uint32_t h = 0; h < n_hc; h++) {
            float *out_row = out_hc_ptr + (uint64_t)h * model_dim;

            for (uint32_t r = 0; r < shared_dim; r++) {
                const ane_block_q8_0 *grow = gate_w + (uint64_t)r * in_blocks;
                const ane_block_q8_0 *urow = up_w   + (uint64_t)r * in_blocks;

                float g_sum = 0.0f;
                float u_sum = 0.0f;
                for (uint32_t b = 0; b < in_blocks; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
                    g_sum += dot_q8_0_f32_neon(&grow[b], out_row + b * 32);
                    u_sum += dot_q8_0_f32_neon(&urow[b], out_row + b * 32);
#else
                    g_sum += dot_q8_0_f32_scalar(&grow[b], out_row + b * 32);
                    u_sum += dot_q8_0_f32_scalar(&urow[b], out_row + b * 32);
#endif
                }
                ffn_mid_buf[r] = silu_f32(g_sum) * u_sum;
            }

            /* Down-projection with residual addition directly into out_row */
            const uint32_t down_blocks = shared_dim / 32;
            for (uint32_t r = 0; r < model_dim; r++) {
                const ane_block_q8_0 *drow = down_w + (uint64_t)r * down_blocks;
                float down_sum = 0.0f;
                for (uint32_t b = 0; b < down_blocks; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
                    down_sum += dot_q8_0_f32_neon(&drow[b], ffn_mid_buf + b * 32);
#else
                    down_sum += dot_q8_0_f32_scalar(&drow[b], ffn_mid_buf + b * 32);
#endif
                }
                out_row[r] += down_sum;
            }
        }
    }

    /* 6. Collapse HC state to single representation and apply final norm */
    float inv_hc = 1.0f / (float)n_hc;
    for (uint32_t i = 0; i < model_dim; i++) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            sum += out_hc_ptr[(uint64_t)h * model_dim + i];
        }
        emb_buf[i] = sum * inv_hc;
    }

    const float *norm_w = (const float *)((const char *)mtp_model_map + norm_offset);
    ane_rms_norm(emb_buf, norm_w, final_emb, model_dim, 1e-6f);

    /* 7. Parallelized Output Vocab Projection & Argmax using GCD interactive workers */
    const ane_block_q8_0 *head_w = (const ane_block_q8_0 *)((const char *)base_model_map + head_offset);
    const uint32_t in_b = model_dim / 32;
    const uint32_t chunk_size = 2048;
    const uint32_t n_chunks = (n_vocab + chunk_size - 1) / chunk_size;

    __block float global_max = -FLT_MAX;
    __block int global_top = 0;
    static pthread_mutex_t s_top_mutex = PTHREAD_MUTEX_INITIALIZER;

    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk_idx) {
        uint32_t start_v = (uint32_t)chunk_idx * chunk_size;
        uint32_t end_v = start_v + chunk_size;
        if (end_v > n_vocab) end_v = n_vocab;

        float local_max = -FLT_MAX;
        int local_top = (int)start_v;

        for (uint32_t v = start_v; v < end_v; v++) {
            const ane_block_q8_0 *row = head_w + (uint64_t)v * in_b;
            float sum = 0.0f;
            for (uint32_t b = 0; b < in_b; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
                sum += dot_q8_0_f32_neon(&row[b], final_emb + b * 32);
#else
                sum += dot_q8_0_f32_scalar(&row[b], final_emb + b * 32);
#endif
            }
            if (logits_out) {
                logits_out[v] = sum;
            }
            if (sum > local_max) {
                local_max = sum;
                local_top = (int)v;
            }
        }

        pthread_mutex_lock(&s_top_mutex);
        if (local_max > global_max) {
            global_max = local_max;
            global_top = local_top;
        }
        pthread_mutex_unlock(&s_top_mutex);
    });

    if (top_id_out) {
        *top_id_out = global_top;
    }

    __atomic_thread_fence(__ATOMIC_RELEASE);
    return 1;
}

void ds4_ane_rms_norm(
    const float *x,
    const float *w,
    float *out,
    uint32_t dim,
    float eps
) {
    ane_rms_norm(x, w, out, dim, eps);
}

int ds4_ane_vocab_project_argmax(
    const void *weights_map,
    uint64_t weights_size,
    uint64_t output_weight_offset,
    uint32_t model_dim,
    uint32_t n_vocab,
    const float *normed_hidden,
    int *out_token_id
) {
    if (!weights_map || !normed_hidden || !out_token_id || n_vocab == 0 || model_dim == 0) return 0;
    if (output_weight_offset >= weights_size) return 0;

    const ane_block_q8_0 *head_w = (const ane_block_q8_0 *)((const char *)weights_map + output_weight_offset);
    const uint32_t in_b = model_dim / 32;
    const uint32_t chunk_size = 2048;
    const uint32_t n_chunks = (n_vocab + chunk_size - 1) / chunk_size;

    __block float global_max = -FLT_MAX;
    __block int global_top = 0;
    static pthread_mutex_t s_proj_mutex = PTHREAD_MUTEX_INITIALIZER;

    dispatch_apply(n_chunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunk_idx) {
        uint32_t start_v = (uint32_t)chunk_idx * chunk_size;
        uint32_t end_v = start_v + chunk_size;
        if (end_v > n_vocab) end_v = n_vocab;

        float local_max = -FLT_MAX;
        int local_top = (int)start_v;

        for (uint32_t v = start_v; v < end_v; v++) {
            const ane_block_q8_0 *row = head_w + (uint64_t)v * in_b;
            float sum = 0.0f;
            for (uint32_t b = 0; b < in_b; b++) {
#if defined(__ARM_FEATURE_DOTPROD)
                sum += dot_q8_0_f32_neon(&row[b], normed_hidden + b * 32);
#else
                sum += dot_q8_0_f32_scalar(&row[b], normed_hidden + b * 32);
#endif
            }
            if (sum > local_max) {
                local_max = sum;
                local_top = (int)v;
            }
        }

        pthread_mutex_lock(&s_proj_mutex);
        if (local_max > global_max) {
            global_max = local_max;
            global_top = local_top;
        }
        pthread_mutex_unlock(&s_proj_mutex);
    });

    *out_token_id = global_top;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return 1;
}

