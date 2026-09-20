#ifndef DS4_ANE_H
#define DS4_ANE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the Apple Neural Engine subsystem.
 * Probes hardware, checks Apple Silicon capability, and sets up execution queues.
 * Returns true if ANE runtime is initialized and available. */
bool ds4_ane_init(void);

/* Check whether ANE acceleration is available and enabled.
 * Controlled by DS4_USE_ANE environment variable (default 1 if hardware supported). */
bool ds4_ane_is_available(void);

/* Returns human-readable status/device info of ANE. */
const char *ds4_ane_get_device_info(void);

/* Begin asynchronous execution of the Shared Expert FFN on ANE.
 *
 * Mathematically evaluates:
 *   gate = Gate_Weight * x
 *   up   = Up_Weight * x
 *   mid  = silu(clamp(gate)) * clamp(up)
 *   out  = Down_Weight * mid
 *
 * Parameters:
 *   model_map: pointer to base mmap'd model file
 *   model_size: total size of mmap'd model
 *   gate_offset: offset of gate weight tensor in model (Q8_0)
 *   up_offset: offset of up weight tensor in model (Q8_0)
 *   down_offset: offset of down weight tensor in model (Q8_0)
 *   model_dim: input/output dimension (e.g. 2048)
 *   shared_dim: hidden dimension of shared expert (e.g. 1536)
 *   x_ptr: pointer to input activation vector (model_dim floats)
 *   shared_out_ptr: pointer to destination vector for shared expert result (model_dim floats)
 *   clamp: SwiGLU clamp value
 *
 * Returns 1 on successful dispatch, 0 on failure / fallback needed. */
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
);

/* Wait for currently in-flight ANE shared expert computation to finish.
 * Guarantees shared_out_ptr memory is coherent and ready for GPU consumption.
 * Returns 1 on success, 0 on error. */
int ds4_ane_shared_ffn_finish(void);

/* Abort any in-flight ANE computation and reset state. */
void ds4_ane_shared_ffn_abort(void);

/* Check whether MTP draft offload on ANE is available and enabled.
 * Controlled by DS4_USE_ANE_MTP environment variable (default 1). */
bool ds4_ane_mtp_is_available(void);

/* Evaluate MTP draft token on Apple Neural Engine.
 *
 * Dispatches MTP input projection (token embedding + enorm + eproj + hnorm + hproj),
 * single-layer Transformer block (Attention + Shared Expert FFN), and output head
 * asynchronously on ANE, extracting the top predicted token ID without GPU overhead.
 *
 * Returns 1 on success, 0 on fallback. */
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
);

/* Compute RMSNorm on host CPU/NEON: out = (x / rms(x)) * w */
void ds4_ane_rms_norm(
    const float *x,
    const float *w,
    float *out,
    uint32_t dim,
    float eps
);

/* Evaluate Vocabulary Projection Head on Apple Neural Engine / NEON multi-core:
 * Computes dot products of normed_hidden (model_dim) against Q8_0 weights,
 * finding the Top-1 token ID with maximum logit.
 *
 * Parameters:
 *   weights_map: base pointer to mmap'd model file
 *   weights_size: total size of mmap'd model file
 *   output_weight_offset: byte offset to Q8_0 output weight tensor (dim: [model_dim, n_vocab])
 *   model_dim: embedding dimension (e.g. 2048 or 4096)
 *   n_vocab: vocabulary size (e.g. 128256 or 151552)
 *   normed_hidden: normalized hidden state vector (model_dim floats)
 *   out_token_id: pointer to receive the argmax token ID
 *
 * Returns 1 on success, 0 on failure.
 */
int ds4_ane_vocab_project_argmax(
    const void *weights_map,
    uint64_t weights_size,
    uint64_t output_weight_offset,
    uint32_t model_dim,
    uint32_t n_vocab,
    const float *normed_hidden,
    int *out_token_id
);

#ifdef __cplusplus
}
#endif

#endif /* DS4_ANE_H */
