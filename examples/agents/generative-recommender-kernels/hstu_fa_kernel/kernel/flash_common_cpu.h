#include <c10/util/Optional.h>
#include <torch/library.h> // @manual
#include <torch/nn/functional.h>

namespace hstu {

std::tuple<at::Tensor, std::optional<at::Tensor>> hstu_mha_fwd_dummy(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1);

std::vector<at::Tensor> hstu_mha_bwd_dummy(
    int64_t max_seq_len,
    double alpha,
    at::Tensor& dout,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& out,
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    bool sort_by_length,
    bool const deterministic,
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    const std::optional<at::Tensor>& softmax_lse = std::nullopt,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1);

std::tuple<at::Tensor, std::optional<at::Tensor>> hstu_mha_fwd_meta(
    const at::SymInt max_seq_len,
    double alpha,
    at::Tensor& q, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& k, // (b, s, h, d) or (total_s, h, d)
    at::Tensor& v, // (b, s, h, d) or (total_s, h, d)
    const std::optional<at::Tensor>& seq_offsets,
    bool causal,
    const std::optional<at::Tensor>& num_targets,
    const std::optional<at::Tensor>& attn_scale,
    int64_t max_attn_len,
    int64_t min_full_attn_seq_len,
    int64_t contextual_seq_len,
    const std::optional<at::Tensor>& q_descale, // (b, h_k), not (b, h)
    const std::optional<at::Tensor>& k_descale, // (b, h_k)
    const std::optional<at::Tensor>& v_descale, // (b, h_k)
    const int64_t sm_margin = 0,
    int64_t max_q_len = 0,
    const std::optional<at::Tensor>& seq_offsets_q = std::nullopt,
    int64_t num_softmax_heads = 0,
    bool training = true,
    const std::optional<at::Tensor>& max_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& contextual_seq_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& max_attn_len_tensor = std::nullopt,
    const std::optional<at::Tensor>& min_full_attn_seq_len_tensor =
        std::nullopt,
    int64_t num_groups = 1);
} // namespace hstu
