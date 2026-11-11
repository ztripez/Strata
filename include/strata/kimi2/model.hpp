#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace strata { class GgufModel; struct TensorInfo; }
namespace strata::kimi2 {
/// K2 tensor/geometry adapter over Strata's existing immutable sharded GGUF mappings.
/// It owns no decoded-weight cache, pager, placement policy or transport.
class Model {
public:
    /// Open all shards and require every native K2 layer, expert table and output tensor.
    explicit Model(const std::string& first_shard);
    /// Return the original bounds-checked GGUF tensor descriptor.
    const TensorInfo& tensor(const std::string& name) const;
    /// Original GGUF quantization ID and packed byte count.
    int type(const std::string& name) const;
    size_t bytes(const std::string& name) const;
    /// Borrow original packed bytes; valid for this model's lifetime.
    const uint8_t* data(const std::string& name) const;
    /// Borrow one expert's original gate/up/down matrix without copying its table.
    const uint8_t* expert(int layer, const std::string& role, int id) const;
    /// Decode only the requested embedding rows with Strata's existing ggml decoder.
    std::vector<float> embeddings(const std::vector<int32_t>& ids) const;
    /// Instruct generation stop token from the pinned GGUF tokenizer metadata.
    int32_t eos_token_id() const;
    /// K2 YaRN inverse frequencies in the native adjacent-pair layout.
    const std::array<float,32>& rope_frequencies() const { return rope_; }
private:
    std::shared_ptr<GgufModel> mapped_;
    std::array<float,32> rope_{};
};
}
