// Reject incompatible metadata before a DeepSeek worker hashes or loads weights.
#include "gguf_fixture.hpp"
#include "strata/core/deepseek_worker_geometry.hpp"
#include <chrono>
#include <cstdio>
#include <limits>

namespace {
std::vector<fixture::Kv> metadata() {
    using namespace fixture;
    return {str("general.architecture", "deepseek41"),
            u32("deepseek41.block_count", 40), u32("deepseek41.expert_count", 384),
            u32("deepseek41.embedding_length", 5120), u32("deepseek41.expert_feed_forward_length", 2304),
            u32("deepseek41.expert_used_count", 6),
            u32s("deepseek41.swiglu_clamp_exp", std::vector<uint64_t>(40, 10))};
}
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() /
        ("strata-deepseek-geometry-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directories(dir);
        int tests = 0;
        auto run = [&](const std::vector<fixture::Kv> &meta, bool valid, const char *name) {
            const auto path = dir / (std::to_string(tests++) + ".gguf");
            fixture::write(path, meta, {});
            // Loader errors must not be mistaken for a successful geometry refusal.
            const auto model = strata::GgufModel::open(path.string());
            bool accepted = false;
            try {
                const strata::core::DeepSeekWorkerGeometry geometry(model);
                accepted = geometry.picks == 6 && geometry.limit == 10;
            } catch (const std::runtime_error &) {
                if (valid) throw;
            }
            if (accepted != valid) throw std::runtime_error(std::string("geometry verdict: ") + name);
            std::printf("%s PASS\n", name);
        };
        run(metadata(), true, "pinned geometry");
        for (const auto &bad : std::vector<fixture::Kv>{
                 fixture::str("general.architecture", "qwen35moe"),
                 fixture::u32("deepseek41.expert_used_count", 5),
                 fixture::u32("deepseek41.embedding_length", 0),
                 fixture::u32("deepseek41.expert_count", 65537),
                 fixture::f32("deepseek41.block_count", 40.5f),
                 fixture::f32("deepseek41.block_count", std::numeric_limits<float>::infinity()),
                 fixture::str("deepseek41.expert_count", "384"),
                 fixture::u32s("deepseek41.swiglu_clamp_exp", std::vector<uint64_t>(39,10)),
                 fixture::strs("deepseek41.swiglu_clamp_exp", std::vector<std::string>(40,"10"))}) {
            auto meta = metadata();
            for (auto &item : meta) if (item.key == bad.key) item = bad;
            run(meta, false, bad.key.c_str());
        }
        auto meta = metadata(); meta.back().arr[39] = 9;
        run(meta, false, "nonuniform clamp");
        meta = metadata(); meta.erase(meta.begin()+5);
        run(meta, false, "missing effective expert count");
        std::filesystem::remove_all(dir);
        std::printf("DeepSeek geometry: %d cases PASS\n", tests);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s (fixtures retained at %s)\n", e.what(), dir.string().c_str());
        return 1;
    }
}
