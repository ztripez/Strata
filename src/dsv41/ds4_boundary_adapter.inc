// Included inside Model::Impl: E2 host adapter around the authoritative layer(),
// plan_window(), head and attention-state paths. No alternative arithmetic/cache.
bool configure_boundary_probe(int cut, const strata::core::DeepSeekBoundaryHeader& identity,
    std::function<bool(const strata::core::DeepSeekLayerBoundary&, std::string&)> observer, std::string& err) {
    if ((cut != 20 && cut != 24) || pos != 0 || !toks.empty() || !ctoks.empty() || probe_cut >= 0 || identity.session <= probe_last_session ||
        devs.size() != 1 || par || !sync_decode || chunk != 0 || g.nl != 40 || g.dim != 5120 || g.hc != 4 ||
        g.n_mtp || o.seqs != 1 || o.vision || o.expert_cache || o.vram_experts != 0 || !pk.gguf || !file_experts || store || net_worker ||
        disk_skip_w > 0 || ram_skip_w > 0) {
        err = "E2 probe requires reset, a fresh session, cut 20/24, one synchronous GPU, windows, no expert cache/VRAM/store/network/skip/vision/DSpark";
        return false;
    }
    for (int il = 0; il < g.nl; ++il)
        if (L[il].k_exp != 6 || L[il].n_vram != 0) {
            err = "E2 probe requires six effective routed experts and zero effective GPU expert residency on every layer";
            return false;
        }
    try {
        auto layout = strata::core::DeepSeekBoundaryLayout{static_cast<uint32_t>(g.dim), static_cast<uint32_t>(g.hc),
            static_cast<uint32_t>(topk_stride()), static_cast<uint32_t>(cand_stride()), static_cast<uint32_t>(g.vocab),
            static_cast<uint64_t>(o.max_context), {}};
        for (size_t si = 0; si < srcs.size(); ++si) {
            const auto &source = srcs[si];
            if (source.layer >= cut) continue;
            bool kv = false, keys = false;
            for (int il = cut; il < g.nl; ++il) {
                kv = kv || L[il].src == static_cast<int>(si);
                keys = keys || (L[il].ix_src && L[il].ik_src == static_cast<int>(si));
            }
            if (!kv && !keys) continue;
            if (!kv) { err = "E2 key-only source transfer is not qualified"; return false; }
            layout.sources.push_back({static_cast<uint32_t>(source.layer), static_cast<uint32_t>(source.ratio),
                                      static_cast<uint32_t>(ckv_rb()), keys ? static_cast<uint32_t>(ik_rb()) : 0});
        }
        auto header = identity;
        header.position = 0; header.sequence = 1; header.rows = 1; header.next_layer = cut;
        strata::core::deepseek_boundary_detail::layout(layout, header);
        probe_identity = identity; probe_layout = std::move(layout);
        probe_cut = cut; probe_last_session = identity.session; probe_sequence = 0; probe_poisoned = false;
        std::fill(cur_topk_n,cur_topk_n+kWin,0);
        std::fill(cur_cand_n,cur_cand_n+kWin,0);
        probe_observer = std::move(observer);
        return true;
    } catch (const std::exception &e) { err = e.what(); return false; }
}

// The candidate count is updated at candidate_source_layer_id and carried to
// later indexers. The selection count follows the last indexer before this cut.
void boundary_counts(int64_t p0, int T, std::vector<uint32_t>& top, std::vector<uint32_t>& cand) const {
    WinPlan plan;
    int top_counts[kWin] = {}, cand_counts[kWin] = {};
    plan_index_counts(p0,T,plan,top_counts,cand_counts,probe_cut);
    top.assign(top_counts,top_counts+T); cand.assign(cand_counts,cand_counts+T);
}

bool emit_boundary(int64_t p0, int T, std::string& err) {
    if (!probe_observer || probe_poisoned) { err = "boundary source session is unavailable"; return false; }
    probe_poisoned = true; // Once prefix state changed, any incomplete handoff aborts this session.
    try {
        strata::core::DeepSeekLayerBoundary b;
        b.header = probe_identity;
        b.header.position = p0; b.header.rows = T; b.header.next_layer = probe_cut;
        b.header.sequence = probe_sequence+1;
        b.tokens.assign(toks.begin()+p0,toks.begin()+p0+T);
        boundary_counts(p0,T,b.topk_valid,b.cand_valid);
        Dev &d = devs[0];
        CK(cudaSetDevice(d.id)); CK(cudaStreamSynchronize(d.st));
        auto get = [&](auto &host, const void *device, size_t count) -> bool {
            using Value = typename std::decay_t<decltype(host)>::value_type;
            host.resize(count);
            if (count) CK(cudaMemcpy(host.data(),device,count*sizeof(Value),cudaMemcpyDeviceToHost));
            return true;
        };
        if (!get(b.h,d.h,static_cast<size_t>(T)*g.hc*g.dim) || !get(b.pre,d.pre,static_cast<size_t>(T)*8) ||
            !get(b.topk,d.topk,static_cast<size_t>(T)*topk_stride()) || !get(b.cand,d.cand,static_cast<size_t>(T)*cand_stride())) return false;
        for (const auto &spec : probe_layout.sources) {
            const Src *source = nullptr;
            for (const auto &s : srcs) if (s.layer == static_cast<int>(spec.layer)) source = &s;
            if (!source) { err = "missing audited source layer"; return false; }
            strata::core::DeepSeekBoundaryCache cache;
            cache.source_layer = spec.layer; cache.first_row = p0/spec.ratio;
            cache.rows = static_cast<uint32_t>((p0+T)/spec.ratio-cache.first_row);
            if (!get(cache.kv,source->ckv[0]+cache.first_row*spec.kv_row_bytes,static_cast<size_t>(cache.rows)*spec.kv_row_bytes) ||
                (spec.key_row_bytes && !get(cache.keys,source->ik[0]+cache.first_row*spec.key_row_bytes,static_cast<size_t>(cache.rows)*spec.key_row_bytes))) return false;
            b.caches.push_back(std::move(cache));
        }
        strata::core::deepseek_boundary_detail::validate(probe_layout,b);
        if (!probe_observer(b,err)) return false;
        ++probe_sequence; probe_poisoned = false;
        return true;
    } catch (const std::exception &e) { err = e.what(); return false; }
}

bool replay_boundary(const strata::core::DeepSeekLayerBoundary& b, int logits_rows, std::string& err) {
    if (probe_cut < 0 || probe_observer || probe_poisoned || logits_rows < 0 || logits_rows > static_cast<int>(b.header.rows)) {
        err = "boundary suffix session is unavailable or logits row count is invalid"; return false;
    }
    try {
        strata::core::deepseek_boundary_detail::validate(probe_layout,b);
        auto expected = probe_identity;
        expected.position = pos; expected.sequence = probe_sequence+1; expected.rows = b.header.rows; expected.next_layer = probe_cut;
        if (!strata::core::deepseek_boundary_detail::equal(expected,b.header)) {
            err = "stale, duplicate or out-of-order boundary capture"; return false;
        }
        std::vector<uint32_t> top, cand;
        boundary_counts(pos,b.header.rows,top,cand);
        if (top != b.topk_valid || cand != b.cand_valid) { err = "boundary selection counts disagree with the model plan"; return false; }
        halted = false;
        if (cancel && cancel()) { halted = true; err = "boundary replay cancelled before mutation"; return false; }
        probe_poisoned = true;
        const int T = static_cast<int>(b.header.rows);
        const auto p0 = pos;
        slot_held = false;
        for (int t = 0; t < T; ++t) {
            toks.push_back(b.tokens[t]);
            ctoks.push_back(ctok(p0+t,b.tokens[t]));
        }
        Dev &d = devs[0];
        CK(cudaSetDevice(d.id));
        auto put = [&](void *device, const auto &host) -> bool {
            using Value = typename std::decay_t<decltype(host)>::value_type;
            if (!host.empty()) CK(cudaMemcpyAsync(device,host.data(),host.size()*sizeof(Value),cudaMemcpyHostToDevice,d.st));
            return true;
        };
        if (!put(d.h,b.h) || !put(d.pre,b.pre) || !put(d.topk,b.topk) || !put(d.cand,b.cand)) return false;
        for (const auto &cache : b.caches) {
            Src *source = nullptr;
            for (auto &s : srcs) if (s.layer == static_cast<int>(cache.source_layer)) source = &s;
            if (!source) { err = "missing destination cache source"; return false; }
            if (!put(source->ckv[0]+cache.first_row*ckv_rb(),cache.kv) ||
                (!cache.keys.empty() && !put(source->ik[0]+cache.first_row*ik_rb(),cache.keys))) return false;
        }
        wplan.nrow = 0; plan_window(p0,T,wplan); n_logits = logits_rows;
        if (!forward_serial(p0,T,err,probe_cut)) return false;
        CK(cudaStreamSynchronize(d.st));
        pos += T; st.tokens += T; ++probe_sequence; probe_poisoned = false;
        return true;
    } catch (const std::exception &e) { err = e.what(); return false; }
}

bool boundary_state(std::vector<uint8_t>& out, std::string& err) {
    if (probe_cut < 0 || probe_poisoned) { err = "no complete boundary state"; return false; }
    Dev &d = devs[0];
    CK(cudaSetDevice(d.id)); CK(cudaStreamSynchronize(d.st));
    strata::core::ExpertWireWriter w;
    w.put(static_cast<uint32_t>(probe_cut)); w.put(static_cast<uint64_t>(pos));
    w.put(probe_sequence);
    w.put(static_cast<uint64_t>(toks.size())); for (const auto token : toks) w.put(token);
    w.put(static_cast<uint64_t>(ctoks.size())); for (const auto token : ctoks) w.put(token);
    for (const auto count : cur_topk_n) w.put(count);
    for (const auto count : cur_cand_n) w.put(count);
    auto get = [&](const void *device, size_t bytes) -> bool {
        w.put(static_cast<uint64_t>(bytes));
        const auto offset = w.bytes.size(); w.bytes.resize(offset+bytes);
        if (bytes) CK(cudaMemcpy(w.bytes.data()+offset,device,bytes,cudaMemcpyDeviceToHost));
        return true;
    };
    for (int il = probe_cut; il < g.nl; ++il) {
        const auto &l = L[il];
        w.put(static_cast<uint32_t>(il));
        if (!get(l.win,static_cast<size_t>(g.win)*win_rb()) ||
            !get(l.c_kv,l.c_kv ? static_cast<size_t>(l.ratio)*g.hd*4 : 0) ||
            !get(l.c_score,l.c_score ? static_cast<size_t>(l.ratio)*g.hd*4 : 0)) return false;
    }
    for (size_t si = 0; si < srcs.size(); ++si) {
        bool kv = false, keys = false;
        for (int il = probe_cut; il < g.nl; ++il) {
            kv = kv || L[il].src == static_cast<int>(si);
            keys = keys || (L[il].ix_src && L[il].ik_src == static_cast<int>(si));
        }
        if (!kv && !keys) continue;
        const auto &s = srcs[si]; const size_t rows = static_cast<size_t>(pos/s.ratio);
        w.put(static_cast<uint32_t>(s.layer)); w.put(static_cast<uint64_t>(rows));
        if (!get(s.ckv[0],kv ? rows*ckv_rb() : 0) || !get(s.ik[0],keys ? rows*ik_rb() : 0)) return false;
    }
    out = std::move(w.bytes);
    return true;
}
