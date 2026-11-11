// Included inside Model::Impl. Remote CPU misses retain the native expert-ID
// reduction order. Each existing RPC carries one expert with unit weights, so
// its unweighted rows can be combined with local rows on the layer owner.
struct NetworkCpuParts {
    std::vector<strata::core::ExpertWorkerAssignment> assignments;
    std::vector<float> values;
};
struct CpuContribution {
    int expert, token;
    float weight;
    const float* values;
};

bool offload_cpu_picks(const Layer& l, int T, const float* rows, size_t stride, const float* weights,
    std::vector<std::pair<int,int>>& picks, NetworkCpuParts& parts) {
    parts.assignments.clear(); parts.values.clear();
    if (!net_worker || net_warned || !net_worker->healthy() || l.mtp || !net_worker->holds_layer(l.il)) return false;
    std::vector<strata::core::ExpertWorkerAssignment> remote;
    std::vector<std::pair<int,int>> local;
    for (const auto pick : picks) {
        const int t=pick.second/kPicks, slot=pick.second%kPicks;
        if (net_worker->holds(l.il,pick.first))
            remote.push_back({static_cast<uint16_t>(t),static_cast<uint16_t>(slot),pick.first,weights[pick.second]});
        else local.push_back(pick);
    }
    if (remote.empty()) return false;
    std::stable_sort(remote.begin(),remote.end(),[](const auto& a,const auto& b) {
        return a.expert_id<b.expert_id || (a.expert_id==b.expert_id && a.token_idx<b.token_idx);
    });
    std::vector<float> activation(static_cast<size_t>(T)*g.dim);
    for (int t=0;t<T;++t)
        std::copy(rows+static_cast<size_t>(t)*stride,rows+static_cast<size_t>(t)*stride+g.dim,
                  activation.data()+static_cast<size_t>(t)*g.dim);
    std::vector<float> result(static_cast<size_t>(T)*g.dim);
    std::vector<uint8_t> mask(T);
    std::string error;
    for (size_t first=0;first<remote.size();) {
        size_t end=first+1;
        while (end<remote.size() && remote[end].expert_id==remote[first].expert_id) ++end;
        std::vector<strata::core::ExpertWorkerAssignment> request(remote.begin()+first,remote.begin()+end);
        for (auto& assignment:request) assignment.weight=1.0f;
        if (!net_worker->execute_sync(l.il,T,activation.data(),request,result.data(),mask.data(),&error)) {
            net_warned=true;parts.assignments.clear();parts.values.clear();
            std::fprintf(stderr,"ds4-network: layer %d failed (%s); recomputing on local CPU and disabling remote execution\n",l.il,error.c_str());
            return false;
        }
        for (size_t i=first;i<end;++i) {
            parts.assignments.push_back(remote[i]);
            const float* row=result.data()+static_cast<size_t>(remote[i].token_idx)*g.dim;
            parts.values.insert(parts.values.end(),row,row+g.dim);
        }
        first=end;
    }
    picks.swap(local);return true;
}

// Reuse the native float multiply/add and expert ordering for both CPU paths.
// Pointers borrow local CPU scratch or the completed remote result for this call.
void combine_network_cpu_parts(const NetworkCpuParts& parts,
    std::vector<CpuContribution> contributions, int T, float* output, size_t stride) {
    for (size_t i=0;i<parts.assignments.size();++i) {
        const auto& a=parts.assignments[i];
        contributions.push_back({a.expert_id,a.token_idx,a.weight,parts.values.data()+i*g.dim});
    }
    std::stable_sort(contributions.begin(),contributions.end(),[](const auto& a,const auto& b) {
        return a.expert<b.expert || (a.expert==b.expert && a.token<b.token);
    });
    for (int t=0;t<T;++t) std::fill(output+t*stride,output+t*stride+g.dim,0.0f);
    for (const auto& part:contributions) {
        float* row=output+static_cast<size_t>(part.token)*stride;
        for (int c=0;c<g.dim;++c) row[c]+=part.weight*part.values[c];
    }
}
