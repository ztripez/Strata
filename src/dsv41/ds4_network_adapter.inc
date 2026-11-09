// Included inside Model::Impl. Shared delivery policy for synchronous and
// asynchronous native CPU misses; local tier selection and arithmetic stay native.
// picks contains (expert, token*kPicks+slot); weights uses that same native stride.
// Only a successful remote result removes picks. Runtime worker loss leaves
// every original pick for local recomputation, reports the failure and disables
// this channel for this Model. Configuration failures still reject load().
bool offload_cpu_picks(const Layer& l, int T, const float* rows, size_t stride, const float* weights,
    std::vector<std::pair<int,int>>& picks, std::vector<float>& sums, std::vector<uint8_t>& mask) {
    sums.clear(); mask.clear();
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
    std::vector<float> activation(static_cast<size_t>(T)*g.dim);
    for (int t=0;t<T;++t)
        std::copy(rows+static_cast<size_t>(t)*stride,rows+static_cast<size_t>(t)*stride+g.dim,
                  activation.data()+static_cast<size_t>(t)*g.dim);
    sums.resize(static_cast<size_t>(T)*g.dim);mask.resize(T);
    std::string error;
    if (net_worker->execute_sync(l.il,T,activation.data(),remote,sums.data(),mask.data(),&error)) {
        picks.swap(local);return true;
    }
    net_warned=true;sums.clear();mask.clear();
    std::fprintf(stderr,"ds4-network: layer %d failed (%s); recomputing on local CPU and disabling remote execution\n",l.il,error.c_str());
    return false;
}
