"""GPU-resident feature bank and episode sampling for meta-training / distillation."""
import torch


class FeatureBank:
    def __init__(self, path, device):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.encm = d["encm"].to(device)  # (N, L, D) fp16
        self.act = d["act"].to(device)  # (N, chunk, A) fp16, normalized, clamped
        self.ptr = d["ptr"]  # (n_tasks, n_demos, 2): start, length
        self.task_emb = d["task_emb"]
        self.descs = d["descs"]
        self.device = device

    def demo_index(self, task, demo):
        s, n = self.ptr[task, demo].tolist()
        return torch.arange(s, s + n, device=self.device)

    def demo(self, task, demo):
        """All frames of one demonstration: (encm, act) in fp32."""
        i = self.demo_index(task, demo)
        return self.encm[i].float(), self.act[i].float()

    def task_frames(self, task, demos):
        return torch.cat([self.demo_index(task, d) for d in demos])

    def gather(self, idx):
        return self.encm[idx].float(), self.act[idx].float()

    def n_demos(self, task):
        if task >= self.ptr.shape[0]:  # partial banks (rehearsals) hold only the first tasks
            return 0
        return int((self.ptr[task, :, 1] > 0).sum())

    def sample_episodes(self, tasks, K, F, Q, g, n_query_demos=2):
        """Support: K demos x F evenly spaced frames (random phase). Query: Q frames from
        n_query_demos other demos of the same task (never in the support).

        Returns sup_idx (K, E*F) episode-major within each write, qry_idx (E*Q,).
        """
        sup, qry = [], []
        for t in tasks:
            assert self.n_demos(t) >= K + n_query_demos, f"task {t}: {self.n_demos(t)} demos < K={K} + queries"
            perm = torch.randperm(self.n_demos(t), generator=g)
            sdemos, qdemos = perm[:K].tolist(), perm[K : K + n_query_demos].tolist()
            rows = []
            for d in sdemos:
                s, n = self.ptr[t, d].tolist()
                pos = ((torch.arange(F) + torch.rand(F, generator=g)) * n / F).long().clamp(max=n - 1)
                rows.append(s + pos)
            sup.append(torch.stack(rows))  # (K, F)
            qi = torch.cat([torch.arange(*self._span(t, d)) for d in qdemos])
            qry.append(qi[torch.randint(len(qi), (Q,), generator=g)])
        sup = torch.stack(sup, dim=1).reshape(K, -1)  # (K, E*F)
        return sup.to(self.device), torch.cat(qry).to(self.device)

    def sample_multitask_episodes(self, groups, K, F, Q, g):
        """groups: E lists of M tasks. Writes are the tasks' demos in stream order (task by task,
        K each); queries are Q frames spread over the M tasks. Returns sup (M*K, E*F), qry (E*Q,)."""
        M = len(groups[0])
        per_task = [self.sample_episodes([grp[m] for grp in groups], K, F, max(1, Q // M), g) for m in range(M)]
        sup = torch.cat([ps[0] for ps in per_task], 0)  # (M*K, E*F)
        qm = [ps[1].reshape(len(groups), -1) for ps in per_task]  # each (E, Q/M)
        qry = torch.cat(qm, 1).reshape(-1)  # episode-major
        return sup, qry

    def _span(self, t, d):
        s, n = self.ptr[t, d].tolist()
        return s, s + n
