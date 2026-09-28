"""Fast -> slow: sequential consolidation of the novel tasks into the slow decoder.

For each held-out LIBERO-90 task j, in order:
  1. teacher = current slow policy + fast memory written with K demos of task j
     (arm 'ft' has no teacher: it fine-tunes on the K demos directly -- the positive control).
  2. collect: teacher rollouts from init states disjoint from evaluation (offset 25), recording
     every visited encoder summary; teacher success is measured on these rollouts.
  3. consolidate: endpoint FM distillation into the slow decoder (proposal eq. 22: teacher
     actions sampled fresh at every step, paired with fresh source noise), mixed 1:1 with
     replay of old training-task demos and of earlier consolidated tasks.
  4. reset the fast memory; evaluate the memory-off student on task j.
Memories are written on whatever the slow policy is at that time, so later tasks also test
whether a memory meta-trained on the original backbone survives backbone drift.
Final: student on all novel tasks (forgetting), 20 old training tasks (retention), LIBERO-10.

    python -m horsea.consolidate --arm fmw
Resumable (progress.pt after every task).
"""
import argparse
import copy
import json
import os
import time

import torch

import horsea  # noqa: F401
from horsea.adapt_eval import load_memory, write_demos
from horsea.bank import FeatureBank
from horsea.base import DECODER_PREFIXES, Flow, load_policy
from horsea.finetune import demo_batcher, train_decoder
from horsea.paths import BASE_CKPT, EXP, FEAT_DIR, HELDOUT_90, LIBERO_10, RETENTION_90, TRAIN_90
from horsea.rollout import install_sampler, make_runner, run_task


def decoder_state(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.velocity_net.state_dict().items()
            if k.startswith(DECODER_PREFIXES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ft", "ttt", "ttt2", "kv", "fmw", "res"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--tasks", type=int, nargs="+", default=HELDOUT_90)
    ap.add_argument("--n_collect", type=int, default=20)
    ap.add_argument("--collect_offset", type=int, default=25)
    ap.add_argument("--n_eval", type=int, default=20)
    ap.add_argument("--n_retention", type=int, default=10)
    ap.add_argument("--n_l10", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--store", type=int, default=4000, help="replay samples kept per consolidated task")
    ap.add_argument("--par", type=int, default=5)
    ap.add_argument("--mem_dir", default=os.path.join(EXP, "memory"))
    ap.add_argument("--ckpt", default=BASE_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--retention_tasks", type=int, nargs="+", default=RETENTION_90)
    ap.add_argument("--l10_tasks", type=int, nargs="+", default=LIBERO_10)
    ap.add_argument("--horizon", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--features", default=FEAT_DIR)
    ap.add_argument("--generic", action="store_true",
                    help="teacher = memory written under the generic instruction (demos give the task); "
                         "student learns the same states under the task's real instruction")
    ap.add_argument("--teacher_base", action="store_true",
                    help="teacher = frozen original base + memory (no interface drift); only the student changes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = args.device
    out = args.out or os.path.join(EXP, "consolidate", args.arm)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    base, sd = load_policy(args.ckpt, dev)
    shape_meta = sd["config"]["task"]["shape_meta"]
    bank = FeatureBank(os.path.join(args.features, "libero_90.pt"), dev)
    old_idx = torch.cat([bank.task_frames(t, range(bank.n_demos(t))) for t in TRAIN_90 if bank.n_demos(t) > 0])
    memory = None if args.arm == "ft" else load_memory(args.arm, os.path.join(args.mem_dir, args.arm, "final.pt"), dev)
    gbank = FeatureBank(os.path.join(args.features, "libero_90_generic.pt"), dev) if args.generic else bank
    temb = gbank.task_emb[0] if args.generic else None

    student, buffers, results, j0 = base, [], [], 0
    prog = os.path.join(out, "progress.pt")
    if os.path.exists(prog):
        p = torch.load(prog, map_location="cpu", weights_only=False)
        student = copy.deepcopy(base)
        student.velocity_net.load_state_dict({**student.velocity_net.state_dict(), **p["decoder"]})
        student.eval().requires_grad_(False)
        buffers = [(e.to(dev), a.to(dev)) for e, a in p["buffers"]]
        results, j0 = p["results"], p["j"]
        print(f"resumed after task index {j0}", flush=True)

    def replay_batch():
        n_old = args.bs if not buffers else args.bs // 2
        e, a = bank.gather(old_idx[torch.randint(len(old_idx), (n_old,), device=dev)])
        if buffers:
            b = torch.randint(len(buffers), (args.bs - n_old,))
            parts = [(e, a)]
            for k in b.unique().tolist():
                be, ba = buffers[k]
                i = torch.randint(len(be), (int((b == k).sum()),), device=dev)
                parts.append((be[i].float(), ba[i].float()))
            e, a = torch.cat([q[0] for q in parts]), torch.cat([q[1] for q in parts])
        return e, a

    for j in range(j0, len(args.tasks)):
        task = args.tasks[j]
        t0 = time.time()
        rec = {"j": j, "task": task, "arm": args.arm}
        sup_e = torch.cat([bank.demo(task, d)[0] for d in range(args.K)])
        sup_a = torch.cat([bank.demo(task, d)[1] for d in range(args.K)])
        flow_T = Flow(base if args.teacher_base else student)
        if memory is not None:
            state = write_demos(memory, flow_T, gbank, task, range(args.K), dev)
            recorder, recorder_real = [], ([] if args.generic else None)
            runner = make_runner(shape_meta, "libero_90", args.n_collect, args.par, args.collect_offset, dev, args.horizon)
            teacher_pol = base if args.teacher_base else student
            col = run_task(runner, install_sampler(teacher_pol, flow_T, memory, state, recorder, temb, recorder_real), task)
            # teacher inputs (generic or real instruction) and student inputs (always real), row-aligned
            sup_g = torch.cat([gbank.demo(task, d)[0] for d in range(args.K)])
            ctx_t = torch.cat([torch.cat(recorder).float().to(dev), sup_g])
            ctx_s = torch.cat([torch.cat(recorder_real).float().to(dev), sup_e]) if args.generic else ctx_t
            rec["teacher"] = col
            rec["n_contexts"] = int(ctx_t.shape[0])

            def new_batch():
                i = torch.randint(len(ctx_t), (args.bs,), device=dev)
                with torch.no_grad():
                    return ctx_s[i], memory.sample(flow_T, state, ctx_t[i])

            with torch.no_grad():  # replay store: one teacher label per stored context
                i = torch.randperm(len(ctx_t), device=dev)[: args.store]
                store = (ctx_s[i].half(), torch.cat([memory.sample(flow_T, state, c) for c in ctx_t[i].split(512)]).half())
        else:
            new_batch = demo_batcher(sup_e, sup_a, args.bs)
            store = (sup_e.half(), sup_a.half())

        t1 = time.time()
        student, hist = train_decoder(student, new_batch, args.steps, lr=args.lr, replay_batch=replay_batch,
                                      tag=f"[{args.arm} consolidate task {task}]")
        rec["distill_hist"], rec["distill_sec"] = hist, round(time.time() - t1, 1)
        buffers.append(store)

        runner = make_runner(shape_meta, "libero_90", args.n_eval, args.par, 0, dev, args.horizon)
        rec["student"] = run_task(runner, install_sampler(student, Flow(student)), task)
        rec["sec"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"[{args.arm}] task {task}: teacher {rec.get('teacher', {}).get('rate', float('nan')):.2f} "
              f"-> memory-off student {rec['student']['rate']:.2f}  ({rec['sec']}s)", flush=True)
        torch.save({"decoder": decoder_state(student), "buffers": [(e.cpu(), a.cpu()) for e, a in buffers],
                    "results": results, "j": j + 1}, prog + ".tmp")
        os.replace(prog + ".tmp", prog)

    final_path = os.path.join(out, "final.json")
    if not os.path.exists(final_path):
        final = {"arm": args.arm, "sequence": results, "novel": [], "retention": [], "libero_10": []}
        install_sampler(student, Flow(student))
        r90 = make_runner(shape_meta, "libero_90", args.n_eval, args.par, 0, dev, args.horizon)
        final["novel"] = [run_task(r90, student, t) for t in args.tasks]
        r90 = make_runner(shape_meta, "libero_90", args.n_retention, args.par, 0, dev, args.horizon)
        final["retention"] = [run_task(r90, student, t) for t in args.retention_tasks]
        r10 = make_runner(shape_meta, "libero_10", args.n_l10, args.par, 0, dev, args.horizon)
        final["libero_10"] = [run_task(r10, student, t) for t in args.l10_tasks]
        with open(final_path, "w") as f:
            json.dump(final, f)
        print(f"[{args.arm}] final: novel {sum(r['rate'] for r in final['novel']) / len(final['novel']):.3f} "
              f"retention {sum(r['rate'] for r in final['retention']) / len(final['retention']):.3f} "
              f"libero10 {sum(r['rate'] for r in final['libero_10']) / len(final['libero_10']):.3f}", flush=True)


if __name__ == "__main__":
    main()
