# Regression of the task-generic pilot code against the first pilot (Stack-Can-Into-Drawer, attempt 10)

Smoke-only run of `s1local.runner_task --spec drawer_regression` (same task, program, operations, bounds, conditions and
starts as `experiments/s1local_2026-10-10`), 2026-10-10 19:17-19:23.

- All 18 smoke checks (reference, repeat, lower / upper bounds, heaviest-slipperiest condition, fresh start e1; three
  operations) give the same success outcome S and the same control steps as the original; scores J agree within the
  restore noise.
- Expected difference: transport at the lower bounds drops the can onto the table. The generic loss criterion D counts
  only a fall off the table (z < 0.95 m; needed for tasks whose target is the table, e.g. Unload-Cans), while the first
  pilot also counted a drop out of the cabinet as a loss; J -0.99 there vs +0.005 here.
- One of twelve starts was invalid this time (transport/e2: the rebuilt reference grasp missed the can shifted 8 mm
  at 120 deg), the knife-edge start reported in the first pilot; invalid starts are recorded as initialization failures.
