"""Prior-skills -> self-generated experience -> adaptation on unseen YCB objects (ManiSkill PoC).

Modules
  common   run records, atomic checkpoints, hashing
  split    predefined, criteria-based object split (source / dev / test)
  env      ManiSkill wrapper: fixed object list, filtered state obs, terminal reward protocol
  sac      actor, LayerNorm critic ensemble, replay, SAC / RLPD-style updates
  train_source   PickCube check and source-skill training (dense), terminal-reward source replay, critic refit
  adapt_run      A0 / A1 / A2 cumulative adaptation with budget snapshots and read-only evaluation
  analyze        development gate
"""
