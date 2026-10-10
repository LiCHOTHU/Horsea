| scope | arm | conditions with a success in 9 attempts | first-success attempt per condition (- = none, censored) | restricted mean first success (censored = 10) | fresh-start success | init failures | object loss (eval / adapt) | constraint violations | adaptation control steps |
|---|---|---|---|---|---|---|---|---|---|
| grasp | A | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 3240 |
| grasp | B | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 3240 |
| grasp | C | 6/6 | 1 1 1 1 1 1 | 1.00 | 14/18 (78 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 3240 |
| transport | A | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 4590 |
| transport | B | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 | 0 % / 2 % | 0 % (signal unavailable) | 4732 |
| transport | C | 6/6 | 1 1 1 1 1 1 | 1.00 | 18/18 (100 %) | 0 | 0 % / 2 % | 0 % (signal unavailable) | 4692 |
| door | A | 5/6 | 1 1 - 2 1 4 | 3.17 | 10/18 (56 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 5080 |
| door | B | 3/6 | 1 1 - - 1 - | 5.50 | 9/18 (50 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 5499 |
| door | C | 3/6 | 1 1 - - 1 - | 5.50 | 9/18 (50 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 5506 |
| overall | A | 17/18 | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 - 2 1 4 | 1.72 | 42/54 (78 %) | 0 | 0 % / 0 % | 0 % (signal unavailable) | 12910 |
| overall | B | 15/18 | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 - - 1 - | 2.50 | 41/54 (76 %) | 0 | 0 % / 1 % | 0 % (signal unavailable) | 13471 |
| overall | C | 15/18 | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 - - 1 - | 2.50 | 41/54 (76 %) | 0 | 0 % / 1 % | 0 % (signal unavailable) | 13438 |

| contrast | fresh-start success difference (pp) | 95 % CI (paired, operation-stratified) | per operation (pp) | attempts saved to first success (censored at 10) | 95 % CI |
|---|---|---|---|---|---|
| H1_B_minus_A | -1.9 | [-5.6, +0.0] | grasp +0, transport +0, door -6 | -0.8 | [-1.7, +0.0] |
| H2_C_minus_B | +0.0 | [+0.0, +0.0] | grasp +0, transport +0, door +0 | +0.0 | [+0.0, +0.0] |
| C_minus_A | -1.9 | [-5.6, +0.0] | grasp +0, transport +0, door -6 | -0.8 | [-1.7, +0.0] |
