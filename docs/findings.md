# Findings

How each DiscoRL mechanism compared against plain PPO in gym-CDA's market, and why two of them
are not in this package.

## 1. Setup

Every arm was run with gym-CDA's `train.compare`, three seeds each (0, 1, 2), on 2026-10-19 and
2026-10-20:

- 4 agents, 2 of them trained, with the `mlp` encoder
- 512-step episodes, 4 episodes per iteration
- 30 iterations, which with the run defaults is about 120 optimiser steps
- each component switched on for both trained modules, `"modules": ["policy_0", "policy_1"]`

Numbers are the final iteration's mean ± standard deviation across seeds. "Return" is
`module_episode_returns_mean`, averaged over the two trained modules.

## 2. Results

| Arm | Return | Critic explained variance | Pass fraction |
|---|---|---|---|
| Plain PPO | 0.217 ± 0.206 | 0.815 ± 0.106 | 0.107 ± 0.004 |
| Target policy | 0.309 ± 0.235 | 0.874 ± 0.011 | 0.128 ± 0.018 |
| Advantage EMA | 0.192 ± 0.218 | 0.886 ± 0.007 | 0.104 ± 0.009 |
| Env variety | 0.226 ± 0.162 | 0.892 ± 0.048 | 0.125 ± 0.044 |
| Policy prediction | 0.029 ± 0.119 | 0.863 ± 0.124 | 0.132 ± 0.015 |
| Value normalisation | 0.117 ± 0.088 | 0.936 ± 0.032 | 0.126 ± 0.021 |
| Categorical value (removed) | -0.028 ± 0.038 | 0.248 ± 0.150 | 0.086 ± 0.031 |
| Disco103 | -0.006 ± 0.300 | – | 0.080 ± 0.024 |

None of the components separated from PPO on return: the spread between seeds (about ±0.2) is
larger than any gap between arms. At this budget the comparison is a check that each component
trains without harm, not evidence that one helps.

What it does show:

- **Target policy** had the highest mean return, and target policy, advantage EMA and env variety
  all kept the critic at least as good as PPO's. They are the recommended set.
- **Policy prediction** had a lower mean return, still within the spread. It stays in the
  package, marked experimental.
- **Env variety** draws fees per episode, so its returns come from a different market from the
  other arms' and are not directly comparable.
- **Value normalisation**, run later on the same settings, gave the best critic of any arm
  (explained variance 0.94 against PPO's 0.82, and steadier across seeds). Its mean return is
  lower than PPO's, but within the spread between seeds. At the end of seed 0 its running
  estimate of the value targets was a mean of 0.11 and a standard deviation of 0.07. So the
  targets grow as the agents learn, from the 0.004 standard deviation measured early in a run.

## 3. A longer run: target policy against PPO

Target policy, the best arm above, was rerun against PPO for 100 iterations and five seeds (0 to
4), with the other settings unchanged. Return here is averaged over each seed's last 20
iterations, which is steadier than the final iteration alone.

| Arm | Return, last 20 iterations | Per seed (0 to 4) | Critic explained variance |
|---|---|---|---|
| Plain PPO | 0.081 ± 0.060 | 0.030, 0.014, 0.103, 0.097, 0.161 | 0.90 |
| Target policy | 0.178 ± 0.223 | 0.236, 0.046, 0.047, 0.545, 0.014 | 0.88 |

Target policy's mean is about twice PPO's, but the difference is not significant. Paired by
seed, it is ahead on three of five seeds (mean difference 0.097, t = 0.92 with 4 degrees of
freedom), and most of its lead comes from seed 3. It also varies more between seeds. Both
critics are equally good.

So target policy stays recommended because it does no harm and may help. Showing that it helps
would take more seeds or a longer budget.

## 4. What was removed, and why

### The categorical critic

DiscoRL's critic predicts a distribution over 601 bins spanning ±300 in a transformed space. This
market's value targets are tiny: an instrumented run measured their standard deviation at about
0.004. So nearly every target falls between the two middle bins.

Every variant tried left the critic far behind PPO's:

| Variant | Return | Critic explained variance |
|---|---|---|
| 601 bins, ±300 (DiscoRL's default) | -0.028 ± 0.038 | 0.248 ± 0.150 |
| 601 bins, ±5 | -0.037 ± 0.162 | 0.074 ± 0.051 |
| 601 bins, ±0.5 | -0.103 ± 0.096 | -0.029 ± 0.032 |
| 601 bins, ±0.02 | 0.087 ± 0.440 | 0.029 ± 0.038 |
| 51 bins, ±5 | -0.025 ± 0.032 | -0.252 ± 0.506 |
| Default, 8× more updates | -0.142 ± 0.017 | -0.039 ± 0.108 |
| Default, zero-initialised head | 0.030 ± 0.213 | -0.006 ± 0.048 |
| Plain PPO, 8× more updates | 0.248 ± 0.280 | 0.615 ± 0.010 |

The width of the support, the number of bins and the number of updates were each ruled out. The
critic's prediction barely varied from state to state, even while its cross-entropy fell. On a
toy regression with larger targets, the same head fit as well as a squared-error head, so the
failure is specific to this setting. The root cause was not found.

`value_norm` is this package's answer to the same problem. It keeps PPO's squared-error critic
and removes its dependence on the reward's scale.

### Disco103

Disco103 is DeepMind's frozen, meta-learned update rule, which replaces PPO's loss entirely. In
this comparison it learned no better than PPO and cost far more. Each module had 9.3M parameters
against 0.26M, each run took about 4× as long, and each held about 4 GB of memory. It was
meta-trained on agent lifetimes in the millions of steps, against about 120 optimiser steps here.

It is back in 0.2.0 as the experimental `disco103` component, which imports the rule from `disco_rl`.
These numbers come from the earlier version, which ran the same rule through the same kind of
bridge; the new one has not been compared yet.

## 5. Reproducing a run

With gym-CDA and this package installed, one arm and one seed is:

```bash
OMP_NUM_THREADS=1 python -m gym_continuousDoubleAuction.train.compare \
  --encoders mlp --seeds 0 --iters 30 --out out/target_policy/seed0 \
  --agents 4 --trained-agents 2 --max-step 512 --episodes-per-iter 4 --no-probe \
  --set 'extensions_enabled=["discorl"]' \
  --set 'extension_settings={"discorl":{"modules":["policy_0","policy_1"],"target_policy":true}}'
```

Leave out both `--set` lines for the plain PPO arm, and pass `--iters 100` for the longer run. A plain PPO run took about 7 minutes on a CPU.
With `OMP_NUM_THREADS=1`, three run side by side without slowing each other.
`gym_continuousDoubleAuction.train.compare.aggregate` and `render` turn the
`compare_results.json` files into the table above.
