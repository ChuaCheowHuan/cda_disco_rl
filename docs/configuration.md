# Configuration

A run sets the package's settings in its gym-CDA train config, under `extension_settings`:

```json
"extensions_enabled": ["discorl"],
"extension_settings": {"discorl": {"modules": ["policy_0"], "target_policy": true}}
```

Any key you leave out takes its default, and gym-CDA rejects a key that is not listed below. The
source of truth is `DEFAULTS` in [`cda_discorl/config.py`](../cda_discorl/config.py). Settings
for an extension that is not enabled are ignored, so a config can keep this block while the
package is uninstalled.

With gym-CDA's command-line tools, pass the same values with `--set`:

```bash
--set 'extensions_enabled=["discorl"]' \
--set 'extension_settings={"discorl":{"modules":["policy_0"],"target_policy":true}}'
```

## 1. Every setting

| Key | Default | Range | Meaning |
|---|---|---|---|
| `modules` | `["policy_0"]` | Non-empty list of trainable module ids | Modules the components apply to; the others stay plain PPO |
| `target_policy` | `false` | bool | KL from the policy to an EMA copy of itself |
| `target_coeff` | `0.9` | [0, 1) | Share of the copy kept at each step; it moves `1 - target_coeff` toward the online weights |
| `target_kl_cost` | `0.1` | ≥ 0 | Weight of the KL term |
| `adv_ema` | `false` | bool | Normalise advantages by a lifetime EMA of their moments |
| `adv_ema_decay` | `0.99` | [0, 1) | Decay of that EMA |
| `value_norm` | `false` | bool | The critic learns returns normalised by an EMA of the targets' moments |
| `value_norm_decay` | `0.99` | [0, 1) | Decay of that EMA |
| `env_variety` | `false` | bool | Redraw some market rules at the start of every episode |
| `env_variety_keys` | `["matching_rule", "step_clearing", "maker_fee_bps", "taker_fee_bps"]` | Subset of those four | Which settings to redraw |
| `env_variety_choices` | See §3 | A non-empty list per key | The values each key is drawn from |
| `policy_prediction` | `false` | bool | Auxiliary head predicting the next step's policy (experimental) |
| `policy_prediction_cost` | `1.0` | ≥ 0 | Weight of the prediction loss |

## 2. Rules checked when the config is built

`config.validate` raises a `ValueError` that names the key when:

- `modules` names a module that is not trainable (`policy_0` to `policy_{num_trained_agents-1}`).
- A switch is not `true` or `false`.
- `target_policy`, `adv_ema` or `value_norm` is on with `num_learners > 1`, because their state
  lives on one Learner.
- A decay or `target_coeff` is outside [0, 1), or a cost is negative.
- `policy_prediction` is on with `encoder_type: lstm`, because it needs a stateless encoder.
- `env_variety_keys` names a key the env config lacks, or one outside the four supported keys.
- An `env_variety_choices` list is empty, holds a matching rule or clearing mode the env does not
  know, or can draw a maker and taker fee pair the env's fee rules reject.

On a resume, gym-CDA also refuses any setting that differs from the checkpoint's. Changing
`value_norm` or `policy_prediction` is refused as structural, because each adds tensors to the
module.

## 3. Env variety choices

| Key | Default choices | Allowed values |
|---|---|---|
| `matching_rule` | `["fifo", "pro_rata"]` | The order book's matching rules |
| `step_clearing` | `["sequential", "batch"]` | The env's clearing modes |
| `maker_fee_bps` | `[0, 1]` | Any fee the env's fee rules accept with every taker choice |
| `taker_fee_bps` | `[0, 2]` | The same, with every maker choice |

## 4. Ready-made settings blocks

**The recommended set.** These components did not hurt the critic in the comparison
([findings.md](findings.md)):

```json
{"discorl": {"modules": ["policy_0"], "target_policy": true, "adv_ema": true}}
```

**Value normalisation alone.** This one is new and is still being measured:

```json
{"discorl": {"modules": ["policy_0"], "value_norm": true}}
```

**Training for generalisation.** Judge this run on held-out markets, not on training return,
because fees change returns:

```json
{"discorl": {"modules": ["policy_0"], "target_policy": true, "env_variety": true}}
```

**An in-run comparison.** With two trained agents, `policy_0` gets the components and `policy_1`
stays plain PPO in the same league:

```json
{"discorl": {"modules": ["policy_0"], "target_policy": true}}
```

To apply the components to both trained modules, list both: `"modules": ["policy_0", "policy_1"]`.
