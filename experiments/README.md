# Reproducible evaluations

## Prepared productive-BOMB selection-free final confirmation

s169000 fixes the successful s168 treatment without changing its mask or checkpoint, then compares
it with unchanged `control-r2-round0150` on eight fresh paired 25-round rule/mixed blocks. Each arm
runs 200 rounds (400 environment rounds total) with zero optimizer updates. Passing requires at
least +0.10 score/round, 5/8 block wins, nonnegative score deltas in both strata, no suicide or
invalid-action increase, kill preservation, and latency safety. Every outcome stops without
training, copying, packaging, or deployment.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_productive_bomb_final.py
```

Only Luna may add `--execute`, following `CNN_N8_PRODUCTIVE_BOMB_FINAL_LUNA_PROMPT.md`.

## Completed control-r2 productive-BOMB paired A/B

s168000 passed every preregistered gate. Treatment/control score was `3.36/3.11`, kills
`0.20/0.15`, suicides `0.21/0.30`, and invalid actions `1.20/1.61` per round. Rule and mixed score
deltas were `+0.36/+0.14`, and treatment won 3/4 cases. The frozen treatment is only the unique
s169 confirmation candidate; it was not trained, copied, packaged, or deployed.

## Historical preparation record for s168000

s168000 freezes `control-r2-round0150` and changes one inference variable only: treatment removes
a currently legal BOMB when its official blast contains neither a crate nor an opponent, falling
back to the control mask if removal would empty the action set. Four fresh paired 25-round cases
cover rule and mixed opponents, for 100 rounds per arm and zero optimizer updates. The implementation
uses only project-native environment semantics; no external source, weights, labels, actions, or
trajectories enter the experiment. Every outcome stops without training, copying, or deployment.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_productive_bomb_ab.py
```

Only Luna may add `--execute`, following `CNN_N8_PRODUCTIVE_BOMB_AB_LUNA_PROMPT.md`.

## Prepared top-3 internal agents vs pinned external strong agent

s161000 fixes the three identities from the latest directly comparable fresh-seed Task4 evaluation:
collision-filtered CNN (`3.3700`), original CNN (`3.3600`), and frozen-v4 (`2.8825`). They share
100 four-player games with the pinned external `feature_is_everything` policy. Four cyclic
25-round rosters place every identity in every command-line slot exactly once. This is a fast
ranking diagnostic, not an automatic selection gate; it cannot train, copy, or deploy a model.

Validate without downloading the external repository or starting formal games:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_top3_external_match.py
```

Only Luna may add `--execute`, following `CNN_N8_TOP3_EXTERNAL_MATCH_LUNA_PROMPT.md`.

## Completed collision-mask selection-free final confirmation

s160000 fixed the s159 treatment without modification and compared it against the original frozen
CNN and frozen-v4 on eight fresh matched 50-round Task4-R/M blocks. Each identity ran 400 rounds;
no training, checkpoint copy, deployment, or result-dependent follow-up occurred.

Candidate/original/v4 pooled score was `3.3700/3.3600/2.8825`. The candidate beat original in 5/8
blocks but gained only `+0.0100` score/round, below the pre-registered `+0.10` confirmation gate.
It beat v4 by `+0.4875` overall and in 6/8 blocks, but lost the Task4-R stratum by `-0.065` and its
suicide rate exceeded v4 by `+0.1475`, failing two more fixed gates. Relative to original CNN, the
mask reduced invalid actions from `1.265` to `0.890` per round while score, kills, and suicides were
essentially unchanged. The candidate is therefore not confirmed and was not deployed. Report
SHA-256 is `c71ee2a0af201f89d87c0bd25c98b017979b8d612ce460493b1b96705adbb36b`;
24/24 formal runs completed, both checkpoint hashes remained unchanged, and 402 tests passed.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_collision_mask_final.py
```

The command now validates completed artifacts only. Do not add `--execute` to rerun the immutable
protocol.

## Completed frozen-CNN immediate-collision mask utility A/B

s159000 converted the s158 signal into one fixed inference intervention. Treatment intersects the
original legal actions with next-step opponent-collision exclusion plus ordinary known-hazard
survival only while an own bomb is active, and falls back to the original mask when the intersection
is empty. It never opens an originally illegal action and never updates the checkpoint.

Across eight fresh matched 25-round Task4-R/M blocks per arm, treatment/control score was
`3.365/3.205` (`+0.160` per round), with treatment winning 6/8 blocks. Suicide fell from `0.250` to
`0.215`, kills rose from `0.135` to `0.145`, invalid actions fell from `1.085` to `1.030`, and both
strata improved. All pre-registered gates passed. Only 77/65,715 treatment decisions changed, all
from MOVE to WAIT, while overall WAIT increased just 0.151 percentage points. The candidate is
frozen but not deployed; report SHA-256 is
`6408d80868a8930d417c92a9a86104a7e3f0c6ab1b1225a6f4c087015d405c53` and 401 tests passed.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning tools/cnn_n8_collision_mask_ab.py
```

The command is completed-artifact validation only. Do not add `--execute`; any final confirmation
must use a new protocol and fresh seeds without modifying the mask.

## Completed frozen-CNN immediate-collision audit

s158000 narrowed s157000's adversarial reachability to the next destination only, then used the
existing ordinary known-explosion survival check. Across 200 fresh Task4-R/M rounds it captured
18/50 self-kills (`36%`) while flagging `1.85%` of safe placements, 1/28 kill placements (`3.57%`),
and `2.12%` overall. Seventeen of the 18 captured deviations produced an actual `INVALID_ACTION`,
so the label is precise for avoidable simultaneous collisions, but its coverage is far below the
pre-registered `70%` gate. Per the terminal rule, do not continue safety-mask rule search and do not
activate or A/B this rule. Report SHA-256 is
`1c8d5a485f505f71fdb2a50434c12b6e8e1abafdda3253e81d0ad174bd7298fb`; the checkpoint remained
unchanged and 400 tests passed.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_postbomb_movement_audit.py \
  --protocol experiments/configs/model-a-cnn-n8-postbomb-immediate-collision-audit-s158000.json
```

This is now completed-artifact validation only; do not add `--execute` to rerun the immutable
protocol.

## Completed frozen-CNN post-bomb movement counterfactual audit

s157000 passively audited 25,900 decisions made while an own bomb was active across 200 fresh
Task4-R/M rounds. A placement was flagged when the frozen policy selected outside a non-empty set of
MOVE/WAIT actions that retained a full-horizon route through known explosions and adversarial
opponent reachability. This captured 36/50 self-kills (`72%`) while flagging only `7.72%` of safe
placements and `8.22%` overall, but it flagged 7/21 kill-producing placements (`33.33%`) and failed
the pre-registered `25%` kill-preservation guard. The rule must not be activated or advanced to A/B.
The report SHA-256 is
`006746061a356346c033c1321d8c5fd85c4f8546550c6904d29bd9c8edd980fc`; the checkpoint remained
unchanged and 399 tests passed.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_postbomb_movement_audit.py
```

The command above validates the immutable completed protocol without executing it. The diagnostic
next step, if authorized, is a new fresh-seed audit that excludes only immediate opponent-reachable
destinations while using ordinary known-hazard survival thereafter; it must not post-hoc relax this
run's gate.

## Completed frozen-CNN robust-BOMB counterfactual audit

s156000 passively labelled all 6,688 actual bomb placements in 200 fresh Task4-R/M rounds. The
fixed rule required a movement-first full-horizon escape path that avoided every tile reachable by
each current opponent at the corresponding future action offset. It vetoed only 138 resolved bombs
and preserved safe/kill bombs well, but captured just 9/43 self-kill placements (`20.93%`) versus the
pre-registered `70%` requirement. The signal is therefore rejected: do not activate this BOMB veto
or start a paired A/B. The frozen checkpoint was neither trained nor overridden and retained SHA-256
`7c35d969f5d798c6ab3135f8bb919e19c99767bad2db4e372146a07024059332`. Report SHA-256 is
`12bc16a3d78e472a1661f119db68667289d2e3b0af77a6983042a52e06a97769`.

The earlier s155000 artifact is invalid because its first version incorrectly counted WAIT as a
post-bomb first move; its directory contains an explicit erratum. Only the corrected s156000 run on
fresh seeds is decision-bearing.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_robust_bomb_audit.py
```

The command above now performs completed-artifact validation only. Do not add `--execute` to rerun
the immutable protocol.

## Completed frozen-CNN self-kill audit

s154000 passively traced the final eight decisions of every official self-kill from 200 fresh
Task4-R/M rounds. The frozen `task4-r3-round0800` checkpoint was never trained or overridden. All
53 self-kills followed a recent own bomb; 52 became statically unavoidable one to four steps after
placement even though the bomb mask originally found one to four safe first moves. At the first
unavoidable state, 43/52 had an adjacent opponent; 35/53 traces contained a simultaneous-movement
`INVALID_ACTION`, and eight acquired a new opponent bomb. Direct dynamic-interference evidence
therefore covers 45/53 cases. Report SHA-256 is
`6fb2727b0c9bcaa09e7a0eca9b18573f136b71990ca5b44ccb412bcf75bcde2e`.

```bash
/opt/anaconda3/envs/prm_env/bin/python tools/cnn_n8_selfkill_audit.py
```

The command above is a completed-artifact dry-run only. Do not add `--execute` unless intentionally
creating a new protocol, because s154000 outputs are immutable.

## Completed CNN+n=8 safety replay fine-tune A/B

s152000 confirmed the CNN candidate's small Task4 score edge (`3.1575` versus
frozen-v4 `2.9400`) but rejected it because suicides and invalid actions were
too high. s153000 changed replay sampling only: paired control batches were 256
uniform transitions, while treatment batches reserve 8 kill-chain and 8
self-kill-chain slots within the same batch size. Rewards, network, action
mask, n=8, optimizer state, opponents and paired seeds remain unchanged.

```bash
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_safety_replay_ab.py
```

Both milestones failed all four gates: balanced replay reduced suicide only slightly while losing
`0.68--1.03` score/round and `0.053--0.067` kills/round. Retention and confirmation did not run,
and no checkpoint was deployed.

## Completed CNN+n=8 selection-free final confirmation

s151000 selected `task4-r3-round0800` as the unique course champion. s152000
compared that immutable checkpoint with frozen-v4 and source-r2 on six fresh
200-round strata. A pinned external strong agent is diagnostic-only on the two
Task4 strata plus a 200-round seat-balanced direct match; it cannot affect the
replacement gate, training, selection or submission.

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/cnn_n8_final_confirmation.py
```

The candidate won 7/8 Task4 blocks and gained `0.2175` score/round over v4,
but failed the frozen suicide and invalid guards, so v4 remained the default.

## Current compact full-board CNN+n=8 unified Task4

Task3-C selected round 150 with pooled score/kills 3.433/0.133 versus
frozen-v4's 3.35/0.08. s151000 continues all three lineages through one
unified 1,600-round three-rule course. It selects a shared milestone on both
three-rule and mixed rosters, ranks all 15 course champions, measures four
retention layers, and stops before selection-free confirmation.

```bash
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
/opt/anaconda3/envs/prm_env/bin/python tools/cnn_n8_task4.py
```

Only Luna may add `--execute`, following `CNN_N8_TASK4_LUNA_PROMPT.md`.

## Completed compact full-board CNN+n=8 Task3-C

Task3-C selected round 150 from pooled milestone scores
3.433/3.193/3.327/3.370. It modestly beat frozen v4 on score and clearly on
kills and WAIT, while exposing higher suicide and old-task forgetting. Its
report SHA-256 is
`bdbb821b268f6f5a3843e15eec949d4462319243f9e7c65731803cea6d999aa9`.

## Completed compact full-board CNN+n=8 Task3-P

The three Task3-P round-600 checkpoints scored 5.69/5.68/3.50 against the
peaceful opponent, with 0.46/0.51/0.17 kills per round. The pooled score 4.9567
comfortably beat frozen v4 and nearly matched source-r2 while doubling its kill
rate. The report SHA-256 is
`b55b18fc6f546575ed068c346a0f8f756bc8a0094270eb855e5833b6b671571b`.

## Completed compact full-board CNN+n=8 Task2

Task1 selected round 400 as the only pooled near-best milestone.  Its three
replicas scored 47.59/46.73/44.70 on the fixed Task1 validation, versus 32.67
for frozen v4.  s148000 continues each corresponding lineage for 800 classic
rounds without opponents.  It selects one shared Task2 milestone from
200/400/600/800, then measures matched Task1 retention and stops.

Validate without starting formal training:

```bash
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
/opt/anaconda3/envs/prm_env/bin/python tools/cnn_n8_task2.py
```

Only Luna may add `--execute`, following `CNN_N8_TASK2_LUNA_PROMPT.md`.

## Completed compact full-board CNN+n=8 Task1

The new independent line starts a 367,863-parameter agent-centred full-board
CNN from random weights.  It uses uniform all-action n=8 Double DQN, bounded
shaping, a 50k uint8 replay and MPS batch 256.  The three Task1 replicas run
sequentially; their 100/200/300/400 checkpoints are evaluated on the same
validation cases and must share one pooled earliest-near-best milestone.

Validate without starting formal training:

```bash
unset PYTORCH_MPS_HIGH_WATERMARK_RATIO
/opt/anaconda3/envs/prm_env/bin/python tools/cnn_n8_task1.py
```

Only Luna may add `--execute`, following `CNN_N8_TASK1_LUNA_PROMPT.md`.  The
runner stops after the Task1 report and never starts Task2 or deploys a model.

## Historical frozen-v4 global-resource n=8 rescue pilot

s140000 showed that raw full-board resource visibility at one-step reduces WAIT
but does not improve Task2 score: full33/local7/frozen-v4 scored
`1.6000/1.7067/2.1600`. s141000 tests one narrow explanation: both local7 and
full33 now use uniform all-action n=8 returns, while representation, network,
reward, optimizer, epsilon, residual cap, seeds within each matched pair, and
all safety code remain fixed.

Three matched pairs train for 200 rounds per arm. Round 100 is diagnostic only;
round 200 is the sole endpoint. If parameters change, frozen v4 and all six
endpoints receive a fresh 490-round Task1/Task2 evaluation. No branch selects,
copies, deploys, or automatically continues a checkpoint.

Validate without starting a game:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_global_resource_n8_pilot.py
```

Only Luna may add `--execute`, following
`V4_GLOBAL_RESOURCE_N8_PILOT_LUNA_PROMPT.md`.

## Historical frozen-v4 global resource visibility Stage 1

s139000 showed that 92.1% of the external strong agent's 2.9975-point lead over
frozen v4 came from coins, while frozen v4 waited on 35.22% of its observed
steps. s140000 therefore tests resource observation extent rather than attack.

Both trained arms use the same frozen-v4 base, zero-initialized bounded CNN
residual, original reward/mask, one-step uniform Double-DQN and fixed training
budget. `local7` masks the common 33x33 resource tensor outside the central 7x7;
`full33` retains the complete 17x17 observer board. The only two channels are
field/crates and visible coins. Opponents, bombs, attack labels, target/BFS
directions, history and teachers are absent.

Validate without starting a game:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_global_resource_stage1.py
```

Only Luna may add `--execute`, following
`V4_GLOBAL_RESOURCE_STAGE1_LUNA_PROMPT.md`. Every branch stops before Stage 2,
checkpoint selection, copying, or deployment.

## Historical frozen-v4 opponent-shift evaluation

s138000 rejected rehearsal25-r3 on 400 fresh Task4 rounds per identity. Frozen
v4 ranked first by pooled score, kills and safety, but that confirmation still
used only the seeded rule-opponent family.

s139000 changes no model. Frozen v4, source-r2, the rejected candidate-r3 and a
pinned external `feature_is_everything` stress opponent share 400 classic games.
Eight 50-round cases balance every identity twice in every command-line roster
position. The external repository is fetched at its pinned commit into a system
temporary directory, hash-checked, and never used for training, teacher labels,
checkpoint selection or submission. The report describes internal ranking
stability and the external-v4 score gap; every branch stops.

Validate without downloading the external repository or starting a game:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_opponent_shift.py
```

Only Luna should add `--execute`, following `V4_OPPONENT_SHIFT_LUNA_PROMPT.md`.

## Historical rehearsal25-r3 Task4-only confirmation

The completed s137000 pilot did not establish fixed 25% rehearsal as a stable
method: it recovered old-task performance and raised pooled score, but reduced
pooled threat-to-kill conversion and produced zero fully supportive matched
pairs. After all six validation endpoints were revealed, rehearsal25-r3 was
the sole post-hoc Task4 candidate: it tied source-r2 at 2.98 score/round while
raising kills from 0.07 to 0.16 and lowering suicides from 0.42 to 0.27. Its
score margin to the rule opponents was worse, so those discovery data cannot
promote it.

s138000 performed no training. It fixed rehearsal25-r3 before looking at 24
new seed values and evaluates candidate-r3, source-r2 and frozen v4 for 200
rounds each in Task4B and 200 rounds each in Task4C (1,200 rounds total).
Task1--3 are diagnostic history, not a hard deployment gate. Own score and
score margin to rule opponents were primary; kills and suicides were secondary
attack/safety gates. Candidate-r3 failed with only one of eight supportive case
blocks, so frozen v4 remained the incumbent.

Validate without starting formal evaluation:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_rehearsal_r3_task4_confirmation.py
```

Only Luna should add `--execute`, following
`V4_REHEARSAL_R3_TASK4_CONFIRMATION_LUNA_PROMPT.md`.

## Historical fixed 25% source-task rehearsal pilot

The completed s136000 A/B provisionally supported stratified sampling versus
matched uniform replay, but all trained endpoints remained below source-r2 and
catastrophically forgot Task1--3. There are no comparable intermediate s136000
checkpoints other than round 100, and the fixed 100-round pilot must not be
repeated.

s137000 changes one variable: 25% of each 64-item batch is frozen source-r2
replay, split equally over Task1, Task2, Task3-peaceful and Task3-coin. Both arms
retain the same n=8 targets, two Task4 clean-kill slots, two Task4 self-kill
slots, model, reward, optimizer, epsilon, Task4C opponents and 100-round fixed
endpoint. Frozen-source rehearsal datasets use training-only seeds; validation
and conditional confirmation use two separate seed sets.

Validate without starting collection or training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_rehearsal_pilot.py
```

Only Luna should add `--execute`, following `V4_REHEARSAL_PILOT_LUNA_PROMPT.md`.
Every terminal branch stops before promotion or another course.

## Historical source-r2 n=8 replay sampling-only A/B

The completed s135000 audit found that n=8 is the smallest tested all-action
horizon covering every uniquely linked BOMB-to-kill outcome, while n=16/32 add
no coverage. Even at n=8, kill-bearing targets are only about 0.286% of targets,
so a uniform batch of 64 has about an 83.26% chance of containing no kill target.

s136000 therefore holds n=8, reward/shaping, model, features, mask, optimizer,
epsilon and training distribution fixed. Three matched pairs compare 64 uniform
samples against 60 uniform + 2 clean kill-chain + 2 self-kill-chain samples.
Trade kills are negative examples only; shortages fall back to unique uniform
samples and are audited. All endpoints are fixed at round 100.

Validate without starting formal training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_sampling_distribution_ab.py
```

Only Luna should add `--execute`, following
`V4_SAMPLING_DISTRIBUTION_AB_LUNA_PROMPT.md`. Evaluation is skipped unless at
least two matched pairs show a tensor-level parameter effect. Either terminal
decision stops before checkpoint promotion or another course.

## Historical source-r2 training-system audit recovery (Phase A/B only)

The original s134000 collector failed after its first MAX_STEPS survivor round:
the environment had already delivered the final per-step callback before adding
`SURVIVED_ROUND` in `end_of_round`. The failed manifest is preserved and s134000
must not be rerun.

The corrected s135000 audit freezes source-r2 and collects 200 rounds split evenly across
Task4B duel and Task4C three-rule. It records the complete official-event reward
ledger, danger shaping, Q values, and a passive seven-stage attack funnel. It
then reconstructs n=1/4/8/16/32 Double-DQN targets offline on those exact same
trajectories. No optimizer is created, no action is overridden, no policy is
updated, and no horizon is selected for training.

Validate without starting formal collection:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_training_system_audit.py
```

Only Luna should add `--execute`, following
`V4_TRAINING_SYSTEM_AUDIT_RECOVERY_LUNA_PROMPT.md`. The terminal decision always stops
before training.

## Historical Task4 safe-kill reward-500 stress gate

The completed s133000 gate changed only safe owned-BOMB kill reward from 12 to
500. Two of three endpoints diverged, so evaluation ran, but pooled Task4C kills
fell from 0.1467 to 0.1133 per round while bombs rose from 18.29 to 20.67.
Task1/Task2 retention also failed. All endpoints were rejected; the historical
runner must not be rerun.

## Historical Task4 six-step BOMB reward-scale signal gate

The completed s132000 experiment kept six-step BOMB targets fixed and changed
only `KILLED_OPPONENT` from 12 to 80. Pooled Task4C score, kills and suicides
improved slightly, but kill gain and Task1/Task2 retention missed the frozen
gates. Two of three matched pairs were tensor-identical at every snapshot, so
all endpoints were rejected and the historical runner must not be rerun.

## Historical Task4 BOMB temporal-credit matched A/B

The five-transition s130000 implementation is a preserved failed experiment and
must not be rerun.  Its first control replica observed BOMB at step 58, explosion
and kill at step 62, then a lingering-flame self-kill at step 63.  The corrected
s131000 lifecycle audit therefore fixes the complete outcome lags to 4 and 5 and
the complete BOMB return horizon to six transitions.

Both s131000 arms use an identical six-transition maturation queue.  The control
writes a one-step BOMB target; the credit arm writes the six-step target containing
both `gamma^4 * r_(s+4)` and `gamma^5 * r_(s+5)`.  All non-BOMB targets, rewards,
model structure, replay sampling, optimizer state, epsilon and Task4C training
distribution remain fixed.

Validate the frozen protocol without starting a formal game:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_task4_bomb_credit6.py
```

Only Luna should add `--execute`, following
`V4_TASK4_BOMB_CREDIT6_LUNA_PROMPT.md`.  The fixed round-100 experiment trains
three matched control/credit pairs (600 rounds), evaluates v4, source-r2 and all
six endpoints on Task1--3 and Task4C (1,600 rounds), then writes a terminal
report.  Either decision stops before Task4B or distillation.

## Historical Task4 counterfactual causal-signal gate

The completed s129000 diagnostic froze exact-v4 source-r2 and used the
historical v7b counterfactual rollout only as a passive Oracle. It collected
eight 50-round cases split between Task4B duel and Task4C three-rule without
overriding an action, updating a policy, or creating a checkpoint. Its terminal
result rejected Oracle-filtered replay and motivated the fixed-lag audit above.

## Historical Task4 kill-head learnability audit

The current terminal diagnostic freezes source-r2 and collects six grouped
Task4C trajectory cases. It fits a fixed action-conditioned future-kill probe
and a matched-capacity state-only control under three held-out folds. It never
updates a policy or creates a policy checkpoint. Validate without collecting a
game:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v4_task4_kill_probe_audit.py
```

Only Luna should add `--execute`, following
`V4_TASK4_KILL_PROBE_AUDIT_LUNA_PROMPT.md`. Either decision writes a terminal
report and stops before any residual-head policy experiment.

Run evaluations through `experiments/evaluate.py` so every result records its
seed, opponents, scenario, command, Git commit and weight SHA-256. For example:

```bash
python experiments/evaluate.py \
  --agents model_a_dqn rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --rounds 100 --seed 4004 \
  --agent-seed 14004 \
  --weight-path agent_code/model_a_dqn/model_a.pt \
  --variant model-a-safe-mask-v1
```

For strict paired tests against random play, use three `seeded_random_agent`
opponents and pass `--opponent-seed`. The stock `random_agent` reseeds from
system entropy in every child process and therefore cannot provide matched
opponent action sequences across arms.

`--weight-path` is passed to the evaluated agent as an absolute path and its
SHA-256 is checked before and after evaluation. `--agent-seed` controls the
evaluated agent's private tie-breaking RNG; it defaults to `--seed`. The
official opponent agents initialize separate random states, so a world seed
alone does not make their behavior deterministic.

`model_a_v6` additionally requires `--agent-config`. Both the checkpoint and
config hashes are recorded and rechecked after evaluation. See
`V6_EXPERIMENT_PLAN.md` for the staged v6 protocol.

Each invocation writes two files under `experiments/logs/evaluations/`:

- `<run-id>.stats.json`: the unmodified statistics emitted by the environment;
- `<run-id>.json`: metadata and normalized per-agent metrics.

Generate a CSV summary and score plot after several seeds have completed:

```bash
python experiments/plots.py
```

Do not edit generated manifests or raw statistics. Commit them together with
the exact code used for the evaluation. A dirty-worktree flag is recorded to
make results from uncommitted code visible rather than silently treating them
as reproducible.

## Isolated Model A training entry point

Validate a training configuration without launching training:

```bash
python experiments/train_model_a.py experiments/configs/model-a-dry-run-example.json
```

Training occurs only when `--execute` is added explicitly. The runner refuses
to overwrite `agent_code/model_a_dqn/model_a.pt`, refuses resume, requires a
new run ID and writes the checkpoint and manifest to isolated experiment
paths. Review the printed dry-run command before using `--execute`.

## Current Task-3 resolved-duel distribution experiment

The first training-distribution signal gate increased kills from 18/200 to
77/200 but exposed a termination mismatch: a survivor continued to 400 steps
after eliminating the only opponent while crates remained. The corrected v2
changes only the training-only kill-rich termination rule, ending the episode on
first elimination. Validate without running any game or training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/task3_training_distribution_v2.py \
  experiments/configs/task3-training-distribution-v2-matched-s113000.json
```

Only Luna should add `--execute`, following
`TASK3_TRAINING_DISTRIBUTION_V2_LUNA_PROMPT.md`. The corrected fresh-seed signal
gate still precedes every training run; all decisions stop without Task 4.

## Historical Task-3 training-distribution v1

Replay sampling increased repeated draws of the same few kill windows without
improving official kills. The current final Task-3 attempt freezes the complete D
learner and changes only the training reset-state distribution. It first runs a
no-learning frozen-v4 signal gate and starts training only if the kill-rich reset
produces sufficiently dense, seed-diverse kill events. Validate without running
any game or training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/task3_training_distribution.py \
  experiments/configs/task3-training-distribution-matched-s112000.json
```

Only Luna should add `--execute`, following
`TASK3_TRAINING_DISTRIBUTION_LUNA_PROMPT.md`. A failed signal gate stops before
training. A passed gate conditionally trains three matched classic/kill-rich pairs,
evaluates all candidates on official classic, and then stops for either decision.
Task 4 is never started automatically.

## Historical Task-3 replay-sampling experiment

The D-only multiseed confirmation improved pooled score but did not improve kills.
This completed one-shot matched experiment froze the full D recipe and changed only
the replay mini-batch distribution: uniform 64/64 controls versus 16/64 kill-causal
samples in the candidates. Validate the bound preregistration without training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/task3_replay_sampling.py \
  experiments/configs/task3-replay-sampling-matched-s110000.json
```

Only Luna should add `--execute`, following
`TASK3_REPLAY_SAMPLING_LUNA_PROMPT.md`. The single command trains three fresh
matched control/candidate pairs, evaluates v4 and all six models in separate
peaceful and coin-collector strata, applies exact rational gates, and stops for
either decision. It never scans sampling settings or starts Task 4.

## Historical Task-3 D-only multiseed confirmation

The completed four-arm discovery selected D. This historical experiment reproduced
that unchanged recipe with three fresh training seeds, then compares every
replica against frozen v4 on twelve new paired evaluation seed tuples. Validate
the bound preregistration without launching training:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/task3_d_multiseed.py \
  experiments/configs/task3-d-multiseed-confirmation-s108300.json
```

Only Luna should add `--execute`, following `TASK3_D_MULTISEED_LUNA_PROMPT.md`.
That one command trains R1/R2/R3, evaluates v4 and all replicas in separate
peaceful and coin-collector strata, applies exact rational-number gates, and
stops after the confirmation decision. It never starts the official-opponent
gate or Task 4.

## Historical v10 diagnostic stop point

The v10.10 discovery gate found a promising but not yet selectable low-margin
WAIT filter: all target-repair checks passed and mean remaining crates improved
by 1.75, but one clear episode was exchanged for another and the strict
per-seed clear safeguard failed. The current stop point is an independent
24-seed confirmation of the unchanged filter. Discovery outcomes are not pooled
for selection. The exact Luna execution protocol is documented in
`V10_TRAINING_PROMPT.md`. Validate it without `--execute` first:

```bash
/opt/anaconda3/envs/prm_env/bin/python -W error::RuntimeWarning \
  tools/v10_teacher_wait_filter_confirmation.py \
  experiments/configs/v10.11-task2-teacher-wait-filter-confirmation-s106180.json
```

The diagnostic requires the explicit `--execute` flag and writes one atomic
JSON report, with no checkpoint or model fit. Never reuse a run ID or delete an
existing artifact to restart it. Stop after any decision; every subsequent
teacher gate or student experiment requires a separate preregistration.
