# Status, Tried Work, Tests, Readiness, and Roadmap

This is the blunt status file. Do not optimize this document for reassurance. Optimize it for correct future work.

## Current Score State

Starting score given by user:

```text
7157.53
```

New accepted optimizer-produced gain:

```text
task233 accepted delta = 0.0005439444473971378
```

Projected score after this accepted gain:

```text
7157.530543944447
```

Gap to 7900:

```text
7900 - 7157.530543944447 = 742.469456055553
```

## Is It 7900+ Ready?

No.

Proof:

```text
accepted candidates found in this pass: 1
total accepted delta in this pass: 0.0005439444473971378
projected score: 7157.530543944447
required score: 7900+
remaining gap: about 742.47 points
```

The optimizer now has correct evaluator diagnostics, a real accepted promotion path, and automatic repair integration. It does not yet have enough rule-understanding or structural synthesis power to approach 7900.

## Accepted Candidate Proof

Task:

```text
task233
```

Optimizer run:

```text
run_id: task233_concat_pad_accept
family: concat_pad
trials: 3
promoted: true
```

Accepted row:

```text
old incumbent cost: 71718
new candidate cost: 71679
cost reduction: 39
public_wrong: 0
arc_gen_wrong: 0
fresh: 199/200
delta_points: 0.0005439444473971378
model: task233/experiments/optimizer_runs/task233_concat_pad_accept/candidate_000003_concat_pad.onnx
```

Post-promotion strict pair proof:

```text
model: task233/experiments/candidate.onnx
pair_runner official_public_arc:
  exact = 266
  wrong = 0
```

This was not merely manual task surgery. The accepted gain came from a fixed reusable optimizer family:

```text
concat_pad:
  two uint8 zero-tail Concats -> Pad
  Pad value dtype fixed to match zero initializer dtype
  bool Pad rewrites skipped because ORT rejects bool Pad
```

## Required Health Check Proofs

Compile:

```text
PYTHONPATH=.chatgpt/operations python3.12 -m compileall -q .chatgpt/operations/neurogolf_optimizer
passed
```

Unit tests:

```text
PYTHONPATH=.chatgpt/operations python3.12 -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
Ran 46 tests
OK
```

## Official Evaluator Mismatch Fixes Tried and Proven

### pair_runner

Fixed comparison modes:

```text
official_public_arc
fresh_grid
both
```

Public/ARC now uses exact tensor equality, matching the official validator. Fresh can still use relaxed grid comparison.

Tests added:

```text
public/ARC exact one-hot equality
fresh relaxed missing-zero grid comparison
active black outside expected shape detected as extra activation
```

### task031 Crop Encoding

Original failure:

```text
program = crop:crop_nonzero_bbox
Python relaxed score = 1.0
official public_wrong = 4
official arc_gen_wrong = 262
cost = null
```

Root cause:

```text
outside real crop: active black channel 0
official expected: no active channel
```

Fix:

```text
crop/bbox emitters use no-color all-zero channels outside valid output shape
static shape preflight rejects cost-null candidates
```

Proof:

```text
task031_fix_probe candidate:
pair_runner official_public_arc = 266/266
official wrapper:
  public_wrong = 0
  arc_gen_wrong = 0
  fresh = 200/200
  cost = 685640
  memory = 685536
  params = 104
```

Not promoted because incumbent is much cheaper.

### task150/task155 Flip/Rotate Encoding

Proof:

```text
strict pair_runner official_public_arc = 266/266
official public_wrong = 0
official arc_gen_wrong = 0
fresh = 200/200
```

Not promoted because incumbent costs are tiny.

## Near Misses Tried

### task014

Known near miss:

```text
task014/experiments/attempt2.onnx
cost = 7905
public_wrong = 0
arc_gen_wrong = 1
fresh = 200/200
incumbent = 7983
```

Strict diagnostic:

```text
one ARC example fails
expected shape = 7x18
predicted missing column 17
missing active cells = 7
no active black outside expected shape
```

Repair attempts:

```text
repair_no_color_outside_7x18:
  public_wrong = 2
  arc_gen_wrong = 129
  fresh = 81/200
  cost = 53806

repair_clear_channel0_outside_7x18:
  public_wrong = 2
  arc_gen_wrong = 114
  fresh = 100/200
  cost = 70905

repair_tiny_coordinate_patch_0:
  public_wrong = 4
  arc_gen_wrong = 261
  fresh = 0/200
  cost = 70905

cleanup_safe_build:
  public_wrong = 0
  arc_gen_wrong = 1
  fresh = 200/200
  cost = 7905
```

Manual experimental branch:

```text
restore missing 18th column symbolically
public/ARC exact = 266/266
official wrapper exact
cost = 8290
not accepted because cost > incumbent 7983
```

Blocker:

```text
late branch repair creates extra intermediates and destroys the cost win.
Need symbolic width-18 handling that does not materialize both 17-wide and 18-wide paths.
```

### task396

Known near miss:

```text
task396/experiments/ablate_141_else.onnx
cost = 27441
incumbent = 27662
public_wrong = 0
arc_gen_wrong = 3
fresh = 199/200
```

Strict diagnostic:

```text
three ARC examples miss bottom row/border activations
failures are no-color where expected color should be active
```

Blocker:

```text
near miss comes from selector/extent branch loss.
Restoring full exact branch likely loses the 221-cost advantage.
Need branch restoration repair that restores only missing scalar extent logic.
```

### task066

Known near miss:

```text
cost = 20079
incumbent = 20081
public_wrong = 1
arc_gen_wrong = 0
fresh = 200/200
```

Strict diagnostic:

```text
one public example
12 cells expected color 3
predicted active black 0
```

Exact graph difference:

```text
one 1x1 Where in red/cyan branch
removing it saves 2 cost but causes failure
```

Tried generic compression:

```text
all-false bool mask initializers -> Constant nodes
exact behavior preserved
cost increased from 20081 to 20581
```

Tried `Pad` for bool:

```text
ORT rejected Pad on bool tensor
```

Blocker:

```text
the exact selector appears already near minimal under current representation.
Need a larger structural rewrite, not a tiny repair.
```

### task233

Successful generic rewrite:

```text
two uint8 zero-tail Concats -> Pad
accepted and promoted
```

Important lesson:

```text
operator dtype legality matters.
Pad works for uint8 case, not bool case observed in task066.
```

## Campaign Accounting Tried

Old failure:

```text
task_order.json listed 11 tasks
score_gain_report.json had only 10 summaries
missing task was not counted as error
```

Fixed:

```text
task_status.json
progress.jsonl
terminal status finalization for pending/running tasks
SIGTERM/KeyboardInterrupt handling
global and per-task time limit handling
```

Proof:

```text
accounting_fix_probe:
completed_tasks = ['task014']
missing_tasks = []
task014 status = completed
```

## Automatic Near-Miss Repair Integration Tried

Old state:

```text
near_miss_lab.py had repairs, but main optimizer only tried subset cleanup-like mutation repair.
```

New state:

```text
optimizer.py calls near_miss_lab.try_repair_model for near-miss candidates
repair outputs become normal Candidate records
official validation, retention, promotion, and accounting are shared
```

Proof:

```text
auto_repair_integration_probe:
task014 dimension_sweep with trials=1
main optimizer produced 2 repair rows
trials became 3
repair rows:
  repair_no_color_outside_6x17
  repair_clear_channel0_outside_6x17
```

No accepted repair from that probe.

## Why This Is Still Far From 7900

The optimizer has mainly gained:

```text
correct evaluator diagnostics
better candidate retention
campaign accounting
some reusable graph rewrites
waterfall synthesis scaffold
official-compatible compiler fixes
automatic near-miss repair loop
one tiny accepted gain
```

The missing capability is:

```text
rule understanding
rule decomposition
compact ONNX compilation of task rules
cost-aware structural replacement of full graphs
symbolic near-miss repair
large-scale cross-task pattern mining
```

To gain about 742 points, the optimizer must produce many large reductions. For intuition:

```text
halving a model cost yields about ln(2) = 0.693 points
reducing cost by 10x yields about ln(10) = 2.303 points
```

The current accepted task233 gain is about:

```text
0.00054 points
```

Therefore small cleanup/pruning gains cannot reach the target.

## Roadmap to a Real 7900+ Optimizer

### Phase 1: Correctness and Accounting Foundation

Status: mostly done.

Required properties:

```text
official-compatible public/ARC diagnostics
fresh relaxed diagnostics isolated
official validator wrapper reliable
no silent campaign task omission
diagnostic rejected candidates retained
cost-null candidates rejected before validation
automatic repair loop integrated
```

Remaining:

```text
make all long campaign interruptions produce clean reports under process kill scenarios
add more integration tests for timeout/resume with worker processes
```

### Phase 2: Cross-Task Graph Pattern Mining

Status: scaffolded.

Build a miner that scans all 400 incumbents and near misses for:

```text
zero-padding concats
all-false/all-zero large masks
repeated macro subgraphs
arange-driven work dimensions
one-hot volume construction
selector branch families
large gather tables
constant payloads
dtype lowering opportunities
crop/pad/no-color patterns
```

Output should be a ranked pattern database:

```text
pattern id
affected tasks
estimated savings
operator/dtype constraints
known ORT legality
safe rewrite family candidate
test fixture to add
```

Goal:

```text
create reusable mutation families with measurable accepted gain across multiple tasks
```

Current implementation:

```text
pattern_miner.py scans accepted incumbents when accept notes exist, otherwise baselines
pattern_report.json and top_opportunities.json are machine-readable ranked reports
PATTERN_REPORT.md summarizes top opportunities and sentinel tasks
broadcast_scalarize adds a low-risk uniform-initializer rewrite from mined broadcast patterns
```

### Phase 3: Rule Decomposition Synthesis

Status: scaffold exists, capability insufficient.

Needed:

```text
extract task examples
infer objects, colors, bboxes, spans, symmetries, markers
generate decomposed subgoals
search subgoal solvers independently
compose into full programs
compile to static ONNX
reject cost-null or high-cost candidates before official validation
```

The DSL must support:

```text
conditional branches
object selection
bbox/span math
line/rectangle drawing
mask composition
color replacement
shape decisions
tiling/repetition
symmetry/rotation/flip
hole/background preservation
```

### Phase 4: Cost-Aware ONNX Compiler

Status: basic templates exist.

Needed:

```text
compiler chooses scalar/grid/one-hot representations per subgoal
static memory estimator predicts scorer impact
operator legality table from ONNX/ORT docs plus local probes
automatic no-color correctness for output tensors
static shape guarantee
fallback compiler strategies when dynamic shapes fail
```

The compiler should produce several alternatives per rule:

```text
scalar-heavy
mask-heavy
gather-heavy
reduce-heavy
where-heavy
conv/pool-heavy
```

and rank by official estimated cost.

### Phase 5: Symbolic Near-Miss Repair

Status: early symbolic scaffold exists, not enough.

Needed:

```text
repair missing border with row/col symbolic condition
repair one wrong color class with color-specific condition
repair output width/height by scalar shape selector
restore minimal removed branch fragments
learn repair templates from pair_runner failures
avoid dense 30x30 correction tensors unless still cheaper
```

Current dense repairs are often too expensive.

Current implementation:

```text
near_miss_lab classifies missing row/column line failures from pair_runner first failing coordinates
symbolic missing-line repair emits row/column/channel masks instead of dense 30x30 coordinate tensors
missing trailing column repair is gated by adjacent predicted extent to avoid unconditional public breakage
repair emitter is opset-aware for ReduceSum axes handling
```

Current proof and blocker:

```text
task014 attempt2 near miss: public_wrong=0, arc_gen_wrong=1, fresh=200/200, cost=7905
symbolic missing_last_column repair builds and validates
result: public_wrong=0, arc_gen_wrong=3, fresh=200/200, cost=85000
not accepted; selector logic is still too broad and cost is too high
```

Next repair work should make selector conditions depend on compact task-derived extent/bbox/color signals, not only adjacent raw-output activity.

### Phase 6: Campaign Intelligence

Status: basic task campaigns exist.

Needed:

```text
priority queue by expected score delta
learned useful-trial predictor
per-family success/failure stats
resume with full deterministic state
automatic follow-up repair campaigns
accepted-gain dashboard
cross-task pattern mining after every campaign
```

Campaigns should stop spending trials on:

```text
families that repeatedly build duplicates
known ORT-invalid operator/dtype combos
cost-null graph shapes
high-cost exact synthesis candidates
near misses with no cheap symbolic repair path
```

### Phase 7: 7900 Readiness Gate

The optimizer should not be called 7900+ ready until a bounded campaign proves:

```text
accepted delta large enough to project 7900+
all promoted candidates officially accepted
fresh_min >= 0.90 for every promoted candidate
no missing tasks in campaign report
repeat run can reproduce accepted candidates or explain cache differences
```

Minimum strong proof:

```text
score_gain_campaign over all 400 tasks
accepted_count high enough to exceed 7900 projected score
score_gain_report.json complete
every accepted model exists
every promoted task has notes and official validation output
```

## What Future Agents Should Do Next

Do not hand-optimize individual tasks unless the result becomes a reusable optimizer feature.

Recommended next work:

1. Build the cross-task graph pattern miner.
2. Auto-generate mutation-family candidates from mined patterns.
3. Expand DSL and compiler for conditional rule decomposition.
4. Replace dense near-miss repairs with symbolic repairs.
5. Add official ONNX/ORT operator compatibility probes.
6. Run short campaigns to measure accepted gain per family.
7. Promote only official accepted lower-cost candidates.

The target is not more wrappers. The target is many official-valid structural rewrites.
