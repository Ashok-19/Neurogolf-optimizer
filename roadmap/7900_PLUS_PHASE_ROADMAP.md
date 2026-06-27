# NeuroGolf Optimizer 7900+ Phase Roadmap

This is the central roadmap for the NeuroGolf optimizer project.

It is intentionally detailed. Future sessions and future coding agents should use this as the project control document before making changes. The goal is not to add random features or keep trying local ONNX cleanup. The goal is to build a deterministic, intelligent optimizer that can take a NeuroGolf task, infer the task rule, and rewrite the ONNX model into the cheapest correct static graph representation possible.

## 0. North Star

### Final Goal

Build a deterministic optimizer that can move the project score from:

```text
current projected score: 7157.530543944447
required score: 7900+
remaining gap: about 742.47 points
```

to a proven 7900+ projected score using officially accepted lower-cost candidates.

### What the Optimizer Must Become

The optimizer must evolve from:

```text
local graph simplifier
mutation searcher
ONNX cleanup utility
near-miss logger
```

into:

```text
rule-aware task analyzer
failure-guided repair system
cross-task pattern learner
static ONNX rule compiler
cost-aware representation selector
deterministic campaign optimizer
```

The final system should be able to take a task, understand the rule class, and emit a compact static graph that solves the rule directly. It should not merely try to remove unnecessary nodes like `onnxsim`, `onnxoptimizer`, or small hand-written cleanup passes.

### Non-Negotiable Success Metric

A candidate counts as success only when the official task validator accepts it:

```text
public_wrong == 0
arc_gen_wrong == 0
fresh_right / fresh_total >= fresh_min
candidate_cost < current_incumbent_cost
```

Default fresh threshold:

```text
fresh_min = 0.90
```

Anything else is diagnostic only:

```text
Python DSL exactness
pair_runner exactness
relaxed grid exactness
near-miss count
number of candidates generated
number of graph patterns mined
number of tests passing
```

These are important, but they do not move the score unless they produce officially accepted lower-cost candidates.

---

## 1. Current Baseline and Reality Check

### Current Score State

```text
user baseline/projected score: 7157.53
accepted optimizer-produced gain: 0.0005439444473971378
current projected score: 7157.530543944447
target score: 7900+
remaining gap: 742.469456055553
```

### Only Accepted Optimizer Gain So Far

```text
task233:
  old cost: 71718
  new cost: 71679
  reduction: 39
  public_wrong: 0
  arc_gen_wrong: 0
  fresh: 199/200
  delta: 0.0005439444473971378
  family: concat_pad
```

This proves the validation/promotion pipeline works, but it also proves that local cleanup is not enough.

### Score-Scale Intuition

The NeuroGolf score contribution behaves like:

```text
points = max(1, 25 - ln(cost))
cost = memory + params
```

Therefore:

```text
halving one model cost gives about ln(2) = 0.693 points
reducing one model by 10x gives about ln(10) = 2.303 points
reducing one model by 100x gives about ln(100) = 4.605 points
```

The current accepted gain is about:

```text
0.00054 points
```

So reaching 7900+ needs many large structural rewrites, not small node trims.

---

## 2. Project Principles

### Principle 1: Reusable Optimizer Capability Only

Task-specific investigation is allowed only if the result becomes at least one of:

```text
new reusable mutation family
new repair family
new DSL primitive
new compiler template
new pattern-mining rule
new cost model improvement
new diagnostics feature
new regression test
new campaign scheduler behavior
```

A one-off model file under `taskXXX/experiments/` is not enough.

### Principle 2: One Feature at a Time

Do not overwhelm the coding agent with multiple broad features in one prompt.

Each implementation cycle should focus on exactly one primary feature:

```text
one feature
one sentinel task set
one measurable success criterion
one review pass
```

The reviewer should audit the feature before starting the next one.

### Principle 3: Official Validator is Final

`pair_runner.py` is for diagnostics. It is not the acceptance oracle.

Acceptance must come from:

```text
taskXXX/validate_task.py
```

through the optimizer validation path.

### Principle 4: Avoid Dense Final Patches

Dense final-output patches can prove a failure is repairable, but they usually destroy the cost advantage.

Avoid as primary score-gain strategy:

```text
full [1,10,30,30] correction tensors
large coordinate masks
late output-wide Where patches
per-example lookup patches
large Constant payloads
```

Prefer:

```text
scalar conditions
row/column masks
bbox extents
channel-specific masks
intermediate selector repair
small Gather/Scatter patches
static rule compiler templates
```

### Principle 5: Cost Must Be Designed, Not Discovered Accidentally

The official score depends on declared static intermediate memory and parameter count. The optimizer must reason about cost before expensive validation.

Every serious candidate path should ask:

```text
What intermediates does this create?
What initializers does this create?
Are shapes static?
Are outputs excluded from memory cost?
Can this be represented with scalar/grid state instead of one-hot volume?
Will shape inference expose huge intermediate tensors?
```

---

## 3. Current System Map

### Existing Strengths

The optimizer already has strong foundation pieces:

```text
official validation wrapper
validation cache with profile isolation
safe promotion path
campaign accounting
pair_runner official tensor diagnostics
near-miss retention
pattern mining scaffold
mutation family interface
synthesis DSL scaffold
ONNX emitter scaffold
unit tests: 55 passing
```

### Existing Weaknesses

The optimizer is still weak in the parts required for 7900+:

```text
near-miss repair is too late and too expensive
synthesis compiler coverage is narrow
rule decomposition is shallow
pattern miner reports opportunities but does not drive ROI-based campaigns
mutation provenance is incomplete for repair planning
cost estimation is not strong enough for search guidance
most mutation families are local graph edits
```

### Current High-Value Evidence

Known near misses show that the optimizer can find lower-cost almost-correct candidates.

#### task014

```text
incumbent cost: 7983
near miss: task014/experiments/attempt2.onnx
near miss cost: 7905
public_wrong: 0
arc_gen_wrong: 1
fresh: 200/200
failure: one ARC example missing final column 17 in a 7x18 output
```

Current symbolic repair builds, but fails:

```text
repair_symbolic_missing_last_column_0:
  public_wrong: 0
  arc_gen_wrong: 3
  fresh: 200/200
  cost: 85000
```

Meaning:

```text
repair plumbing works
symbolic classification works partially
final-output repair is too expensive
selector logic is too broad
```

#### task396

```text
incumbent cost: 27662
near miss: task396/experiments/ablate_141_else.onnx
near miss cost: 27441
public_wrong: 0
arc_gen_wrong: 3
fresh: 199/200
failure: missing bottom row/border activations
```

Likely cause:

```text
selector/extent branch loss
```

Required solution:

```text
restore only missing scalar/border logic, not full branch
```

#### task066

```text
incumbent cost: 20081
near miss cost: 20079
public_wrong: 1
arc_gen_wrong: 0
fresh: 200/200
failure: one public example has 12 cells expected color 3 but predicted black/no-color
```

Likely lesson:

```text
tiny local repair is not enough; needs larger structural rule rewrite
```

---

## 4. Readiness Gates

### Gate A: Safety Foundation Gate

Status: mostly passed.

Required:

```text
unit tests pass
public/ARC pair diagnostics use official tensor equality
fresh relaxed diagnostics are isolated
validation cache profiles separate quick/full/config differences
promotion revalidates candidate
campaign reports no missing tasks silently
```

Current proof:

```text
PYTHONPATH=.chatgpt/operations PYTHONDONTWRITEBYTECODE=1 python3.12 -B -m unittest discover -s .chatgpt/operations/neurogolf_optimizer/tests -q
Ran 55 tests
OK
```

Remaining safety cleanup:

```text
standalone near_miss_lab promotion should write accept notes or be avoided
validation_mode=official-first should be implemented or removed
allow_relaxed_fresh flag should be made meaningful if strict fresh modes are needed
```

### Gate B: Near-Miss Conversion Gate

The optimizer must prove it can convert lower-cost near misses into accepted lower-cost candidates.

Required proof:

```text
at least one accepted repair from a reusable repair family
repair cost remains below incumbent
public_wrong == 0
arc_gen_wrong == 0
fresh >= 0.90
```

Primary sentinel tasks:

```text
task014
task396
```

This gate should be completed before broad synthesis expansion.

### Gate C: Synthesis Compiler Utility Gate

The synthesis system must produce at least one accepted lower-cost candidate, not only high-cost exact candidates.

Required proof:

```text
Python DSL candidate exact or near-exact
compiler emits static ONNX
estimated cost < incumbent before validation
official validator accepts
candidate is reusable through synth_solver
```

### Gate D: Campaign ROI Gate

A short campaign must prove a feature has measurable return.

Required report fields:

```text
accepted_count
accepted_delta
near_miss_count
useful_trial_rate
effective_trial_rate
family-specific accepted/near-miss counts
estimated-vs-actual cost error
errors/timeouts/missing tasks
```

### Gate E: 7900+ Readiness Gate

The optimizer is not 7900+ ready until a bounded campaign proves:

```text
projected_total_score >= 7900
every accepted model exists
every promoted model has official validation proof
every promoted task has accept notes
fresh_min >= 0.90 unless explicitly public-arc-only
no missing campaign tasks
repeat run can reproduce or explain accepted candidates
```

---

## 5. Phase-Level Plan

## Phase 0: Stabilize Project Control and Review Loop

### Goal

Make development controlled and reviewable. Avoid random feature accumulation.

### Status

Partially complete. This roadmap file starts the phase.

### Deliverables

```text
roadmap/7900_PLUS_PHASE_ROADMAP.md
feature-specific prompts generated one at a time
review checklist used after every agent implementation
progress tracked by accepted score gain, not code volume
```

### Required Workflow Per Feature

Each feature cycle must follow this process:

```text
1. Reviewer selects one feature from this roadmap.
2. Reviewer writes a focused prompt for the coding agent.
3. Agent implements only that feature.
4. Agent runs required compile/unit tests.
5. Agent runs sentinel probes.
6. Reviewer audits code and results.
7. Reviewer decides: accept, revise, or rollback/rework.
8. Roadmap/progress is updated.
```

### Done Criteria

```text
roadmap exists
first feature selected
future sessions can understand the plan without prior chat context
```

---

## Phase 1: Internal Selector and Extent Repair

### Goal

Convert lower-cost near misses into accepted candidates by repairing the internal rule logic cheaply.

This is the highest-priority phase.

### Why This Phase Comes First

The optimizer already creates lower-cost near misses. The fastest path to accepted gain is to repair those near misses without losing the cost advantage.

Current failures are localized:

```text
task014: missing final column
task396: missing bottom row/border
```

Current repair system is too expensive because it appends final-output patches.

### Target Transformation

Move from:

```text
late final-output patching
```

to:

```text
internal selector/extent restoration
```

### Required Capabilities

#### 1. Failure Classification

Classify pair_runner diagnostics into reusable failure types:

```text
missing_last_column
missing_column
missing_bottom_row
missing_row
missing_border
wrong_output_width
wrong_output_height
wrong_color_region
missing_object_pixels
extra_active_black_outside_shape
```

Inputs:

```text
first_failing_coordinates
expected_tensor_active_channels
predicted_tensor_active_channels
expected_grid_shape
predicted_tensor_shape
missing_active_channels
extra_active_channels
no_color_cells
```

#### 2. Mutation Provenance for Repair

High-risk mutations must preserve enough metadata for repair:

For `dimension_sweep`:

```text
axis
old_len
new_len
source arange initializer
node name
node output
related row/col initializers modified
pad_to_30 old/new values
shape effects if inferable
```

For `where_ablation`:

```text
condition tensor
selected branch
dropped branch
selected shape
dropped shape
output shape
consumer list
estimated dead nodes
estimated dead initializers
```

For `where_combo`:

```text
all selected/dropped branches
per-choice output names
per-choice conditions
```

#### 3. Internal Repair Candidate Generation

Generate cheap repair candidates that modify or restore internal logic instead of final output.

Examples:

```text
restore old width condition only when needed
restore old col arange for final extent only
restore bottom-row mask condition
restore dropped scalar selector branch
insert row/col comparison before final one-hot composition
patch intermediate bool mask, not full output tensor
```

#### 4. Cost Guard

Before official validation, reject or deprioritize repair candidates that clearly cannot win:

```text
estimated cost >= incumbent
large full-output replacement tensors
large Constant payloads
many new [1,10,30,30] intermediates
cost-null risk
```

Dense repairs may still run as diagnostic fallback only when requested.

### Sentinel Commands

Focused near-miss lab:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/near_miss_lab.py \
  --task task014 \
  --model task014/experiments/attempt2.onnx \
  --fresh-min 0.90 \
  --repair \
  --dry-run \
  --keep-rejected \
  --timeout 1200
```

If the task396 near-miss exists:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/near_miss_lab.py \
  --task task396 \
  --model task396/experiments/ablate_141_else.onnx \
  --fresh-min 0.90 \
  --repair \
  --dry-run \
  --keep-rejected \
  --timeout 1200
```

Focused optimizer run:

```bash
python3.12 .chatgpt/operations/neurogolf_optimizer/run_one.py \
  --task task014 \
  --families dimension_sweep \
  --trials 50 \
  --fresh-min 0.90 \
  --dry-run \
  --keep-rejected \
  --run-id internal_selector_repair_probe \
  --timeout 1200
```

### Tests Required

```text
unit test failure classification for missing_last_column
unit test failure classification for missing_bottom_row
unit test repair candidate ordering prefers internal repairs over dense output patches
unit test primary repair path does not add full [1,10,30,30] replacement initializers
unit test repair candidate metadata includes source mutation provenance
unit test cost gate rejects obvious high-cost final-output patches
unit test existing official tensor/no-color semantics remain passing
```

### Phase 1 Done Criteria

Minimum acceptable result:

```text
new reusable internal repair framework exists
unit tests pass
sentinel probes run
current 85000-cost symbolic repair is no longer the primary candidate
repair candidates are cost-gated
```

Strong result:

```text
task014 or task396 gets an officially accepted lower-cost repair
```

Ideal result:

```text
multiple selector/extent near misses convert into accepted candidates
```

---

## Phase 2: Repair-Aware Search and Provenance Loop

### Goal

Make the main optimizer deliberately search for repairable near misses, not only accepted candidates.

### Problem

Current search can find near misses, but the repair loop is still reactive and shallow.

The optimizer should learn:

```text
which families create repairable failures
which mutation metadata predicts repairability
which failure classes are cheap to repair
which near misses should receive extra budget
```

### Required Capabilities

#### 1. Near-Miss Value Model

Rank near misses by repair value:

```text
cost reduction
public_wrong
arc_gen_wrong
fresh rate
failure locality
failure class
source mutation family
repair history for similar failures
```

High-value near miss examples:

```text
public_wrong = 0
arc_gen_wrong <= 5
fresh >= 0.90
cost reduction > 0
localized row/col/border/color failure
```

#### 2. Repair Budget Scheduler

Instead of applying all repairs equally, allocate more budget to high-value candidates.

Budget should depend on:

```text
near_miss_category
cost reduction
failure locality
family-specific historical success
estimated repair cost
```

#### 3. Repair Result Feedback

Record per repair family:

```text
attempted
built
validated
accepted
near_miss_after_repair
cost_delta
public_delta
arc_delta
fresh_delta
estimated_vs_actual_cost
```

#### 4. Search Queue Integration

If a near miss is repairable but not yet accepted, generate follow-up mutations such as:

```text
undo only one risky mutation
restore one dimension
restore one selector branch
combine repair with cheap cleanup
try adjacent extent values
```

### Tests Required

```text
unit test near-miss ranking prefers localized lower-cost failures
unit test repair budget does not spend dense fallback first
unit test repair feedback is recorded in campaign_metrics.json
unit test accepted repair is promoted only through official path
```

### Phase 2 Done Criteria

```text
near-miss repair is part of search strategy, not just diagnostics
campaign reports repair conversion metrics
repairable near misses receive prioritized follow-up
```

---

## Phase 3: Cost Model and Candidate Preflight

### Goal

Stop wasting validation budget on candidates that cannot win due to cost.

### Problem

The optimizer currently validates some candidates that are obviously too expensive after graph construction. Example: task014 symbolic repair with cost 85000 against incumbent 7983.

### Required Capabilities

#### 1. Static Cost Estimator Improvement

Improve estimator to approximate official cost:

```text
initializer params
Constant payload params
inferred value_info memory
large full-volume intermediates
dtype sizes if scorer uses element count or byte count
shape inference missing/dynamic risk
```

#### 2. Cost Risk Labels

Every candidate should get a cost-risk classification before official validation:

```text
cheap_likely
near_incumbent
expensive_likely
cost_null_risk
unknown
```

#### 3. Validation Queue Filtering

Rules:

```text
validate cheap_likely first
validate near_incumbent only if high correctness potential
validate expensive_likely only if keep_high_cost_probes
skip cost_null_risk unless diagnostic mode
```

#### 4. Estimated-vs-Actual Tracking

Every validated candidate should record:

```text
estimated_cost
actual_cost
absolute_error
relative_error
family
mutation ids
```

### Tests Required

```text
unit test full-output patch estimated cost exceeds small incumbent
unit test scalar repair estimated cost is lower than dense patch
unit test cost-null/dynamic shape risk is detected
unit test campaign report includes estimated-vs-actual error
```

### Phase 3 Done Criteria

```text
validation budget shifts toward candidates that can actually improve score
campaign report shows cost-estimation accuracy
high-cost diagnostics are separated from score-gain candidates
```

---

## Phase 4: Actionable Pattern Mining

### Goal

Turn pattern mining from a report into a planning engine.

### Current State

`pattern_miner.py` scans all tasks and ranks opportunities. It is useful but not yet directly connected to implementation or campaign ROI.

### Required Capabilities

#### 1. Pattern-to-Feature Planning

For each top pattern, generate:

```text
candidate rewrite family
sentinel tasks
estimated savings
risk
operator/dtype constraints
needed tests
expected failure modes
```

#### 2. Pattern ROI Tracking

After campaigns, update pattern/family statistics:

```text
attempted tasks
build success rate
validation success rate
accepted count
accepted delta
near-miss count
near-miss-to-accepted conversion
cost estimate accuracy
common failure reasons
```

#### 3. Sentinel Selection

Automatically select sentinel tasks that maximize:

```text
estimated savings
clear pattern occurrence
manageable incumbent cost
known near-miss or exact diagnostic value
low runtime risk
```

#### 4. Pattern Classes to Prioritize

High-priority classes:

```text
selector_branch_family
arange_work_dimension
dimension_sweep near misses
one_hot_volume_construction
large_float_intermediate
large_lookup_table
concat_zero_tail non-bool
dynamic-to-static crop/bbox patterns
```

### Tests Required

```text
unit test pattern opportunity includes sentinel task
unit test accepted campaign updates ROI stats
unit test risky broad patterns are not ranked solely by affected task count
unit test generated feature plan is machine-readable
```

### Phase 4 Done Criteria

```text
pattern miner outputs actionable rewrite plans
campaign results feed back into ranking
agent prompts can be generated from pattern plans
```

---

## Phase 5: Compact Synthesis Compiler Primitives

### Goal

Make the synthesis system produce cheap accepted ONNX candidates, not just correct high-cost candidates.

### Current State

Synthesis audit shows:

```text
tasks_checked: 400
tasks_with_python_public_arc_exact: 14
tasks_with_compilable_candidate: 8
tasks_with_cheap_compilable_candidate: 2
tasks_with_accepted_candidate: 0
```

The DSL supports more than the ONNX compiler can emit.

### Required Compiler Philosophy

The compiler must emit cost-aware static graphs.

Prefer:

```text
scalar state
small row/col masks
bbox extents
uint8/bool labels
Gather/Where/Reduce logic
static shape with no-color masks
```

Avoid:

```text
many full [1,10,30,30] float intermediates
large constant output grids
public-only lookup tables
cost-null dynamic shapes
```

### Compiler Primitives to Add One at a Time

Each primitive should be its own feature cycle.

#### Primitive 1: Cheap BBox Extent Selector

Purpose:

```text
compute bbox width/height/edges cheaply
support internal selector repair and crop rules
```

Needed for:

```text
task014-style width logic
task396-style bottom border logic
crop/bbox synthesis
```

#### Primitive 2: Cheap Border/Line Composer

Purpose:

```text
draw row/column/border based on symbolic conditions
```

Needed for:

```text
missing border repairs
line drawing ARC tasks
rectangle completion tasks
```

#### Primitive 3: Object Color BBox Crop

Purpose:

```text
crop or mask object by color while preserving official no-color semantics
```

Needed for:

```text
crop tasks
object extraction tasks
small exact synthesis wins
```

#### Primitive 4: Scale 2x/3x

Purpose:

```text
compile Scale DSL operation cheaply
```

Must avoid huge repeated one-hot intermediates when possible.

#### Primitive 5: Gravity Row/Column

Purpose:

```text
compile simple gravity tasks with compact row/column movement logic
```

Should only support simple cases first.

#### Primitive 6: Largest/Smallest Object BBox

Purpose:

```text
compile common object selection tasks
```

This is harder. Do after bbox/color/line primitives.

### Tests Required Per Primitive

```text
Python DSL behavior test
ONNX compiler output test
static_shape_preflight test
official tensor no-color semantics test
cost estimate test
sentinel task probe
```

### Phase 5 Done Criteria

```text
at least one synthesis primitive produces an accepted lower-cost candidate
synth_solver can queue it automatically
synthesis audit improves accepted_candidate count above 0
```

---

## Phase 6: Rule Decomposition Engine

### Goal

Infer task rules from examples and decompose them into solver subproblems.

### Problem

Current synthesis mostly enumerates primitive programs. It does not deeply infer rule structure.

### Required Subproblem Types

```text
input/output shape relation
background/no-color handling
object detection
object selection
color mapping
bbox/span computation
geometry transform
line/border/rectangle drawing
copy/paste/stamp placement
symmetry/rotation/flip
conditional branches
output composition
```

### Required Rule Features

For each task, extract:

```text
input colors
output colors
new/removed colors
object list
object bboxes
object sizes
object positions
object symmetries
touches border
dominant colors
shape transformations
color correspondences
marker objects
line/border structures
```

### Program Search Should Become Hierarchical

Instead of only enumerating full programs, search by stages:

```text
1. infer output shape
2. infer selected object/mask
3. infer color transform
4. infer spatial transform
5. infer output composition
6. compile alternatives
7. rank by official exactness and cost
```

### Tests Required

```text
feature extraction tests for objects/colors/bboxes
rule hypothesis tests on synthetic ARC-like tasks
decomposition tests for crop/recolor/line/bbox tasks
compiler integration tests
```

### Phase 6 Done Criteria

```text
rule decomposition generates candidates not found by flat enumeration
compiler can emit at least some decomposed programs
accepted synthesis gains increase across multiple tasks
```

---

## Phase 7: Static Rule-Skeleton Replacement

### Goal

Replace large exported ONNX graphs with compact hand-compiled rule skeletons.

### Why This Is Needed

7900+ requires large cost reductions. The biggest gains will come from replacing expensive generated graphs with direct compact solvers.

### Rule Skeleton Families

Potential skeletons:

```text
identity/recolor
crop bbox
crop color bbox
object mask extraction
border/rectangle draw
line/ray draw
symmetry transform
scale transform
tile/repeat pattern
gravity/simple movement
selector/conditional output
copy/paste object
small lookup-driven shape rule
```

### Skeleton Requirements

Each skeleton must have:

```text
Python verifier
ONNX compiler
cost estimate
static shape proof
official no-color semantics
sentinel tasks
unit tests
failure diagnostics
```

### Candidate Selection

For a given task, generate multiple representations:

```text
scalar-heavy
mask-heavy
gather-heavy
reduce-heavy
where-heavy
```

Pick by:

```text
official exactness probability
fresh generalization
estimated cost
static-shape confidence
known operator legality
```

### Phase 7 Done Criteria

```text
rule-skeleton replacements produce repeated accepted gains
some high-cost tasks reduce by large factors
score projection moves materially, not by tiny deltas
```

---

## Phase 8: Campaign Intelligence and Full-Scale Optimization

### Goal

Run efficient campaigns over many tasks and learn what works.

### Required Campaign Features

```text
per-family ROI
per-pattern ROI
repair conversion metrics
cost model error tracking
failure clustering
timeout accounting
resume safety
deterministic run IDs
accepted score dashboard
```

### Task Scheduling Strategy

Prioritize tasks by:

```text
potential score gain
known near misses
pattern miner opportunities
synthesis exact candidates
high cost with simple rule signatures
family-specific historical ROI
```

Avoid spending on:

```text
families with repeated zero yield
duplicate model hashes
known build-fail signatures
cost-null dynamic models
high-cost exact candidates unless diagnostic
near misses with no cheap repair path
```

### Full Campaign Requirements

For 7900+ claim:

```text
all 400 tasks or selected full scoring set covered
no missing tasks
accepted models exist
accept notes exist
official validation outputs preserved
projected score >= 7900
repeatability documented
```

### Phase 8 Done Criteria

```text
campaign proves sustained score growth
projected score crosses milestones
optimizer can resume/continue intelligently
```

---

## Phase 9: 7900+ Proof and Hardening

### Goal

Prove the optimizer reaches 7900+ and make the result reproducible.

### Required Proof Artifacts

```text
score_gain_report.json
accepted.json
accepted_contributions
per-task accept notes
model paths for every accepted candidate
validation outputs
campaign config
task order
git commit hash if available
cache profile info
```

### Required Checks

```text
no accepted candidate has public_wrong > 0
no accepted candidate has arc_gen_wrong > 0
fresh threshold satisfied
candidate cost below prior incumbent
no missing task summaries
no deleted accepted models
no promotion without notes
```

### Final Readiness Statement Template

Only use this when proven:

```text
The optimizer is 7900+ ready under the current projected scoring baseline.
Projected score: <score>
Accepted delta: <delta>
Accepted candidates: <count>
Campaign: <run_id>
Fresh threshold: <fresh_min>
All accepted candidates passed official validation and are present on disk.
```

Until then, use:

```text
This improves optimizer capability, but it is not 7900+ ready until accepted campaign gain proves it.
```

---

## 6. Immediate Next Feature

### Feature 1: Failure-Guided Internal Selector/Extent Repair

This is the first feature to implement after this roadmap.

### Scope

Only implement the first version of internal selector/extent repair. Do not expand synthesis compiler in the same prompt. Do not add broad campaign learning in the same prompt.

### Target Files

Likely files:

```text
near_miss_lab.py
optimizer.py
pair_runner.py
models.py or new repair metadata module
mutations/dimension_sweep.py
mutations/where_ablation.py
mutations/where_combo.py
tests/test_official_diagnostics.py
tests/test_integration_controls.py
new tests if needed
```

### Specific Requirements

```text
classify missing_last_column and missing_bottom_row failures
preserve mutation provenance required for repair
generate internal repair attempts before final-output patch attempts
cost-gate repair attempts
avoid dense final-output patch as primary repair
run task014 and task396 sentinel probes
```

### Success Criteria

Minimum:

```text
unit tests pass
repair candidates are generated with internal-first ordering
cost gate prevents 85000-style repair from being treated as primary score candidate
sentinel probes produce clear diagnostics
```

Strong:

```text
task014 or task396 accepted lower-cost repair
```

Ideal:

```text
new repair family converts multiple selector/extent near misses into accepted candidates
```

---

## 7. Review Checklist for Every Agent Feature

After each agent implementation, reviewer must check:

### Source Scope

```text
Were changes limited to allowed paths?
Did the agent avoid official task source files?
Did it avoid one-off task hardcoding?
```

### Tests

```text
compileall passed?
unittest passed?
new tests added for new behavior?
old evaluator/no-color tests still pass?
```

### Validation

```text
were sentinel probes run?
were results recorded?
were official validator results used for acceptance?
was pair_runner used only for diagnostics?
```

### Cost

```text
candidate cost below incumbent?
estimated cost recorded?
actual cost recorded?
large dense tensors avoided?
```

### Score Impact

```text
accepted_count
accepted_delta
projected score
near_miss_count
new high-value blockers
```

### Reusability

```text
is this reusable across tasks?
does metadata avoid task-specific node IDs when possible?
is there a pattern/family/test to reuse later?
```

---

## 8. Milestone Score Targets

These are progress milestones, not final readiness claims.

```text
M0: 7157.5305 current projected score
M1: +1 point accepted gain
M2: +10 points accepted gain
M3: +50 points accepted gain
M4: +100 points accepted gain
M5: +250 points accepted gain
M6: +500 points accepted gain
M7: projected score >= 7900
```

The first meaningful proof is not another 0.0005 gain. The first meaningful proof is at least one accepted structural reduction or accepted near-miss repair with visible score impact.

---

## 9. Feature Backlog, Ordered

### P0: Must Do First

```text
internal selector/extent repair
mutation provenance for repair
repair cost gating
sentinel probes on task014/task396
```

### P1: Next

```text
repair-aware search scheduling
near-miss value model
repair conversion metrics
estimated-vs-actual cost tracking
```

### P2: Synthesis Utility

```text
cheap bbox extent compiler
cheap border/line composer
object color bbox crop compiler
scale 2x/3x compiler
gravity simple compiler
```

### P3: Rule Intelligence

```text
rule decomposition engine
conditional DSL constructs
object selection hypotheses
failure-guided synthesis
```

### P4: Campaign Intelligence

```text
pattern ROI tracking
family ROI tracking
full campaign dashboard
7900 readiness gate automation
```

---

## 10. What Not To Do

Do not spend time on:

```text
more documentation without code or validation
manual per-task rewrites that do not become optimizer features
broad prompts asking the agent to implement many phases at once
public-only hacks
dense output correction masks as primary strategy
large constant lookup tables
calling Python DSL exactness success when compiler output is high-cost
promoting from pair_runner alone
ignoring fresh threshold unless explicitly running public-arc-only diagnostics
```

---

## 11. Current Project Truth

The optimizer is promising because it has:

```text
correct validation foundation
real accepted tiny gain
real high-value near misses
pattern miner inventory
synthesis scaffold
repair scaffold
```

The optimizer is not yet 7900+ capable because it lacks:

```text
cheap internal near-miss repair
rule-level structural replacement
cost-aware synthesis compiler
campaign learning from actual ROI
```

The immediate next move is therefore clear:

```text
Build failure-guided internal selector/extent repair first.
Review it rigorously.
Then move to the next single feature.
```

This roadmap should be updated after every major feature review with:

```text
what changed
what passed
what failed
accepted delta
projected score
next selected feature
```
