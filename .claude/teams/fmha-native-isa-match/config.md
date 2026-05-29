---
template: hip-kernel-team
version: "2.0"
name: fmha-native-isa-match
created: 2026-05-28
updated: 2026-05-29
---

# Team: fmha-native-isa-match

## Goal

**Primary (must achieve):** Pass 50/50 correctness tests + 48/48 GPU
ref tests using CK-matched data layouts verified by golden dumps.

**Secondary (target):** Match CK's ISA — VGPR ≤ 127, AGPR = 0, 0
spills, 0 function calls, 32 MFMA.

**Stretch:** Performance within ±2% TFLOPS of CK.

Design spec: `docs/superpowers/specs/2026-05-29-fmha-native-d64-design.md`
Plan: `docs/superpowers/plans/2026-05-29-fmha-native-d64.md`

## Roles

| Name | Role | Model | Focus |
|------|------|-------|-------|
| lead | Lead | (parent) | Coordinate, assign, decide |
| research | Researcher | opus | Read CK source, instrument CK, dump tensors, document layouts |
| impl | Implementer | opus | Write kernel code, CPU refs, test harnesses |
| build | Builder | sonnet | Compile, extract assembly, report errors |
| test | Tester | sonnet | Run tests, compare against golden/CPU ref, report |
| debug | Debugger | opus | Investigate failures, report root cause with fix |
| prof | Profiler | opus | Assembly analysis, instruction count checks |

## Critical Thinking Obligation

Every agent — regardless of role — must sanity-check the inputs they
receive before acting. Do NOT blindly execute a task that looks wrong.
Your job is to be smart, not obedient.

### What every agent checks

1. **Does the task make sense?** If a hypothesis document claims MFMA
   produces 32 registers per lane but the ISA spec says 16, flag it
   before implementing. Don't build something you know is wrong.

2. **Is the data plausible?** If golden dump data is all zeros, or all
   identical values, or has obvious patterns that don't match expected
   behavior — flag it. "Test passed" on garbage data is worse than a
   test failure.

3. **Does the direction match the goal?** If you're asked to do
   something that contradicts the spec, the layout map, or prior
   verified results — escalate. "The spec says X but this task asks
   for Y" is exactly the kind of feedback that prevents wasted work.

4. **Are assumptions still valid?** If during your work you discover
   that an earlier assumption was wrong (e.g., CK doesn't use
   `shuffle_tile()` for P — it uses a different mechanism), escalate
   immediately. This is more valuable than completing your current task.

### Per-role checks

| Role | Must check before acting |
|------|-------------------------|
| research | Does the CK source match what the plan assumes? If CK's pipeline structure differs from the spec's model, escalate before writing hypotheses based on wrong assumptions. |
| impl | Does the hypothesis document have internal consistency? Do register counts match MFMA spec? Does the addressing formula produce valid LDS offsets (within 13824 bytes)? |
| build | Does the build produce unexpected warnings? Are there implicit type conversions or truncations that suggest a bug? Report warnings even if build succeeds. |
| test | Is the comparison meaningful? Are both sides non-trivial? If golden data or kernel output is all zeros / all identical, the test is vacuous — report it. |
| debug | Does the root cause point to a design flaw (not just an implementation bug)? If yes, escalate to lead — don't route a design fix to impl as a code fix. |
| prof | Does the assembly structure match what the spec predicts? If the compiler generated a fundamentally different structure (e.g., loop instead of unrolled), that's a design concern — not just a gate fail. |

### How to escalate

Message lead with:
- **What you noticed** (specific observation, not vague concern)
- **What you expected** (what the spec/plan/hypothesis says)
- **Why it matters** (what goes wrong if we proceed)

Do NOT: proceed silently, implement a workaround, or assume someone
else will catch it.

## Communication Patterns

Each member communicates with specific peers. No broadcasting.

```
impl → build         "ready to compile test_X"
build → test          "binary ready" (on success)
build → impl          "build failed: <error>" (on failure)
test → prof           "tests pass, ready for assembly check" (on pass)
test → debug          "test failed: <details>" (on fail)
debug → impl          "root cause: X, fix: Y"
prof → lead           "assembly gate pass/fail: <details>"
research → lead       "Phase 0 deliverable ready"
lead → user           "gate failure: <what/expected/observed>"
```

Members do NOT message lead for routine progress — only for gate
results, escalations, and phase completions.

## Bug Handling Workflow

### Simple bugs (build error, wrong offset, off-by-one)

Autonomous loop — no lead intervention needed:

```
test detects failure → messages debug
debug investigates → messages impl with root cause + fix
impl fixes → messages build
build recompiles → messages test
test reruns → pass (done) or fail (loop again)
```

The original task stays in_progress until it passes.

### Escalation triggers

Debug escalates to lead when:
- Root cause unclear after investigation
- Fix requires approach change (not just code fix)
- Same task fails 3 times without progress
- Bug contradicts Phase 0 golden data

### Gate failures → user (ALWAYS)

The lead NEVER accepts or skips a gate failure. Gate failures always
escalate to the user with:
- What failed (which kernel, which gate level)
- What was expected
- What was observed
- Suggested next step

Gate failures include:
- CPU ref mismatch beyond tolerance
- Golden data mismatch (bit-exact fail)
- Assembly checklist violation (wrong instruction type, scratch spill)
- Integration test regression

### Iteration budget

| Failure type | Max loops | Then |
|-------------|:---------:|------|
| Build error | 3 | debug escalates to lead |
| CPU ref test fail | 3 | debug escalates to lead |
| Golden data test fail | 2 | debug escalates to lead (golden may be wrong) |
| Assembly gate fail | 0 | lead escalates to user immediately |

## Environment

- Container: `poyenc-fmha` (fmha_native mounted at `/root/workspace`)
- Container (CK baseline): `poyenc-ck` (CK instrumentation only)
- Host workspace: `/home/poyenc/workspace/repo/fmha_native`

## Workflows

### Build standalone kernel test
```bash
docker exec poyenc-fmha bash -c "cd /root/workspace && mkdir -p build && cd build && cmake .. -GNinja && ninja test_<name> 2>&1"
```

### Run standalone kernel test (CPU ref)
```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_<name> --ref=cpu 2>&1"
```

### Run standalone kernel test (golden data)
```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_<name> --golden=/tmp/fmha-native-isa-match/golden/ 2>&1"
```

### Build full FMHA kernel
```bash
docker exec poyenc-fmha bash -c "cd /root/workspace && mkdir -p build && cd build && cmake .. -GNinja && ninja test_fmha_fwd_d64 test_fmha_gpu_ref bench_fmha_fwd 2>&1"
```

### Test (GPU ref — must pass first)
```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_fmha_gpu_ref 2>&1"
```

### Test (kernel correctness)
```bash
docker exec poyenc-fmha bash -c "/root/workspace/build/test_fmha_fwd_d64 2>&1"
```

### Extract Assembly
```bash
docker cp poyenc-fmha:/root/workspace/build/fmha_fwd_d64_kernel-hip-amdgcn-amd-amdhsa-gfx942.s /home/poyenc/workspace/repo/fmha_native/native_d64_kernel.s
```

### Build CK instrumented kernel
```bash
docker exec poyenc-ck bash -c "cd /root/workspace && <CK build command> 2>&1"
```

### Benchmark
```bash
docker exec poyenc-fmha bash -c "cd /root/workspace && bash scripts/run-benchmark.sh --d64"
```

## Key Files

See design spec for full file organization:
`docs/superpowers/specs/2026-05-29-fmha-native-d64-design.md`

### Existing kernel (untouched until Phase 2)
- `src/kernel/fmha_fwd_d64_device.hpp`
- `src/kernel/fmha_fwd_d64_gemm.hpp`
- `src/kernel/fmha_fwd_d64_lds.hpp`
- `src/kernel/fmha_fwd_d64_softmax.hpp`
- `src/kernel/fmha_fwd_d64_epilog.hpp`

### CK Reference (read-only)
- CK assembly: `~/workspace/repo/rocm-libraries/projects/composablekernel/ck_d64_kernel.s`
- CK source: `~/workspace/repo/rocm-libraries/projects/composablekernel/`

## Output Directory

`/tmp/fmha-native-isa-match/<member-name>/`

Golden data: `/tmp/fmha-native-isa-match/golden/`

## Recall

- Enabled: true
- Project: fmha_native
- Branch: isa-match-rewrite
- Task: isa-match-rewrite
- Path: `~/.local/share/claude/recall/fmha_native/branches/isa-match-rewrite/tasks/isa-match-rewrite/`

## Memory

- sync_to_memory: false

## Documentation Protocol

This is substantial work. ALL agents — including the lead — must record
findings, decisions, and intermediate results as files. Messages are for
coordination; files are for knowledge.

### Why

- Agents will be rotated frequently (context limits)
- Successor agents read files, not prior message history
- User and lead spot errors early by reviewing files, not trusting agent claims
- If it's not in a file, it doesn't exist

### What each role writes

| Role | Writes to | Content |
|------|-----------|---------|
| research | `/tmp/fmha-native-isa-match/research/<topic>.md` | Findings, traced layouts, distribution types, worked examples |
| impl | `/tmp/fmha-native-isa-match/impl/<kernel>_<desc>.md` | Design decisions, deviations from spec, self-resolved issues |
| build | `/tmp/fmha-native-isa-match/build/<kernel>_build_NNN.txt` | Full build output (stdout+stderr, unfiltered) |
| test | `/tmp/fmha-native-isa-match/test/<kernel>_<ref>_NNN.txt` | Full test output + pass/fail summary |
| debug | `/tmp/fmha-native-isa-match/debug/<kernel>_<issue>.md` | Root cause analysis, what was tried, fix recommendation |
| prof | `/tmp/fmha-native-isa-match/prof/<kernel>_asm_NNN.md` | Assembly analysis, instruction counts, gate pass/fail |
| lead | `/tmp/fmha-native-isa-match/lead/decisions_NNN.md` | Task assignment rationale, gate results, escalation decisions |

### Rules

1. **Write before messaging.** Save findings to file first, then
   message the recipient with the file path. Never send findings inline
   in a message body.
2. **Decisions must be written.** Every approach choice, deviation from
   spec, or judgment call gets a one-line entry in a decisions file.
   Format: `YYYY-MM-DD HH:MM | <decision> | <rationale>`
3. **Reference, don't duplicate.** When referring to prior work, cite
   the file path. Don't copy content from one file into another.
4. **Command output goes to files.** Never filter command output with
   grep/head/tail when capturing. Save full stdout+stderr to a file,
   then read/analyze the file separately.

## Rotation and Handoff

### When to rotate

- Lead monitors teammate token usage (%) via tmux windows
- Rotate at **60%** context usage — do NOT wait until degradation
- Override: rotate immediately on quality degradation regardless of usage
- Never rotate mid-task

### Rotation point tracking

Track each member's completed tasks: heavy=1 point, light=0.5 points.
Rotate at 3 points. Reset to 0 after rotation.

**Heavy tasks:** build+test, large file analysis (>500 lines), code
changes with build, multi-step research.
**Light tasks:** small edits without build, single grep/read, status
saves.

### Handoff format

When rotating a member, the outgoing agent writes a handoff file
following this structure. Keep it **concise** — reference files by path,
don't duplicate their content.

**File:** `.claude/teams/fmha-native-isa-match/status/<name>.md`

```markdown
# Handoff: <role> — <date>

## Current Task
- Task ID: #N — <subject>
- Status: <done / in-progress at step X / blocked on Y>

## Completed This Session
- Task #A: <one-line result> → <output file path>
- Task #B: <one-line result> → <output file path>

## Key Findings
- <one-line finding> — see <file path>
- <one-line finding> — see <file path>

## Decisions Made
- <decision> — rationale: <why> — see <file path>

## Open Questions / Blockers
- <question or blocker, if any>

## Files Written
- <path>: <one-line description>
- <path>: <one-line description>

## Next Action
<what the successor should do first>
```

### Handoff procedure

1. Lead messages member: "Prepare for rotation — save status"
2. Member writes handoff to `.claude/teams/fmha-native-isa-match/status/<name>.md`
3. Member confirms to lead
4. Lead sends `shutdown_request`
5. Lead spawns replacement with same role
6. Replacement reads handoff file and TaskList as first action
7. Lead assigns the unfinished or next task

### Lead's own handoff

When lead's context is getting high:
1. Write lead handoff to `.claude/teams/fmha-native-isa-match/status/lead.md`
2. Shut down all members (they save handoff first)
3. Tell user: "Team paused. Run `/hip-kernel-team load fmha-native-isa-match` to resume."

## Context Management

- Delegate reads above: 500 lines to Explore subagent
- Assembly files (.s): ALWAYS via Explore subagent, never read directly

## Constraints

See design spec for full constraints. Key limits:
- Target: gfx942 (MI-300X, CDNA3)
- Final kernel: VGPR ≤ 127, AGPR = 0, no spills, LDS = 13824
- Standalone kernels: no register pressure requirement
