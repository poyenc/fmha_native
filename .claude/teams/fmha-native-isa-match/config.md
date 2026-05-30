---
template: hip-kernel-team
version: "2.0"
name: fmha-native-isa-match
created: 2026-05-28
updated: 2026-05-30
---

# Team: fmha-native-isa-match

## Goal

See the current spec and plan in `docs/superpowers/specs/` and
`docs/superpowers/plans/`. This config defines HOW the team operates,
not WHAT it builds.

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
- ISA structural divergence (CFG or dependency chain mismatch)
- Test regression (any previously-passing test now fails)
- Spec compliance failure (independent reviewer reports NON-COMPLIANT)

### Iteration budget

| Failure type | Max loops | Then |
|-------------|:---------:|------|
| Build error | 3 | debug escalates to lead |
| CPU ref test fail | 3 | debug escalates to lead |
| Golden data test fail | 2 | debug escalates to lead (golden may be wrong) |
| Assembly gate fail | 0 | lead escalates to user immediately |

## Quality Gate

All members MUST read and follow the quality gate in the current plan
(`docs/superpowers/plans/` — the most recent plan file). The gate is
identical at every task: G0→G1→G2→G3→G4→G5.

**G0 is every member's responsibility, not just lead's.** Before acting
on a formula, file path, golden data, or teammate's claim — verify it
against the actual artifact. Trust the file on disk, not the message.

### Two-Layer Verification

**Layer 1 — Teammate self-verification (mandatory):**
Every teammate, after completing a task and reviewing their own output,
MUST spawn a fresh subagent (1M context) to independently verify their
deliverable. The subagent reads the spec, reads the actual output files,
and reports PASS/FAIL with evidence. The teammate includes the
subagent's verdict in their report to lead. Teammates who report "done"
without a self-verification subagent verdict will have the task
rejected.

**Layer 2 — Lead verification (mandatory):**
The lead NEVER trusts a teammate's claim — including the self-
verification verdict. For every completed task, the lead spawns a
SEPARATE fresh subagent (1M context) to independently verify the
deliverable against the spec and plan. If the subagent cannot give a
specific, evidence-backed reason to pass the gate (e.g., "looks correct"
without citing file:line), the lead reads the artifacts directly and
makes the judgment. Vague subagent verdicts are treated as FAIL.

### What "verify" means

Not "read the summary and agree." Verify means:
- Subagent reads the ACTUAL file on disk (not a teammate's description)
- Subagent checks formulas against the spec (re-derives, not pattern-matches)
- Subagent runs grep to confirm claimed changes exist / old code is gone
- Subagent cites specific file:line for every claim in its verdict
- If the subagent says PASS, it must say WHY with evidence
- If the subagent says FAIL, it must say WHAT is wrong with evidence

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

- Lead monitors teammate context usage (%) and idle/busy state via the
  teammate's tmux pane (see Monitoring Teammates below)
- Rotate at **60%** context usage — do NOT wait until degradation
- Override: rotate immediately on quality degradation regardless of usage
- Never rotate mid-task

### Monitoring Teammates (tmux)

Spawned teammates run as `claude` agent processes, each in its own tmux
pane. The pane is NOT named after the teammate — find it by process, not
by window name:

```bash
# 1. Map panes to PIDs and commands
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid} #{pane_current_command}'

# 2. Find the pane whose pid (or child) runs the target agent
ps -ef | grep -- '--agent-name <name>'   # e.g. --agent-name research

# 3. Read that pane's live output (status bar shows context% + cost + busy/idle)
tmux capture-pane -t <session:window.pane> -p | tail -50
```

The teammate's status bar reports `[context: NN%]` and `[cost: $X]`, and
the spinner line (`✻ Crunched for …` vs an empty `❯` prompt) shows
busy vs idle. Use `context %` for the 60% rotation trigger and the
spinner/prompt state to tell whether a teammate is mid-task before
sending a check-in. This is the same mechanism for both idle detection
and context-based rotation.

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

**Default to subagents to protect your context.** Every teammate should
offload context-heavy work to short-lived subagents (Explore for
read-only, general-purpose for work that writes) and keep only the
distilled result in its own context. The point is to avoid context rot —
your own window stays small and focused while the subagent absorbs the
bulk.

- Files > 500 lines: ALWAYS via subagent (never read directly)
- Files 100–500 lines: subagent or offset/limit
- Files < 100 lines: direct read is fine
- Assembly files (.s): ALWAYS via Explore subagent, never read directly
- CK source headers (most are 500–4000 lines): ALWAYS via Explore
  subagent with a precise extraction prompt — ask for the specific
  formula/table/struct you need, not a full dump
- When a subagent returns, save its distilled finding to your output
  file; do not paste large excerpts back into your own context
- Parallelize independent reads across multiple subagents in one batch

## Verification Enforcement

Every implementation deliverable MUST be independently verified before
being marked done. "Impl says it's done" is NOT verification.

### Mandatory verification steps (no exceptions)

1. **Lead QA gate (per kernel):** Lead spawns a fresh subagent or runs
   the test binary independently. Clean rebuild from scratch, both
   golden dirs, no filters. Every test must PASS (exit 0). If impl
   claims a fix was applied, lead MUST grep/read the actual file to
   confirm — never trust the claim alone.

2. **Spec compliance review (per phase):** Spawn independent subagent(s)
   with 1M context to read the spec/plan and verify each kernel's code
   matches. The reviewer has NO prior context — they verify from source.

3. **ISA pattern review (per phase):** Prof or independent subagent
   extracts assembly and verifies instruction types/counts match CK's
   design patterns. Golden correctness alone is insufficient — the
   implementation must use the right ISA patterns.

4. **Documentation audit (per phase):** Before committing, spawn an
   independent subagent to verify that knowledge.md, status.md, and
   team config accurately reflect the actual state of code, tests, and
   git history. Flag stale claims, wrong paths, and unimplemented fixes.

### How to verify a claimed fix

When a member reports "fix applied":
```bash
# 1. Check the actual file — don't trust the message
grep -rn "<pattern>" <file>
# 2. If the fix should have removed something, verify it's gone
grep -rn "<old_pattern>" <file>  # should return nothing
# 3. Rebuild and run tests to confirm no regression
```

### What happens when verification catches a lie

If an independent audit finds that a claimed fix was NOT applied:
1. The member is notified with the specific evidence
2. The fix is re-assigned with explicit verification grep command
3. Lead verifies the grep output before marking done
4. The incident is logged in knowledge.md as a process finding

## Constraints

See design spec for full constraints. Key limits:
- Target: gfx942 (MI-300X, CDNA3)
- Final kernel: VGPR ≤ 127, AGPR = 0, no spills, LDS = 13824
- Standalone kernels: no register pressure requirement
