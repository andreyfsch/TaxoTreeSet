# Working agreement

## Never end a turn with a plan

If a message says "next I will do X", X must already be launched **in that same
turn**. A turn that ends with an intention ends the work: nothing runs until Andrey
replies, and he should not have to be the scheduler.

Ending a turn is allowed only when:

- a job is running **and** a completion watcher is registered, or
- a genuine fork needs his decision — one where different answers change what gets
  built, not "may I proceed?"

Do not ask for confirmation to continue an investigation he already authorised.

## Launch discipline

**Verify the premise before the run, not after.** Ask: what is the cheapest thing
that would show this experiment is worthless? Two cases from this project:
`belonging_margin` was swept for ten GPU-hours before anyone read where the variable
is used — it only charges a score the arbitration never reads. And a `--limit 200`
smoke test scored 0.835 where the full set scores 0.096, because `--limit` takes the
head of the file.

**Extremes before curves.** If the endpoints agree, nothing between them differs.
Two values on 400 reads, then a curve only if it moved.

**Random samples, never prefixes.** `--sample-n`, not `--limit`. A 100-read random
sample reproduced the full set to 0.004.

## Completion notification — the exact recipe

Every experiment gets a watcher that fires when it finishes and **carries the
result**. A notification saying only "done" wastes a round trip.

### What does not work, and why

**`pgrep -f <script name>`.** It matches the watcher's own command line, because
that command line contains the script name. Used three times here; once a watcher
waited on itself for eight hours while the job had finished in twelve minutes, and
the result was only found because Andrey asked how long was left.

**`setsid nohup ... &` with no watcher.** Runs fine and notifies nobody. The turn
ends, the job finishes silently, and the investigation stalls until he asks.

**A foreground call with a long `sleep`.** It hits the 600 s tool timeout, gets
moved to the background automatically, and what comes back is the sleep's output
rather than the experiment's.

**Polling a diagnostics file for progress.** Those are written at the END. So is
the per-head JSON. Neither shows progress.

**Reading a progress counter too early.** `log_every=250` prints nothing before
read 250, so a rate computed at t+254 s divides by zero reads.

### What works

The job writes a sentinel; the watcher waits on the **file** and prints the answer:

```bash
# launch — sentinel written by the job itself, after the work
LOG=experiments/logs/<name>.log; rm -f "$LOG.done"
setsid nohup bash -c "<command> > $LOG 2>&1; echo done > $LOG.done" </dev/null &
```

```bash
# watcher — separate Bash call with run_in_background: true
until [ -f experiments/logs/<name>.log.done ]; do sleep 90; done
echo "=== <name> ==="
grep -A8 "^=== PhyloCascadeGLM ===" "$LOG"      # the RESULT, not just "finished"
```

A watcher may be registered after the job is already running — it only waits on the
file. Chaining is the same trick: `until [ -f A.done ]; do sleep 60; done && <run B>`
launches B the moment A lands, with no turn in between.

Sleep 60–120 s in the loop. Shorter adds nothing; the notification is what matters,
not the polling rate.

## Measurement discipline

**A metric is not established until it predicts ground truth.** Seven head-quality
metrics were announced and then collapsed here: val f1, sibling margin (contaminated
by backup directories in the same parent), FA_far as a max over 8 samples, FA_far
mixing two lineage bands, compositional separability, clade genome count, and an
acceptance filter that turned out to be reading taxonomic breadth — the heads it
flagged were Riboviria and Orthornavirae, which *should* accept nearly everything.
The end-to-end diagnostics are ground truth. Check against them before ranking
anything.

**Report the statistic that answers the question.** A max over 8 samples called 39
heads useless where the median called 10. A saturating measure hides a 70% → 4%
improvement.

**Pre-register the reading.** Write what each outcome will mean before the numbers
exist, in the script header. After the fact there is no way to tell analysis from
rationalisation.

## Scoring

`correct` in the harness is exact match. The project's measure is **hierarchical F
with beta = 0.5**. They disagree sharply: commits at a correct ancestor score zero
on the first and partial credit on the second, and three heads here land on a
correct ancestor 93–97% of the time. Report F(0.5) and say which is which.

## Machine

The GPU is a 4 GiB card that also draws the desktop. Keep inference near ~1 GiB and
it stays usable; at 3.9 GiB the machine freezes for seconds at a time. `polite.sh`
caps VRAM and yields between batches for when he is working.

`/mnt/f` is a **mechanical** drive behind 9p. Sequential reads are fine, random
small reads are not — the harness's adapter swaps cost 111 ms there against 53 ms on
NVMe. Bundles and adapters belong in `~/.cache`; the 278 GB of datasets can stay.

## Repos

TaxoTreeSet and CascadeHeadFactory are on `master`; PhyloCascadeGLM is on `main`.
Commit and push whenever something significant lands. The living record is
`audit/HEAD_TRAINING_INVESTIGATION.md` — refuted hypotheses stay in it, with their
numbers, because a hypothesis that quietly disappears gets retried.
