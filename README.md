# Contrary to Reason

**Sixteen boolean functions, and two ways to fail at them.**

There are exactly sixteen two-input boolean functions. This notebook trains one small
network per function and uses that complete, tiny problem space to ask a larger
question: when a network fails to learn something, how do you tell whether the
*architecture* was incapable or the *training* merely missed?

**[Read the rendered notebook on nbviewer →](https://nbviewer.org/github/SidneyHarrell/contrary-to-reason/blob/main/Boolean_16_Truth_Tables_PyTorch_GPU.ipynb)**

nbviewer renders the saved tables, figures and animations without running anything.
GitHub's own viewer may fall back to "view raw" here, since the notebook is ~3 MB.

## Results

- **A dead ReLU sits on an exactly flat plateau.** Measured across the entire dead
  region of a loss slice, the loss varies by `0.00e+00` — not small, zero. There is no
  gradient to escape on, which is why extra epochs never rescued those runs.

- **Narrowing the hidden layer to 4, 2 and 1 units drops accuracy to 52/64, but only 2
  of the 12 failures are capacity limits.** Nine of them vanish under a different
  random seed. The dominant failure is dying ReLU units, not insufficient capacity —
  which is the opposite of what the experiment appears to be testing.

- **Swapping ReLU for `tanh` removes every dead layer (9 → 0) and lifts accuracy to
  61/64,** while leaving the two provable capacity limits exactly where they were. That
  is what the impossibility proof requires, since the proof never mentions the
  activation.

- **`XOR` and `XNOR` at width 1 are impossible, and that is proved rather than
  observed.** A single hidden unit makes the network a monotone function of one linear
  projection, and both diagonals of the truth table have equal sums, so no such
  function separates them.

- **The width-2 `XOR` circuit reads straight off the weights.** Both hidden units point
  in the *same* direction — cosine similarity 1.000 — and differ only in bias. The
  second unit buys a second kink along one axis, not a second dimension, which is
  precisely what one kink provably cannot do.

## Structure

| Part | Question | Answer |
|:--|:--|:--|
| 1 | Can a small network learn all 16 functions? | Yes — 16/16, 64/64 bits |
| 2 | What breaks as the hidden layer narrows? | Mostly dead units, not capacity |
| 3 | What does "dead" look like on the loss surface? | An exactly flat plateau |
| 4 | Can the failed runs be fixed? | Yes, all but the two proved limits |
| 5 | What do the solved networks compute? | Circuits you can read off the weights |

## Method notes

Where a claim can be proved, it is proved and then asserted in code, so the assertions
check the theory rather than the observed outcome. Failure modes land on loss values
computable in closed form — the entropy of the label distribution for a dead network,
`(3·ln3 − 2·ln2)/4 ≈ 0.477386` for a monotone width-1 model on `XOR` — and the runs
match those to six decimal places. Where a claim is only observed, it is labelled as
observed.

The problem space is small enough that several claims are settled exhaustively rather
than sampled: all 16 functions, all 4 input rows, and all 256 subsets of an 8-unit
hidden layer when finding the smallest circuit that still computes each function.

## Running it

The notebook ships with every output saved, so it reads without being run.

To re-run it: PyTorch 2.3.0 with CUDA 12.1, plus numpy, pandas and matplotlib. It runs
unchanged on CPU, more slowly. Cells are strictly sequential — later parts reuse models
trained earlier — so run it top to bottom.

Expect roughly **35–40 minutes on a consumer GPU**, dominated by two sweeps of 16
models × 4 widths and the hyperparameter grid in Part 4. At this size the GPU is bound
by kernel-launch latency rather than arithmetic: about 0.75 ms per step regardless of
hidden width, since a 33-parameter model does not meaningfully load an RTX 4060.

## License

MIT — see [LICENSE](LICENSE).

---

Sidney Harrell · Senior Sysadmin, Support Dept. ·
[valiantysfederal.com](https://www.valiantysfederal.com)
With help from Claude Opus 5.
