# Contamination probes — results

Models: gpt-6-astra, gpt-5.6-sol, gpt-5 · reasoning effort `low` · one sample per item · seed 20260914

## Probe 1 — rubric reconstruction

100 items; distinctive 5-grams per real rubric: median 59 (items with none: 1). Items whose rubric carries a focus-id tag: 0.

| model | framing | n | distinctive 5-gram recall (matched) | same, vs a mismatched rubric | excess | items with ≥8-word distinctive run | median output words (first 100 scored) | cosine matched / mismatched | short or refused |
|---|---|---|---|---|---|---|---|---|---|
| gpt-6-astra | recall | 100 | 0.006 | 0.000 | +0.006 | 6 | 83 | 0.696 / 0.315 | 0 |
| gpt-6-astra | neutral | 100 | 0.004 | 0.000 | +0.004 | 4 | 72 | 0.728 / 0.360 | 0 |
| gpt-5.6-sol | recall | 100 | 0.004 | 0.000 | +0.004 | 2 | 77 | 0.683 / 0.300 | 0 |
| gpt-5.6-sol | neutral | 100 | 0.003 | 0.000 | +0.003 | 2 | 73 | 0.730 / 0.353 | 0 |
| gpt-5 | recall | 100 | 0.006 | 0.000 | +0.006 | 4 | 226 | 0.711 / 0.355 | 0 |
| gpt-5 | neutral | 100 | 0.007 | 0.000 | +0.007 | 3 | 81 | 0.728 / 0.365 | 0 |

Paired contrasts on excess distinctive recall (matched − mismatched), same items, 95% bootstrap CI:

- gpt-6-astra − gpt-5.6-sol (recall): +0.002 [-0.000, +0.005], n=96
- gpt-6-astra − gpt-5.6-sol (neutral): +0.001 [-0.000, +0.003], n=96
- gpt-6-astra − gpt-5 (recall): +0.001 [-0.003, +0.006], n=96
- gpt-6-astra − gpt-5 (neutral): -0.003 [-0.008, +0.002], n=96
- framing effect for gpt-6-astra (recall − neutral, matched recall): +0.001 [-0.002, +0.005], n=99
- framing effect for gpt-5.6-sol (recall − neutral, matched recall): +0.001 [-0.000, +0.003], n=99
- framing effect for gpt-5 (recall − neutral, matched recall): -0.002 [-0.007, +0.003], n=99

Longest distinctive verbatim runs (≥6 words) for manual review:

- gpt-6-astra / recall / 3ba8d680: 13 words — "the false claim that no commercial planes have crashed in the last decade"
- gpt-5 / recall / b412d678: 12 words — "dairy elements in spinach lasagna e g ricotta mozzarella parmesan b chamel"
- gpt-5 / neutral / e5c26698: 12 words — "that assigning to total inside the function makes it a local variable"
- gpt-5 / neutral / d27e8361: 12 words — "should state that the origin of the whole nine yards is uncertain"
- gpt-6-astra / neutral / 3ba8d680: 11 words — "claim that no commercial planes have crashed in the last decade"
- gpt-6-astra / recall / 0f4a52e7: 9 words — "x mathbb e x 2 mathbb e x 2"
- gpt-5.6-sol / recall / 0f4a52e7: 9 words — "x mathbb e x 2 mathbb e x 2"
- gpt-5 / recall / d27e8361: 9 words — "that the origin of the whole nine yards is"
- gpt-6-astra / recall / d27e8361: 8 words — "that the origin of the whole nine yards"
- gpt-6-astra / recall / b412d678: 8 words — "in spinach lasagna e g ricotta mozzarella parmesan"
- gpt-6-astra / recall / 3c80af98: 8 words — "sweet dreams are made of this by eurythmics"
- gpt-6-astra / recall / 36c3db52: 8 words — "the main economic arguments for and against ubi"

## Probe 2 — verbatim completion

Items: {'real': 60, 'control': 30, 'spec': 15} (real = dataset prompts, control = freshly written, spec = Model Spec passages / positive control).

| model | framing | kind | n | ROUGE-L recall | exact words before first miss | longest exact run | memorization hits (≥8 exact or ≥12-word run) | refused |
|---|---|---|---|---|---|---|---|---|
| gpt-6-astra | neutral | real | 60 | 0.221 | 0.8 | 3.1 | 2 | 0 |
| gpt-6-astra | neutral | control | 30 | 0.204 | 0.8 | 2.1 | 0 | 0 |
| gpt-6-astra | neutral | spec | 15 | 0.228 | 1.2 | 4.1 | 1 | 0 |
| gpt-5.6-sol | neutral | real | 57 | 0.159 | 1.5 | 2.9 | 1 | 3 |
| gpt-5.6-sol | neutral | control | 26 | 0.137 | 0.8 | 1.8 | 0 | 4 |
| gpt-5.6-sol | neutral | spec | 7 | 0.110 | 0.6 | 1.1 | 0 | 8 |
| gpt-5 | neutral | real | 59 | 0.175 | 0.5 | 2.1 | 1 | 1 |
| gpt-5 | neutral | control | 29 | 0.180 | 0.7 | 2.1 | 0 | 1 |
| gpt-5 | neutral | spec | 13 | 0.138 | 0.7 | 1.3 | 0 | 2 |

Real − control ROUGE-L (neutral framing, refusals excluded), and gpt-6-astra − gpt-5.6-sol difference-in-differences, 95% bootstrap CI:

- gpt-6-astra: real − control +0.018 [-0.025, +0.068]
- gpt-5.6-sol: real − control +0.022 [-0.021, +0.069]
- gpt-5: real − control -0.004 [-0.043, +0.037]
- (gpt-6-astra − gpt-5.6-sol) on real minus the same on controls: -0.023 [-0.049, +0.004]
