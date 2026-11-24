# BIRD 16-draw sampling, best-of-16, and matched-cost results

Matched-cost rule: for each trained operating point and timing repeat, take the largest frozen draw budget `k` in `{1, 2, 4, 16}` whose saturated GPU-seconds per question are at most `1.15` times the trained point's cost (or `k=1` if even frozen `k=1` exceeds `1.15` times the trained cost), and compare paired accuracy on the 1,534 BIRD dev questions. Outcomes are `training wins at matched cost` (paired 95% lower bound above `+3.00` points at matched `k`, or above `0.00` when frozen `k=1` exceeds `1.15`x trained cost), `favors frozen` (upper bound below `+3.00` points at matched `k`), `training adds on top of sampling` (same `k` in `{2, 4}`, frozen-to-trained cost ratio in `[0.85, 1.15]`, and lower bound above `+3.00` points), or `open` (otherwise, when repeats disagree on matched `k` or rule branch, or when a repeat lacks valid isolation timing).

Questions: 1,534. Paired 95% intervals: 10,000 question bootstrap resamples with seed `20261120` (linearly interpolated 2.5th and 97.5th percentiles).

## First-draw and any-of-16 counts per arm

`first draw` is draw 0 of 16; `any-of-16` counts a question correct if any of its 16 draws is correct (an oracle upper bound); `best-of-16` picks one answer per question using a fixed selector that never sees grades.

| Arm | First draw | First draw % | Any-of-16 | Any-of-16 % | Selected k=2 | Selected k=4 | Best-of-16 | Unbiased pass@1 % | Unbiased pass@16 % | First-draw tokens | vs frozen 12-turn % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen (12-turn limits) | 603 | 39.31% | 1,150 | 74.97% | 716 | 809 | 894 | 40.33% | 74.97% | 680,187 | +0.00% |
| frozen (6-turn limits) | 600 | 39.11% | 1,127 | 73.47% | 726 | 807 | 892 | 39.13% | 73.47% | 494,763 | -27.26% |
| seed 44, update 8 (12-turn limits) | 761 | 49.61% | 1,130 | 73.66% | 810 | 856 | 882 | 48.71% | 73.66% | 573,268 | -15.72% |
| seed 44, update 8 (6-turn limits) | 738 | 48.11% | 1,123 | 73.21% | 797 | 854 | 873 | 47.95% | 73.21% | 473,434 | -30.40% |
| seed 42, update 50 (12-turn limits) | 926 | 60.37% | 1,176 | 76.66% | 963 | 979 | 1,011 | 60.91% | 76.66% | 1,141,945 | +67.89% |
| seed 42, update 50 (6-turn limits) | 829 | 54.04% | 1,133 | 73.86% | 901 | 944 | 988 | 55.04% | 73.86% | 939,630 | +38.14% |
| seed 43, update 50 (12-turn limits) | 941 | 61.34% | 1,201 | 78.29% | 969 | 1,003 | 1,020 | 60.39% | 78.29% | 1,112,756 | +63.60% |
| seed 43, update 50 (6-turn limits) | 866 | 56.45% | 1,164 | 75.88% | 898 | 949 | 997 | 57.09% | 75.88% | 788,181 | +15.88% |

## Paired first-draw and any-of-16 contrasts

| Contrast (A minus B) | Mode | A correct | B correct | A only | B only | Difference (pp) | 95% interval (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| seed 42, update 50 (12-turn limits) vs frozen (12-turn limits) | first draw | 926 | 603 | 379 | 56 | +21.06 | +18.64 to +23.53 |
| seed 43, update 50 (12-turn limits) vs frozen (12-turn limits) | first draw | 941 | 603 | 393 | 55 | +22.03 | +19.62 to +24.58 |
| seed 43, update 50 (12-turn limits) vs seed 42, update 50 (12-turn limits) | first draw | 941 | 926 | 119 | 104 | +0.98 | -0.91 to +2.93 |
| seed 44, update 8 (12-turn limits) vs frozen (12-turn limits) | first draw | 761 | 603 | 273 | 115 | +10.30 | +7.89 to +12.78 |
| seed 44, update 8 (6-turn limits) vs frozen (6-turn limits) | first draw | 738 | 600 | 239 | 101 | +9.00 | +6.71 to +11.28 |
| seed 42, update 50 (6-turn limits) vs frozen (6-turn limits) | first draw | 829 | 600 | 314 | 85 | +14.93 | +12.45 to +17.41 |
| seed 43, update 50 (6-turn limits) vs frozen (6-turn limits) | first draw | 866 | 600 | 340 | 74 | +17.34 | +14.86 to +19.75 |
| seed 43, update 50 (6-turn limits) vs seed 42, update 50 (6-turn limits) | first draw | 866 | 829 | 149 | 112 | +2.41 | +0.39 to +4.43 |
| frozen (12-turn limits) vs frozen (6-turn limits) | first draw | 603 | 600 | 190 | 187 | +0.20 | -2.28 to +2.61 |
| seed 42, update 50 (12-turn limits) vs frozen (12-turn limits) | any-of-16 | 1,176 | 1,150 | 93 | 67 | +1.69 | +0.13 to +3.32 |
| seed 42, update 50 (12-turn limits) vs frozen (6-turn limits) | any-of-16 | 1,176 | 1,127 | 108 | 59 | +3.19 | +1.56 to +4.82 |
| seed 43, update 50 (12-turn limits) vs frozen (12-turn limits) | any-of-16 | 1,201 | 1,150 | 90 | 39 | +3.32 | +1.89 to +4.76 |
| seed 43, update 50 (12-turn limits) vs frozen (6-turn limits) | any-of-16 | 1,201 | 1,127 | 111 | 37 | +4.82 | +3.26 to +6.32 |
| seed 43, update 50 (12-turn limits) vs seed 42, update 50 (12-turn limits) | any-of-16 | 1,201 | 1,176 | 67 | 42 | +1.63 | +0.26 to +3.00 |
| seed 43, update 50 (6-turn limits) vs seed 42, update 50 (6-turn limits) | any-of-16 | 1,164 | 1,133 | 77 | 46 | +2.02 | +0.65 to +3.46 |

## Frozen best-of-16 and adapter first-draw contrasts

- Frozen (`12-turn limits`) best-of-16: **894** / 1,534 (58.28%)
- Frozen (`6-turn limits`) best-of-16: **892** / 1,534 (58.15%)

| Adapter first draw vs frozen best-of-16 | Adapter correct | Frozen best-of-16 | Adapter only | Frozen only | Difference (pp) | 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| seed 42, update 50 (12-turn limits) vs frozen (12-turn limits) | 926 | 894 | 130 | 98 | +2.09 | +0.13 to +3.98 |
| seed 43, update 50 (12-turn limits) vs frozen (12-turn limits) | 941 | 894 | 135 | 88 | +3.06 | +1.17 to +4.95 |
| seed 44, update 8 (12-turn limits) vs frozen (12-turn limits) | 761 | 894 | 64 | 197 | -8.67 | -10.63 to -6.65 |
| seed 42, update 50 (6-turn limits) vs frozen (6-turn limits) | 829 | 892 | 93 | 156 | -4.11 | -6.13 to -2.09 |
| seed 43, update 50 (6-turn limits) vs frozen (6-turn limits) | 866 | 892 | 115 | 141 | -1.69 | -3.72 to +0.33 |
| seed 44, update 8 (6-turn limits) vs frozen (6-turn limits) | 738 | 892 | 60 | 214 | -10.04 | -12.06 to -8.02 |

## GPU timing saturation check

| Cohort questions | 2 draws (s/draw) | 4 draws (s/draw) | 8 draws (s/draw) | Passed +/-10% vs 4 draws |
| ---: | ---: | ---: | ---: | --- |
| 256 | 0.4609 | 0.3691 | 0.3442 | no |
| 512 | 0.3818 | 0.3550 | 0.3379 | yes |

## Matched-cost results: single-seed session (seeds 44 and 42)

Slot timings (512 questions x 4 draws = 2,048 attempts per slot; isolated reruns used for repeat 2 where present):

| Arm | Repeat 1 (s) | Repeat 2 (s) | Repeat 3 (s) | Mean s/draw |
| --- | ---: | ---: | ---: | ---: |
| frozen (6-turn limits) | 694 | 719 | 715 | 0.3464 |
| frozen (12-turn limits) | 864 | 893 | 945 | 0.4398 |
| seed 44, update 8 (12-turn limits) | 758 | 782 | 797 | 0.3804 |
| seed 42, update 50 (12-turn limits) | 1,585 | 1,666 | 1,671 | 0.8011 |
| seed 44, update 8 (6-turn limits) | 615 | 597 | 617 | 0.2977 |
| seed 42, update 50 (6-turn limits) | 1,300 | 1,255 | 1,268 | 0.6222 |

Repeat-2 isolated reruns vs initial repeat 2: `frozen (12-turn limits)` 893 s (initial 904 s); `seed 42, update 50 (6-turn limits)` 1,255 s (initial 1,280 s), clearing the frozen k=2 cutoff (`2 * 719 / 1.15 = 1250.4348 s`) by 4.5652 s (initial 29.5652 s).

| Trained point | Comparator | Matched frozen k (rep 1, 2, 3) | Cost ratio frozen/trained (rep 1, 2, 3) | Difference (pp) | 95% interval (pp) | Outcome | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| seed 44, update 8 (12-turn limits), k=1 | frozen (6-turn limits) | 1, 1, 1 | 0.916, 0.919, 0.897 | +10.50 | +8.15 to +12.84 | training wins at matched cost |  |
| seed 44, update 8 (12-turn limits), k=2 | frozen (6-turn limits) | 2, 2, 2 | 0.916, 0.919, 0.897 | +5.48 | +3.19 to +7.76 | training wins at matched cost |  |
| seed 42, update 50 (12-turn limits), k=1 | frozen (12-turn limits) | 2, 2, 2 | 1.090, 1.072, 1.131 | +13.69 | +11.41 to +15.97 | training wins at matched cost |  |
| seed 42, update 50 (12-turn limits), k=1 | frozen (6-turn limits) | 2, 2, 2 | 0.876, 0.863, 0.856 | +13.04 | +10.63 to +15.32 | training wins at matched cost |  |
| seed 42, update 50 (6-turn limits), k=1 | frozen (6-turn limits) | 2, 2, 2 | 1.068, 1.146, 1.128 | +6.71 | +4.37 to +8.93 | training wins at matched cost |  |
| seed 44, update 8 (6-turn limits), k=1 | frozen (6-turn limits) | 1, 1, 1 | 1.128, 1.204, 1.159 | +9.00 | +6.71 to +11.28 | open | repeats disagree on the 1.15x cost-cutoff branch (repeat 1 ordinary matched-cost, repeats 2 and 3 cheaper than frozen k=1); all three repeats favor training |
| seed 44, update 8 (6-turn limits), k=2 | frozen (6-turn limits) | 2, 1, 1 | 1.128, 0.602, 0.579 | varies by repeat | varies by repeat | open | repeats disagree on matched frozen k ([2, 1, 1]) |
| seed 44, update 8 (12-turn limits), same k=2 | frozen (6-turn limits), k=2 | 2, 2, 2 | 0.916, 0.919, 0.897 (mean 0.911) | +5.48 | +3.19 to +7.76 | training adds on top of sampling | comparable cost band [0.85, 1.15] |

## Matched-cost results: two-seed same-session comparison (seeds 42 and 43)

Same-session slot timings (512 questions x 4 draws = 2,048 attempts per slot):

| Arm | Repeat 1 (s; s/draw) | Repeat 2 (s; s/draw) | Repeat 3 (s; s/draw) | Valid mean s/draw | Status |
| --- | --- | --- | --- | ---: | --- |
| frozen (12-turn limits) | 872 (0.4258) | 859 (0.4194) | 890 (0.4346) | 0.4266 | all 3 repeats valid |
| frozen (6-turn limits) | 698 (0.3408) (unresolved) | 698 (0.3408) | 716 (0.3496) | 0.3452 | repeat 1 unresolved; mean over repeats 2, 3 |
| seed 42, update 50 (12-turn limits) | 1,629 (0.7954) | 1,580 (0.7715) | 1,593 (0.7778) | 0.7816 | all 3 repeats valid |
| seed 43, update 50 (12-turn limits) | 1,372 (0.6699) | 1,398 (0.6826) | 1,380 (0.6738) | 0.6755 | all 3 repeats valid |
| seed 43, update 50 (6-turn limits) | 1,044 (0.5098) (unresolved) | 1,044 (0.5098) | 1,027 (0.5015) | 0.5056 | repeat 1 unresolved; mean over repeats 2, 3 |

Same-session `seed 43, update 50 (12-turn limits)` / `seed 42, update 50 (12-turn limits)` cost ratio by repeat: 0.842, 0.885, 0.866 (mean 0.864).

| Trained point | Comparator | Matched frozen k (rep 1, 2, 3) | Cost ratio frozen/trained (valid repeats) | Difference on valid repeats (pp) | 95% interval (pp) | Outcome | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| seed 43, update 50 (12-turn limits), k=1 | frozen (12-turn limits) | 1, 1, 1 | 0.636, 0.614, 0.645 | +22.03 | +19.62 to +24.58 | training wins at matched cost |  |
| seed 43, update 50 (12-turn limits), k=1 | frozen (6-turn limits) | unresolved, 2, 2 | 0.999, 1.038 | +14.02 | +11.73 to +16.30 | open | repeat 1 lacked valid isolation timing; valid repeats (2, 3) give training wins at matched cost at frozen k=2 |
| seed 43, update 50 (6-turn limits), k=1 | frozen (6-turn limits) | unresolved, 1, 1 | 0.669, 0.697 | +17.34 | +14.86 to +19.75 | open | repeat 1 lacked valid isolation timing; valid repeats (2, 3) give training wins at matched cost at frozen k=1 |

## Two-seed replication summary

Decisive checks replicated on valid evidence: **4 of 5**. Assigned two-seed label: **the update-50 result is seed-dependent**.

| Decisive comparison | Criterion | Replicated | Result |
| --- | --- | --- | --- |
| seed 43, update 50 (12-turn limits) first draw vs frozen (12-turn limits) | 95% lower bound > +3.00 points | yes | 941 vs 603 (+22.03, +19.62 to +24.58) |
| seed 43, update 50 (12-turn limits) matched cost vs frozen (12-turn limits) | training wins at matched cost across all 3 repeats | yes | training wins at matched cost (matched frozen k=1 in repeats 1, 2, 3) |
| seed 43, update 50 (12-turn limits) matched cost vs frozen (6-turn limits) | training wins at matched cost across all 3 repeats | no | open (repeat 1 lacked valid isolation timing; valid repeats (2, 3) give training wins at matched cost at frozen k=2) |
| seed 43, update 50 (12-turn limits) any-of-16 vs frozen (12-turn limits) | 95% lower bound > 0.00 points | yes | 1,201 vs 1,150 (+3.32, +1.89 to +4.76) |
| seed 43, update 50 (12-turn limits) any-of-16 vs frozen (6-turn limits) | 95% lower bound > 0.00 points | yes | 1,201 vs 1,127 (+4.82, +3.26 to +6.32) |
