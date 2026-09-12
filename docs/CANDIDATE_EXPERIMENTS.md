# Approximate photo-to-point experiments

These opt-in collectors explore faster alternatives to the byte-identical
production candidate collector. They do not change the production CLI or the
exact collector. Their outputs are intentionally not byte-identical and should
not replace the exact path without validation on additional scans.

The frozen short indoor comparison has 6,136,485 geometry points and 62
calibrated photos. The exact optimized candidate stage has a 7.134-second
median. Timings below are single full-stage runs on the same inputs and host.

| Strategy | Candidate time | Speedup | Exact coverage retained | Top-photo agreement | RGB channel MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: 20 coverage-selected keyframes | 3.173 s | 2.25× | 98.12% | 51.80% | 6.58 / 255 |
| B: 8 cm analytic cell assignment, top 1 | 4.011 s | 1.78× | 99.80% | 82.45% | 16.68 / 255 |
| C: 8 cm proxy visibility, top 1 | 5.375 s | 1.33× | 99.80% | 83.51% | 15.91 / 255 |

Strategy A selects ten left and ten right frames by greedy weighted coverage of
20 cm normal-split surface cells. It then runs the exact projection, depth,
visibility, mask, scoring, and four-candidate logic on those frames. It is the
only experiment here that exceeds the 2× goal and it has the lowest color
drift, at the cost of losing 113,296 points covered by the exact result.

Strategy B scores all cameras analytically on 8 cm normal-split surface cells,
assigns the best camera to every dense point in a cell, and rechecks projection,
incidence, and masks on the dense points. It skips dense occlusion and retains
one candidate per point.

Strategy C performs depth/visibility/mask selection on one actual source-point
representative per 8 cm normal-split cell, transfers the best camera to the
dense points, then rechecks projection, incidence, and masks. Analytic scoring
filled 3.12% of the four-slot proxy shortlist. The rescue affected the
transferred top slot for 1.07% of cells and 0.19% of assigned dense points. It
also retains one candidate per point and skips a second dense occlusion pass.

The top-1 design in B and C leaves no cross-photo overlap for global/local
exposure solving or multi-view blending. Their higher color error is therefore
expected even though their geometric coverage is high. These variants are
useful visual experiments, not production defaults.

Run one candidate experiment with `scripts/run_candidate_experiment.py`, run
the unchanged exposure/blend/export tail with
`scripts/finalize_candidate_experiment.py`, compare it with the exact control
using `scripts/evaluate_candidate_experiments.py`, and publish the five viewer
clouds with `scripts/publish_candidate_experiment_viewer.py`.
