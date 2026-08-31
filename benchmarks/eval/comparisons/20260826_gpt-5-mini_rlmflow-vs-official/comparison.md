# Benchmark comparison

## Summary

| Run | Tasks | Accuracy | Mean score | Errors | Input tokens | Output tokens | Time (s) | Recursive tasks | Subcall tasks | Subcalls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rlmflow | 20 | 10.0% | 0.1083 | 0 | 216400 | 61393 | 1040.9 | 0 | 0 | 0 |
| official | 20 | 30.0% | 0.3348 | 0 | 422397 | 122331 | 1784.8 | 0 | 1 | 1 |

## Per-problem results

| Dataset | Problem | rlmflow score | rlmflow correct | rlmflow agents | rlmflow subcalls | rlmflow tokens | rlmflow time (s) | official score | official correct | official agents | official subcalls | official tokens | official time (s) | Score delta (official − rlmflow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| delegation_arc_agi | delegation_arc_agi_19_1ae2feb7 | 0.0000 | no | 1 | 0 | 10811 | 30.4 | 0.0000 | no | 1 | 0 | 181382 | 508.1 | +0.0000 |
| delegation_arc_agi | delegation_arc_agi_20_2ba387bc | 0.0000 | no | 1 | 0 | 33064 | 105.2 | 1.0000 | yes | 1 | 0 | 61394 | 129.7 | +1.0000 |
| delegation_codeqa | delegation_codeqa_17_official_codeqa_00072 | 0.0000 | no | 1 | 0 | 19136 | 36.0 | 0.0000 | no | 1 | 0 | 49301 | 103.0 | +0.0000 |
| delegation_dabstep | delegation_dabstep_11_2536 | 0.0000 | no | 1 | 0 | 49085 | 135.7 | 0.0000 | no | 1 | 0 | 6111 | 19.2 | +0.0000 |
| delegation_dabstep | delegation_dabstep_12_2769 | 0.0000 | no | 1 | 0 | 24801 | 99.8 | 0.0000 | no | 1 | 0 | 11769 | 51.3 | +0.0000 |
| delegation_entailmentbank | delegation_entailmentbank_10_Mercury_SC_416126 | 0.0000 | no | 1 | 0 | 10699 | 27.5 | 0.0000 | no | 1 | 0 | 8236 | 53.5 | +0.0000 |
| delegation_grsqa | delegation_grsqa_09_sample-comparison-1 | 0.0000 | no | 1 | 0 | 7147 | 24.2 | 0.0000 | no | 1 | 0 | 4565 | 20.3 | +0.0000 |
| delegation_musique | delegation_musique_05_4hop1__38130_8966_31714_79432 | 0.0000 | no | 1 | 0 | 13394 | 33.7 | 0.3636 | no | 1 | 0 | 19806 | 66.4 | +0.3636 |
| delegation_musique | delegation_musique_06_4hop1__38130_8966_31714_79432 | 0.0000 | no | 1 | 0 | 8848 | 36.2 | 0.0000 | no | 1 | 0 | 9032 | 48.8 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_13_trip_planning_example_593 | 0.0000 | no | 1 | 0 | 5187 | 49.4 | 0.0000 | no | 1 | 0 | 17740 | 98.0 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_14_meeting_planning_example_594 | 0.0000 | no | 1 | 0 | 5447 | 45.6 | 0.0000 | no | 1 | 0 | 18404 | 85.0 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_15_calendar_scheduling_example_976 | 0.0000 | no | 1 | 0 | 4383 | 34.5 | 0.0000 | no | 1 | 0 | 17700 | 59.9 | +0.0000 |
| delegation_parallelqa | delegation_parallelqa_11 | 0.0000 | no | 1 | 0 | 2813 | 14.9 | 0.0000 | no | 1 | 0 | 7260 | 30.3 | +0.0000 |
| delegation_parallelqa | delegation_parallelqa_22 | 1.0000 | yes | 1 | 0 | 2895 | 15.8 | 1.0000 | yes | 1 | 0 | 5032 | 27.3 | +0.0000 |
| delegation_parallelqa | delegation_parallelqa_63 | 0.0000 | no | 1 | 0 | 8650 | 34.6 | 1.0000 | yes | 1 | 0 | 7683 | 35.8 | +1.0000 |
| delegation_parallelqa | delegation_parallelqa_94 | 0.0000 | no | 1 | 0 | 9073 | 44.8 | 1.0000 | yes | 1 | 0 | 11159 | 46.8 | +1.0000 |
| delegation_planbench | delegation_planbench_16_logistics_instance_20 | 1.0000 | yes | 1 | 0 | 13178 | 44.4 | 1.0000 | yes | 1 | 1 | 30505 | 134.0 | +0.0000 |
| delegation_sudoku | delegation_sudoku_18_cross-product | 0.1667 | no | 1 | 0 | 21767 | 134.2 | 1.0000 | yes | 1 | 0 | 42264 | 128.9 | +0.8333 |
| delegation_twowiki | delegation_twowiki_07_948c33ea0baf11ebab90acde48001122 | 0.0000 | no | 1 | 0 | 14961 | 46.8 | 0.0000 | no | 1 | 0 | 15898 | 59.8 | +0.0000 |
| delegation_twowiki | delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6 | 0.0000 | no | 1 | 0 | 12454 | 47.3 | 0.3333 | no | 1 | 0 | 19487 | 78.6 | +0.3333 |
