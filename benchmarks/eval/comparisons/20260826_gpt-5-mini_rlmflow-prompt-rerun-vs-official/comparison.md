# Benchmark comparison

## Summary

| Run | Tasks | Accuracy | Mean score | Errors | Input tokens | Output tokens | Time (s) | Recursive tasks | Subcall tasks | Subcalls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rlmflow | 20 | 0.0% | 0.0139 | 0 | 636875 | 364185 | 2490.1 | 17 | 0 | 0 |
| official | 20 | 30.0% | 0.3348 | 0 | 422397 | 122331 | 1784.8 | 0 | 1 | 1 |

## Per-problem results

| Dataset | Problem | rlmflow score | rlmflow correct | rlmflow agents | rlmflow subcalls | rlmflow tokens | rlmflow time (s) | official score | official correct | official agents | official subcalls | official tokens | official time (s) | Score delta (official − rlmflow) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| delegation_arc_agi | delegation_arc_agi_19_1ae2feb7 | 0.0000 | no | 7 | 0 | 110696 | 230.6 | 0.0000 | no | 1 | 0 | 181382 | 508.1 | +0.0000 |
| delegation_arc_agi | delegation_arc_agi_20_2ba387bc | 0.0000 | no | 6 | 0 | 70804 | 157.3 | 1.0000 | yes | 1 | 0 | 61394 | 129.7 | +1.0000 |
| delegation_codeqa | delegation_codeqa_17_official_codeqa_00072 | 0.0000 | no | 4 | 0 | 19895 | 87.3 | 0.0000 | no | 1 | 0 | 49301 | 103.0 | +0.0000 |
| delegation_dabstep | delegation_dabstep_11_2536 | 0.0000 | no | 8 | 0 | 121333 | 208.9 | 0.0000 | no | 1 | 0 | 6111 | 19.2 | +0.0000 |
| delegation_dabstep | delegation_dabstep_12_2769 | 0.0000 | no | 9 | 0 | 43880 | 81.3 | 0.0000 | no | 1 | 0 | 11769 | 51.3 | +0.0000 |
| delegation_entailmentbank | delegation_entailmentbank_10_Mercury_SC_416126 | 0.0000 | no | 1 | 0 | 11208 | 29.0 | 0.0000 | no | 1 | 0 | 8236 | 53.5 | +0.0000 |
| delegation_grsqa | delegation_grsqa_09_sample-comparison-1 | 0.0000 | no | 4 | 0 | 25791 | 121.5 | 0.0000 | no | 1 | 0 | 4565 | 20.3 | +0.0000 |
| delegation_musique | delegation_musique_05_4hop1__38130_8966_31714_79432 | 0.0000 | no | 2 | 0 | 23738 | 88.2 | 0.3636 | no | 1 | 0 | 19806 | 66.4 | +0.3636 |
| delegation_musique | delegation_musique_06_4hop1__38130_8966_31714_79432 | 0.0000 | no | 7 | 0 | 39676 | 80.7 | 0.0000 | no | 1 | 0 | 9032 | 48.8 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_13_trip_planning_example_593 | 0.0000 | no | 8 | 0 | 108940 | 238.2 | 0.0000 | no | 1 | 0 | 17740 | 98.0 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_14_meeting_planning_example_594 | 0.0000 | no | 4 | 0 | 40484 | 129.7 | 0.0000 | no | 1 | 0 | 18404 | 85.0 | +0.0000 |
| delegation_natural_plan | delegation_natural_plan_15_calendar_scheduling_example_976 | 0.0000 | no | 8 | 0 | 52828 | 160.6 | 0.0000 | no | 1 | 0 | 17700 | 59.9 | +0.0000 |
| delegation_parallelqa | delegation_parallelqa_11 | 0.0000 | no | 3 | 0 | 9884 | 37.0 | 0.0000 | no | 1 | 0 | 7260 | 30.3 | +0.0000 |
| delegation_parallelqa | delegation_parallelqa_22 | 0.0000 | no | 1 | 0 | 4145 | 22.6 | 1.0000 | yes | 1 | 0 | 5032 | 27.3 | +1.0000 |
| delegation_parallelqa | delegation_parallelqa_63 | 0.0000 | no | 1 | 0 | 20614 | 65.0 | 1.0000 | yes | 1 | 0 | 7683 | 35.8 | +1.0000 |
| delegation_parallelqa | delegation_parallelqa_94 | 0.0000 | no | 5 | 0 | 119897 | 112.5 | 1.0000 | yes | 1 | 0 | 11159 | 46.8 | +1.0000 |
| delegation_planbench | delegation_planbench_16_logistics_instance_20 | 0.0000 | no | 5 | 0 | 24684 | 105.4 | 1.0000 | yes | 1 | 1 | 30505 | 134.0 | +1.0000 |
| delegation_sudoku | delegation_sudoku_18_cross-product | 0.2778 | no | 5 | 0 | 70966 | 263.1 | 1.0000 | yes | 1 | 0 | 42264 | 128.9 | +0.7222 |
| delegation_twowiki | delegation_twowiki_07_948c33ea0baf11ebab90acde48001122 | 0.0000 | no | 6 | 0 | 37876 | 129.4 | 0.0000 | no | 1 | 0 | 15898 | 59.8 | +0.0000 |
| delegation_twowiki | delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6 | 0.0000 | no | 6 | 0 | 43721 | 141.8 | 0.3333 | no | 1 | 0 | 19487 | 78.6 | +0.3333 |
