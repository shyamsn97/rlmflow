SCALE ?= 1.0

baseline_run := ../../_runs/word-search/baseline

render_baseline:
	rlmflow render frames ./$(baseline_run) \
	./$(baseline_run)/frames/baseline \
	--scale $(SCALE)

render_all: render_baseline
