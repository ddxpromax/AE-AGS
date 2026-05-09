.PHONY: quick paper paper-j8 paper-json paper-plot sweep status

# Exported as env for subprocesses; same defaults as scripts/parallel_defaults.sh. Override example:
#   make paper-json PARALLEL_BLAS="OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
PARALLEL_BLAS ?= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

quick:
	python -m ae_ags.run_experiment --preset quick

paper:
	./run_paper_default.sh

paper-j8:
	$(PARALLEL_BLAS) ./run_paper_default.sh --runs 20 --jobs 8

paper-json:
	$(PARALLEL_BLAS) ./run_paper_default.sh --runs 20 --jobs 8 --record-every 1000 --save-json results/paper_run/one_run_curve.json

paper-plot:
	python -m ae_ags.plot_from_run_json --input-json results/paper_run/one_run_curve.json --output-dir results/paper_run/plots

sweep:
	./run_appendix_e.sh

status:
	git status --short --branch
