.PHONY: quick paper paper-j8 paper-json paper-plot paper-figure1 paper-json-rectified paper-figure1-rectified sweep status

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
	mkdir -p results/paper_run/fig1_knee15k
	$(PARALLEL_BLAS) ./run_paper_default.sh --runs 20 --jobs 8 --record-every 1000 --save-json results/paper_run/fig1_knee15k/one_run_curve.json

paper-plot:
	python -m ae_ags.plot_from_run_json --input-json results/paper_run/fig1_knee15k/one_run_curve.json --output-dir results/paper_run/fig1_knee15k/plots

paper-figure1:
	python -m ae_ags.plot_from_run_json --input-json results/paper_run/fig1_knee15k/one_run_curve.json --output-dir results/paper_run/fig1_knee15k/plots --paper-figure1

# Nonnegative cumulative regret (panels (a)–(e)): same knee15k hyperparameters, rectify_regret only.
paper-json-rectified:
	mkdir -p results/paper_run/fig1_knee15k/plots
	$(PARALLEL_BLAS) python -m ae_ags.run_experiment --preset paper_fig1_knee15k_rectified --config configs/paper_fig1_knee15k_rectified.json --runs 20 --jobs 8 --record-every 1000 --save-json results/paper_run/fig1_knee15k/one_run_curve_rectified.json

paper-figure1-rectified:
	python -m ae_ags.paper_figure1 --input-json results/paper_run/fig1_knee15k/one_run_curve_rectified.json --output results/paper_run/fig1_knee15k/plots/figure1_sixpanels_rectified.png

sweep:
	./run_appendix_e.sh

status:
	git status --short --branch
