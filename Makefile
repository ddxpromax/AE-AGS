.PHONY: quick paper paper-j8 paper-json paper-plot sweep status

quick:
	python -m ae_ags.run_experiment --preset quick

paper:
	./run_paper_default.sh

paper-j8:
	./run_paper_default.sh --runs 20 --jobs 8

paper-json:
	python -m ae_ags.run_experiment --preset paper_default --runs 20 --jobs 8 --record-every 1000 --save-json results/paper_run/one_run_curve.json

paper-plot:
	python -m ae_ags.plot_from_run_json --input-json results/paper_run/one_run_curve.json --output-dir results/paper_run/plots

sweep:
	./run_appendix_e.sh

status:
	git status --short --branch
