.PHONY: all ig eg gg cg

all: ig eg gg cg

ig:
	@echo "Running with ig_name=integrated_gradients"
	python main_ig.py --ig_name integrated_gradients

eg:
	@echo "Running with ig_name=expected_gradients"
	python main_ig.py --ig_name expected_gradients

gg:
	@echo "Running with ig_name=guided_gradients"
	python main_ig.py --ig_name guided_gradients

cg:
	@echo "Running with ig_name=contrastive_gradient"
	python main_ig.py --ig_name contrastive_gradient
