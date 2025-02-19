.PHONY: all ig eg gg cg

all: ig eg gg cg

ig:
	@echo "Running with ig_name=integrated_gradient"
	python main_ig.py --ig_name integrated_gradient

eg:
	@echo "Running with ig_name=expected_gradient"
	python main_ig.py --ig_name expected_gradient

gg:
	@echo "Running with ig_name=guided_gradient"
	python main_ig.py --ig_name guided_gradient

cg:
	@echo "Running with ig_name=contrastive_gradient"
	python main_ig.py --ig_name contrastive_gradient
