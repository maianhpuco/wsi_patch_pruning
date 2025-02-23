.PHONY: all ig eg gg cg

all: ig eg gg cg

ig:
	@echo "Running with ig_name=integrated_gradient"
	python main_ig.py --ig_name integrated_gradient

eg:
	@echo "Running with ig_name=expected_gradient"
	python main_ig.py --ig_name expected_gradient

igd:
	@echo "Running with ig_name=integrated_decision_gradient"
	python main_ig.py --ig_name integrated_decision_gradient

cg:
	@echo "Running with ig_name=contrastive_gradient"
	python main_ig.py --ig_name contrastive_gradient

sig:
	@echo "Running with ig_name=squareintegrated_gradient"
	python main_ig.py --ig_name squareintegrated_gradient

vg:
	@echo "Running with ig_name=vanilla_gradient"
	python main_ig.py --ig_name vanilla_gradient 

alldr: drig dreg drigd drcg drsig 

drig:
	@echo "Running with ig_name=integrated_gradient"
	python main_ig.py --ig_name integrated_gradient --dry_run 1 
	python main_plot_ig.py --ig_name integrated_gradient --dry_run 1 

dreg:
	@echo "Running with ig_name=expected_gradient"
	python main_ig.py --ig_name expected_gradient --dry_run 1 
	python main_plot_ig.py --ig_name expected_gradient --dry_run 1  
drigd:
	@echo "Running with ig_name=integrated_decision_gradient"
	python main_ig.py --ig_name integrated_decision_gradient --dry_run 1 
	python main_plot_ig.py --ig_name integrated_decision_gradient --dry_run 1 

drcg:
	@echo "Running with ig_name=contrastive_gradient"
	python main_ig.py --ig_name contrastive_gradient --dry_run 1 
	python main_plot_ig.py --ig_name contrastive_gradient --dry_run 1 

drsig:
	@echo "Running with ig_name=squareintegrated_gradient"
	python main_ig.py --ig_name squareintegrated_gradient --dry_run 1 
	python main_plot_ig.py --ig_name squareintegrated_gradient --dry_run 1 

drvg:
	@echo "Running with ig_name=vanilla_gradient"
	python main_ig.py --ig_name vanilla_gradient --dry_run 1 
	python main_plot_ig.py --ig_name vanilla_gradient --dry_run 1 


get_ground_truth: 
	python metrics_segmentation/main_interior_mask.py
	python metrics_segmentation/main_interior_mask.py