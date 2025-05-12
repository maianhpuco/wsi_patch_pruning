.PHONY: all ig eg gg cg

SIMEA_CHECKPOINT_FOLDER=/home/mvu9/processing_datasets/camelyon16/checkpoints
CHECKPOINT_FILE=mil_checkpoint_exp002.pth
SIMEA_CONFIG_FILE=simea_ma_exp002


classifing: 
	python main.py \
		--config_file $(CONFIG_FILE) \
		--checkpoint_folder $(SIMEA_CHECKPOINT_FOLDER) \
		--checkpoint_file $(CHECKPOINT_FILE) 

predicting:
	python main_predict.py \
		--config_file $(CONFIG_FILE) \
		--checkpoint_folder $(SIMEA_CHECKPOINT_FOLDER) \
		--checkpoint_file $(CHECKPOINT_FILE) 

group1: ig eg  

group2: cg sig 

group3: vg idg sig2  

ig:
	@echo "Running with ig_name=integrated_gradient"
	python main_ig.py --ig_name integrated_gradient
	# python main_plot_ig.py --ig_name integrated_gradient

eg:
	@echo "Running with ig_name=expected_gradient"
	python main_ig.py --ig_name expected_gradient
	# python main_plot_ig.py --ig_name expected_gradient

cg:
	@echo "Running with ig_name=contrastive_gradient"
	python main_ig.py --ig_name contrastive_gradient
	# python main_plot_ig.py --ig_name contrastive_gradient

sig:
	@echo "Running with ig_name=square_integrated_gradient"
	python main_ig.py --ig_name square_integrated_gradient
	# python main_plot_ig.py --ig_name square_integrated_gradient

vg:
	@echo "Running with ig_name=vanilla_gradient"
	python main_ig.py --ig_name vanilla_gradient 
	# python main_plot_ig.py --ig_name vanilla_gradient 

idg:
	@echo "Running with ig_name=integrated_decision_gradient"
	python main_ig.py --ig_name integrated_decision_gradient
	# python main_plot_ig.py --ig_name integrated_decision_gradient
 

sig2:
	@echo "Running with ig_name=optim_square_integrated_gradient"
	python main_ig.py --ig_name optim_square_integrated_gradient
	# python main_plot_ig.py --ig_name optim_square_integrated_gradient 

dr_all : drig dreg drigd drcg drsig drsig2 

# drig:
# 	@echo "Running with ig_name=integrated_gradient"
# 	python main_plot_ig.py --ig_name integrated_gradient --dry_run 1 

# dreg:
# 	@echo "Running with ig_name=expected_gradient"
# 	python main_plot_ig.py --ig_name expected_gradient --dry_run 1  
# drigd:
# 	@echo "Running with ig_name=integrated_decision_gradient"
# 	python main_plot_ig.py --ig_name integrated_decision_gradient --dry_run 1 

# drcg:
# 	@echo "Running with ig_name=contrastive_gradient"
# 	python main_plot_ig.py --ig_name contrastive_gradient --dry_run 1 

# drsig:
# 	@echo "Running with ig_name=square_integrated_gradient"
# 	python main_plot_ig.py --ig_name square_integrated_gradient --dry_run 1 

# drvg:
# 	@echo "Running with ig_name=vanilla_gradient"
# 	python main_plot_ig.py --ig_name vanilla_gradient --dry_run 1 


drig:
	@echo "Running with ig_name=integrated_gradient"
	# python main_ig.py --ig_name integrated_gradient --dry_run 1 
	python main_plot_ig.py --ig_name integrated_gradient --dry_run 1 

dreg:
	@echo "Running with ig_name=expected_gradient"
	# python main_ig.py --ig_name expected_gradient --dry_run 1 
	python main_plot_ig.py --ig_name expected_gradient --dry_run 1  
drigd:
	@echo "Running with ig_name=integrated_decision_gradient"
	# python main_ig.py --ig_name integrated_decision_gradient --dry_run 1 
	python main_plot_ig.py --ig_name integrated_decision_gradient --dry_run 1 

drcg:
	@echo "Running with ig_name=contrastive_gradient"
	# python main_ig.py --ig_name contrastive_gradient --dry_run 1 
	python main_plot_ig.py --ig_name contrastive_gradient --dry_run 1 

drsig:
	@echo "Running with ig_name=square_integrated_gradient"
	# python main_ig.py --ig_name square_integrated_gradient --dry_run 1 
	python main_plot_ig.py --ig_name square_integrated_gradient --dry_run 1 

drvg:
	@echo "Running with ig_name=vanilla_gradient"
	python main_ig.py --ig_name vanilla_gradient --dry_run 1 
	python main_plot_ig.py --ig_name vanilla_gradient --dry_run 1 

drsig2:
	@echo "Running with ig_name=optim_square_integrated_gradient"
	# python main_ig.py --ig_name optim_square_integrated_gradient --dry_run 1 
	python main_plot_ig.py --ig_name optim_square_integrated_gradient --dry_run 1  

get_ground_truth: 
	python metrics_segmentation/main_interior_mask.py
