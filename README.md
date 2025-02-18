
# TEST BED FOR PATCH MERGING: 

## About test bed:
#### Todo before running main file for test bed: 
1. activate conda/pip env and then:  
```export PROJECT_DIR=$(pwd)```  
2. Fix your path in ```testbed_config/{yourname_expnumber}.yaml```



#### Main file:
```python main_testbed_slide.py``` -> loop through all slide and return the patch (image) in the whole slide. 

```python main_testbed_superpixel.py``` -> loop through each slide, then loop through all superixel and return the patch (image) in the whole slide.

---
Loaded model checkpoint from /project/hnguyen2/mvu9/camelyon16/checkpoints/mil_checkpoint.pth (Epoch 31, Best AUC: 0.8403)
---- Evaluation result:
Test Loss = 0.2242, Test Accuracy = 0.8333
AUC = 0.8403
Best Threshold: 0.6127
Class 0: 17/17 correct (1.0000 accuracy)
Class 1: 3/7 correct (0.4286 accuracy) 