
# TEST BED FOR PATCH MERGING: 

## About test bed:
#### Todo before running main file for test bed: 
1. activate conda/pip env and then:  
```export PROJECT_DIR=$(pwd)```  
2. Fix your path in ```testbed_config/{yourname_expnumber}.yaml```



#### Main file:
```python main_testbed_slide.py``` -> loop through all slide and return the patch (image) in the whole slide. 

```python main_testbed_superpixel.py``` -> loop through each slide, then loop through all superixel and return the patch (image) in the whole slide.

- Train a Bag Classifier alone:
2025-01-28 14:15:32,674 - Class 0: Accuracy: 0.667, Correct: 8/12
2025-01-28 14:15:32,674 - Class 1: Accuracy: 0.333, Correct: 2/6
2025-01-28 14:15:32,674 -  Validation Loss: 2.8346 
