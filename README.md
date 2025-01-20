
# TEST BED FOR PATCH MERGING: 

## About test bed:
#### Todo before running main file for test bed: 
1. activate conda/pip env and then:  
```export PROJECT_DIR=$(pwd)```  
2. Fix your path in ```testbed_config/{yourname_expnumber}.yaml```



#### Main file:
```python main_testbed_slide.py``` -> loop through all slide and return the patch (image) in the whole slide. 

```python main_testbed_superpixel.py``` -> loop through each slide, then loop through all superixel and return the patch (image) in the whole slide.


