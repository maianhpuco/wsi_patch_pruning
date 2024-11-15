import tensorflow as tf 
import sys
import os
import sys 

PRJ_DIR = os.environ.get("PROJECT_DIR")
sys.path.append(os.path.join(PRJ_DIR))

from nystromformer.nystrom_attention import  * 
from nystromformer.nystromformer import * 



if __name__ == "__main__":
    print(">> dome import ")
