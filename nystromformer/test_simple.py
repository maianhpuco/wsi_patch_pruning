import tensorflow as tf 
import sys
import os


PROJECT_DIR = os.environ("PROJECT_DIR")
sys.path.append(os.path.join(PROJECT_DIR))

from nystromformer.nystrom_attention import  * 
from nystromformer.nystromformer import * 
if __name__ == "__main__":
    print("dome import ")
