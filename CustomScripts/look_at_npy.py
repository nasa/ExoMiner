import os
import tensorflow as tf
import pandas as pd
from pathlib import Path
import src_preprocessing.tf_util.example_util as example_util
import sklearn
import matplotlib.pyplot as plt
import numpy as np
arr = np.load("/Users/agiri1/Library/CloudStorage/OneDrive-NASA/brown_dwarf_model_test_results/model_without_centroid+radius/res_eval.npy", allow_pickle=True)
print(arr)