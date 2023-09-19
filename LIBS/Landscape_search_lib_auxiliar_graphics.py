# 
###
################################################################################

import os, time, copy, shutil, random, pickle, sys, subprocess
import multiprocess as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tensorflow import keras
from keras import backend as K
from scipy.interpolate import BSpline
from functools import partial
from sklearn.svm import SVR
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

##############################################################
##############################################################

#1) Using subprocess with shell=True, when should instead be shell=False

# https://stackoverflow.com/questions/29889074/how-to-wait-for-first-command-to-finish

# https://stackoverflow.com/questions/29889074/how-to-wait-for-first-command-to-finish

# https://stackoverflow.com/questions/2837214/python-popen-command-wait-until-the-command-is-finished

# https://stackoverflow.com/questions/23611396/python-execute-cat-subprocess-in-parallel/23616229#23616229 

# On Python threading multiple bash subprocesses?
# https://stackoverflow.com/questions/14533458/python-threading-multiple-bash-subprocesses/14533902#14533902

# https://stackoverflow.com/questions/61763029/tensorflow-tf-reshape-causes-gradients-do-not-exist-for-variables

# https://www.programcreek.com/python/example/69298/subprocess.getoutput

################################################################################

# Where to save the figures
PROJECT_ROOT_DIR="."
IMAGES_PATH=os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path=os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
