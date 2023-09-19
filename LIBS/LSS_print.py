#!/usr/bin/env python

#####################################################################
# This code is part of the simulations done for the paper
# 'Landscape-Sketch-Step: An AI/ML-Based Metaheuristic
#     for Surrogate Optimization Problems',
# by Rafael Monteiro and Kartik Sau
#
# author : Rafael Monteiro
# affiliation : Mathematics for Advanced Materials -
#               Open Innovation Lab(MathAM-OIL, AIST)
#               Sendai, Japan
# email : rafael.a.monteiro.math@gmail.com
# date : July 2023
#
#####################################################################

# IMPORT LIBRARIES
#####################################################################
__author__="Rafael de Araujo Monteiro"
__affiliation__=\
    """Mathematics for Advanced Materials - Open Innovation Lab,
        (Matham-OIL, AIST),
        Sendai, Japan"""
__copyright__="None"
__credits__=["Rafael Monteiro"]
__license__=""
__version__="0.0.0"
__maintainer__="Rafael Monteiro"
__email__="rafael.a.monteiro.math@gmail.com"
__github__="https://github.com/rafael-a-monteiro-math/"
__date__=""

import numpy as np
import tensorflow as tf
from loguru import logger

#########################################
# PRINTING
#########################################


def print_details(LSS):
    """
    Print details of LandscapeSketchandStep model,
    appending it to its log file.
    """
    append_to_log_file(
        LSS.log_output,
        "\nRunning full search with parameters\n", verbose=LSS.verbose)

    print_this="eps :" +str(tf.squeeze(LSS.sigma_sim_an_low_temp).numpy())+\
                ", alpha_wgt_updt :"+str(LSS.alpha_wgt_updt)+"\n\n"+\
                    "\nmulti armed :" +str(LSS.multi_armed)+\
                        "\nwhich model :" +str(LSS.which_models)

    append_to_log_file(LSS.log_output, print_this, verbose=LSS.verbose)


def save_as_npy(x_value, name):
    """
    Save x as a npy file name +".npy". 
    It assumes that x is in numpy format
    """
    filename_now=name.split(".")[0]
    with open(filename_now+'.npy', 'wb') as file:
        np.save(file, x_value)


def load_npy(name="ab_initio_chunk"):
    """
    Load npy file from file name +".npy" as numpy vector, which
    is then returned.
    
    name : {str, "ab_initio_chunk"}

    """
    with open(name+'.npy', 'rb') as file:
        x=np.load(file)
    return x


def save_as_txt(x_value, name):
    """
    Save the numpy vector x
    as a txt file, one entry per row, 
    to a file with name name +".txt", 
    """
    if name.split(".")[-1]=="txt":
        filename_now=name
    else:
        filename_now=name + ".txt"

    with open(filename_now,"w") as f_now:
        np.savetxt(f_now, x_value, delimiter='\n')


def print_parameters(data_x, name="ab_initio_chunk", in_parts=True):
    """
    Print data elements form the numpy vector data_x into txt files. 

    If in_parts=True, then it prints each entry of data_x to a separate file, 
    else print the wholedataset to a single file with name name+".npy"


    """
    if in_parts:
        name_base="parameters_"
        data_shape=data_x.shape[0]
        for i in range(data_shape):
            x=data_x[i]
            save_as_txt(x, name_base + str(i))
        logger.info("Printed in parts, each parameter to a different txt file.")
    else:
        save_as_npy(data_x, name)
        print("All printed to a single npy file.")


def load_txt(configurations):
    """
    Create numpy matrix M from files "parameters_"+ str(conf)+".txt", where
    conf in configurations. 
    Each row of M is made up of data read in txt files.
    """
    files=[]
    for conf in configurations:
        ext="parameters_"+str(conf) + ".txt"
        lower=tf.reshape(
            tf.constant(np.loadtxt(ext, dtype=np.float32)),(1,-1))
        files.append(lower)

    m_stacked=tf.Variable(np.vstack(files), dtype=tf.float32)

    return m_stacked


def append_to_log_file(filename, string, verbose=False):
    """
    Append content to a txt file.
    """
    with open(filename, "a") as file:
        print(string, file=file)
    if verbose:
        logger.info(string)


def clean_up_log_file(filename):
    """
    Erase content of a txt file.
    """
    with open(filename, "w") as file:
        print("", file=file)
    