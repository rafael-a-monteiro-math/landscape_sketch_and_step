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
#####################################################################
# IMPORT LIBRARIES
#####################################################################

import numpy as np
import tensorflow as tf
from scipy import stats
from loguru import logger

from .LSS_connector import run_and_retrieve


################################################################################
#   EXPLORATION AND RELATED THINGS

def simulated_annealing(LSS, x_values, beta_accept, sigma_sim_an, classical=False):
    """
    In LSS_exploratory_policies.py.

    Simulated annealing, starting at points x. 

    It uses run_and_retrieve in the classical simulated annealing case.
    """
    box_length=LSS.box[1] - LSS.box[0]
    assert all(LSS.box[1] > LSS.box[0])
    noise=sigma_sim_an * tf.random.normal(shape=x_values.shape)
    z_param=x_values + noise
    # 'Reflect' about the boundaries if necessary
    z_param=tf.where(
        z_param - LSS.box[1] > 0,
        LSS.box[1] -(z_param - LSS.box[1]) %(box_length),
        z_param)
    z_param=tf.where(
        z_param - LSS.box[0] < 0,
        LSS.box[0] +(LSS.box[0] - z_param) %(box_length),
        z_param)
    assert tf.reduce_all(LSS.box[1] >=z_param)
    assert tf.reduce_all(LSS.box[0] <=z_param)
    
    if classical or LSS.past_agents_x.shape[0] < 3:
        if LSS.past_agents_x.shape[0] < 3:
            print("past agents' list still too small!")
        #########
        # EVALUATION COUNTER!!
        if 2 * len(x_values) + LSS.evaluation_counter > LSS.evaluation_budget:
            LSS.on_budget=False
        if LSS.on_budget:
            f_pred_x=run_and_retrieve(
                LSS, LSS.input_name, LSS.output_name, LSS.program_name,
                x_active_agents=x_values)
            f_pred_z=run_and_retrieve(
                LSS, LSS.input_name, LSS.output_name, LSS.program_name,
                x_active_agents=z_param)
        else:
            print("Evaluation limit has been reached!")
            
            return None, None
    else:
        predictor=LSS.predictor.predict
        f_pred_x=tf.reshape(predictor(x_values),(-1, 1))
        f_pred_z=tf.reshape(predictor(z_param),(-1, 1))

    aux=tf.exp(tf.minimum(beta_accept *(f_pred_x - f_pred_z), 0))
    # FIX
    unif_rand_noise=np.random.uniform(size=aux.shape)
    
    if classical or LSS.past_agents_x.shape[0] < 3:
        return tf.where(aux > unif_rand_noise, z_param, x_values),\
            tf.where(aux > unif_rand_noise, f_pred_z, f_pred_x)
    else:
        return tf.where(aux > unif_rand_noise, z_param, x_values), []

def exploratory_policy(
    x_old, x_new_low_temp, x_new_high_temp,
    y_old, y_new_low_temp, y_new_high_temp,
    eps_low_temp_or_high_temp, number_samples,
    relaxation_factor=.5, eps_policy=1e-1):
    """
    In LSS_exploratory_policies.py.
    
    
    eps_low_temp_or_high_temp : probability of sampling from deep exploration policy
    number samples : total number of samples

    RMK: when eps==1, policy is deep exploration.
    when eps==0 policy is high_temp exploration

    Return 
    """
    eps_policy=min(eps_policy, 1/tf.pow(x_old.shape[0], 2))

    ###########################################
    # Figure out how many elements will be draw according to each policy
    # To begin with, gather n_low_temp samples using deep exploratory policy
    # Then sample n_total - n_low_temp elements using low_temp exploratory policy.
    
    #The higher eps_low_temp, more concentrated, therefore
    # more high_temp explorationswe want
    n_total=min(x_old.shape[0], number_samples)
    n_high_temp=np.sum(
        np.random.binomial(n=1, p=eps_low_temp_or_high_temp, size=n_total))
    n_low_temp=n_total - n_high_temp
    
    print(
        "\n\nExploratory_policy. Deep : ", n_low_temp," high_temp : ", n_high_temp,
        "n_total", n_total,"number samples", number_samples)
    
    # DEEP EXPLORATION
    low_temp_agents_index=np.empty((0,), dtype=np.int64)
    high_temp_agents_index=np.empty((0,), dtype=np.int64)
    if n_low_temp > 0:
        low_temp_agents_index=\
            exploration_low_temp(y_new_low_temp, n_low_temp, eps_policy=eps_policy)
            
    # Now need to remove the exploration low_temp from
    # the pool of possible contenders
    exploration_high_temp_vct, _=exploration_high_temp(
        x_old, y_old, x_new_high_temp, y_new_high_temp, k=n_low_temp,
        relaxation_factor=relaxation_factor)
    # high_temp EXPLORATION
    # There is a chance that an erroro of type 
    # 'ValueError: Fewer non-zero entries in p than size' happens.
    # That may happen if eps_low_temp is 0 or 1 and the particle does not "move"
    # during simulated annealing. For that reason, we perturb the
    # exploration_high_temp_vct a little
    exploration_high_temp_vct=tf.Variable(exploration_high_temp_vct) + 1e-6
    exploration_high_temp_vct=tf.tensor_scatter_nd_update(
        tensor=exploration_high_temp_vct,
        indices=low_temp_agents_index.reshape(-1, 1),
        updates=tf.zeros_like(low_temp_agents_index, dtype=tf.float32)
        )
    
    # Make a probability vector from the exploration vector
    if n_high_temp > 0:
        p_exploration_high_temp=\
            exploration_high_temp_vct / tf.reduce_sum(exploration_high_temp_vct)
        high_temp_agents_index=\
            np.random.choice(
                np.arange(len(p_exploration_high_temp)),
                p=p_exploration_high_temp.numpy(),
                size=n_high_temp, replace=False)
    
    return low_temp_agents_index, high_temp_agents_index,\
        exploration_high_temp_vct, n_low_temp, n_high_temp

def exploration_high_temp(
    x_old, y_old, x_new, y_new,
    k=5, relaxation_factor=.5, eps_policy=1e-2):
    """
    In LSS_exploratory_policies.py.
    
    
    relaxation_factor=1 considers only vertical, 0 considers only horizontal.
    """
    logger.info(f"Exploration high_temp with vertical index {relaxation_factor}")
    AVOID_DIVISION_ZERO=1e-8
    prob_vec_build=ProbabilityVecBuilder()
    theta=tf.cast(relaxation_factor, dtype=tf.float32)
    min_k=x_old.shape[0]
    where_min=np.argmin(y_old)
    #  Inspired by K-means++, to keep centroids apart
    exploration_horizontal=tf.pow(
        tf.linalg.norm(x_old[where_min] - x_new, axis=1), 2)
    # This could give problems if there's just one point that does not move!!!
    normalizer_horizontal=tf.reduce_sum(exploration_horizontal)
    if normalizer_horizontal !=0 :
        exploration_horizontal /=tf.reduce_sum(exploration_horizontal)
    else: 
        exploration_horizontal=\
            tf.ones(shape=[x_new.shape[0]], dtype=tf.float32)
    # exploration_horizontal and Exploration_vertical
    # should be probability vectors
    exploration_vertical=AVOID_DIVISION_ZERO +\
        tf.cast(tf.abs(y_old - y_new), dtype=tf.float32)
    exploration_vertical=\
        prob_vec_build.max(
            np.ravel(exploration_vertical), eps_policy=eps_policy)
    exploration=\
        theta * exploration_vertical +(1 - theta) * exploration_horizontal
    # Retrieve k agents with maximal exploration
    aux=tf.math.top_k(tf.transpose(exploration), k=min(k, min_k))
    
    return exploration, aux.indices

def exploration_low_temp(vector, size, eps_policy=1e-2):
    """ 
    In LSS_exploratory_policies.py.
    
    Receives a vector v, creates a related probability vector, 
    and returns the index of the 
    elements chosen according to the probability vector.
    """
    prob_vec_build=ProbabilityVecBuilder()
    prob=prob_vec_build.min(np.ravel(vector), eps_policy=eps_policy).numpy()
    
    # This is already a weights vector with value 1
    # at the minumim, therefore we use MAX
    return  np.random.choice(
        np.arange(vector.shape[0]), p=prob, size=size, replace=False)

class ProbabilityVecBuilder():
    """
    In LSS_exploratory_policies.py.
    
    """
    def __init__(self):
        self.eps=1e-8
        
    def max(self, vector, eps_policy):
        """ Build probability vec based on v,
        having max probability at argmax(v)"""
        beta_accept = -np.log(eps_policy)
        min_v=tf.reduce_min(vector)
        max_v=tf.reduce_max(vector)
        
        return tf.nn.softmax(
            -beta_accept*tf.abs(vector - max_v) /(max_v - min_v + self.eps))
        
    def min(self, vector, eps_policy):
        """ Build probability vec based on v,
        having max probability at argmin(v)"""
        beta_accept = -np.log(eps_policy)
        min_v, max_v=tf.reduce_min(vector), tf.reduce_max(vector)
        
        return  tf.nn.softmax(
            - beta_accept * tf.abs(vector - min_v) /(max_v - min_v + self.eps))

################################################################################
# AUTO-ADJUSTING FUNCTIONS AND RELATED METRICS
################################################################################

def adjust_sigma_sim_an(
    LSS, eps=0,
    bm_step_size_low_temp=20.,
    bm_step_size_high_temp_1=20.,
    bm_step_size_high_temp_2=4.):
    """
    In LSS_exploratory_policies.py.
    """
    logger.info("Adjusting sigma")
    length=(LSS.box[1] - LSS.box[0])
    LSS.sigma_sim_an_high_temp=\
        length *\
           (1 / bm_step_size_high_temp_1 + eps / bm_step_size_high_temp_2)    
    LSS.sigma_sim_an_low_temp=length / bm_step_size_low_temp

def concentration_measure(target, X, box):
    """
    In LSS_exploratory_policies.py.
    
    target has shape(1, dims)
    X has shape(batch, dims)
    box has shape(dims, 2)
    
    The function returns 0 for each dimension that is not concentrated,
    1 whenever it is concentrated.
    
    
    For KL divergence, see 
    https://docs.scipy.org/doc/scipy/reference/
    generated/scipy.stats.entropy.html?highlight=entropy
    
    For histogram, see 
    https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    """
    # THE RELATIVE ENTROPY
    # length around target
    limits=tf.transpose(
        tf.concat((box[0] - target, box[1] - target), axis=0)).numpy()
    
    relative_dist=(X - target).numpy() # centralize

    n_particles, n_dims=X.shape #this will denote the size of the bins
    measure_of_concentration=(1 / 2)*np.ones(n_dims, dtype=np.float32)
    
    if n_particles > 1:

        for i in range(n_dims):
            
            concentration, edges=np.histogram(
                relative_dist[:,i], bins=n_particles, range=limits[i]
            )
            concentration=concentration /n_particles

            logger.info(f"\nDimension {i}, concentration={concentration}")
            measure_of_concentration[i]=\
                stats.entropy(
                    concentration,
                    qk=np.ones(n_particles)/n_particles) /(np.log(n_particles))
        
            compare_centered_at_target=np.zeros(n_particles, dtype=np.float32)
            j=0
            while(edges[j] < 0):
                j+=1   # len(edges)=len(compare_centered_at_target) + 1
            compare_centered_at_target[j-1]=1
            
            # beta splits the measure aroung the target(minimum know at the time)
            # attributing at most %50 of mass to the l_1 norm
            beta=concentration[j-1]
            # THE L1 DISTANCE(useful, because the concentrated prob is sparse)
            l1_dist=1 - np.sum(concentration - compare_centered_at_target)/2
            # In other words: if they overlap - or
            # get clustered, then the diff between them will maximize """
            
            # Combine both measures
            measure_of_concentration[i]=\
                (1 - beta)* measure_of_concentration[i]+\
                    beta * l1_dist

    return measure_of_concentration

def epsilon_low_temp_adjust(target, x_data, box, adjustment_rate=1., average_method="max"):
    """
    In LSS_exploratory_policies.py.
    
    """
    if average_method=="max":
        return adjustment_rate * tf.reduce_max(
            concentration_measure(target, x_data, box)).numpy()
    else:
        return adjustment_rate * tf.reduce_mean(
            concentration_measure(target, x_data, box)).numpy()
