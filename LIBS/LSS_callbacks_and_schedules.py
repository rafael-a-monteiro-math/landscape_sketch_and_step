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

from .LSS_print import clean_up_log_file, append_to_log_file, save_as_txt, save_as_npy
from .LSS_connector import evaluate_at_validation_data
import time
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
from loguru import logger

##############################################################
##############################################################

#1) Using subprocess with shell=True, when should instead be shell=False
# https://stackoverflow.com/questions/29889074/...
# ...how-to-wait-for-first-command-to-finish
# https://stackoverflow.com/questions/2837214/...
# ...python-popen-command-wait-until-the-command-is-finished
# https://stackoverflow.com/questions/23611396/...
# ...python-execute-cat-subprocess-in-parallel/23616229#23616229

#2) On Python threading multiple bash subprocesses?
# https://stackoverflow.com/questions/14533458/...
# ...python-threading-multiple-bash-subprocesses/14533902#14533902
# https://stackoverflow.com/questions/61763029/...
# ...tensorflow-tf-reshape-causes-gradients-do-not-exist-for-variables
# https://www.programcreek.com/python/example/69298/subprocess.getoutput

from sklearn.model_selection import train_test_split


########################################################################
# CALLBACKS
########################################################################

# Checkpoint callback
# https://towardsdatascience.com/...
# ...deep-neural-networks-for-regression-problems-81321897ca33


class EvaluationCallback():

    def __init__(self, patience=np.inf, save_history_as="history"):
        """
        TODO
        """
        self.score=np.inf * np.ones((1,1))
        self.patience=patience 
        self.counter=0
        self.save_history_as=save_history_as
        print("Callback", self.save_history_as)

    def on_train_begin(self, LSS):
        clean_up_log_file(LSS.log_output)
        append_to_log_file(LSS.log_output, 50 * "*", verbose=LSS.verbose)
        # Getting the timestamp
        dt=datetime.now()
        append_to_log_file(
            LSS.log_output,
            "\n\tLandscape sketch, " + str(dt) + "\n\t" + 25*"-",
            verbose=LSS.verbose)

        # Classical
        append_to_log_file(
            LSS.log_output,f"\nClassical : {LSS.classical}", verbose=LSS.verbose)

        LSS.running_time=time.time()

    def on_box_search_begin(self, LSS, box_search_epoch):
        """
        This is where we check early stopping conditions
        and if the model is on budget
        """

        # Register parameters into log_file
        LSS.history [str(box_search_epoch)]={
            "box_search_epoch" : box_search_epoch, 
            "box lower" : LSS.box[0].numpy(),
            "box upper" : LSS.box[1].numpy(),
            "number of inner searches" :\
                LSS.rounds_within_box_search(box_search_epoch),

        }
        update_log=f"""Parameters used during box search :\n\
            + {LSS.history[str(box_search_epoch)]}"""
        append_to_log_file(
            LSS.log_output, update_log, verbose=LSS.verbose)

        # BUDGET
        if not LSS.on_budget:
            print("Not on budget callback!")
            return False

        # Early stopping
        return self.counter < self.patience

    def on_box_search_end(self, LSS, box_search_epoch):
        """
        This is where we evaluate the parameter
        at the validation data(if exists),
        and increment the counter used for Early stopping
        """
        X, y=LSS.validation_data        
        mean_on_validation=\
            evaluate_at_validation_data(
                LSS, X, y, box_search_epoch=box_search_epoch)

        # mean_on_validation is a list with size best_x
        # Early stopping
        #print("mean_on_validation in callback", mean_on_validation.shape)
        if not mean_on_validation:
            logger.info("No validation set. Use best_y value")
            eval_box_search_end=tf.reduce_min(LSS.best_y)
        else:
            eval_box_search_end=mean_on_validation
            if LSS.classical:
                # mean_on_validation has length best_y.
                if len(eval_box_search_end) !=1:
                    self.score=tf.where(
                        eval_box_search_end < self.score,
                        eval_box_search_end,
                        self.score)
            else:
                logger.info("There is a validation set!")
            print(eval_box_search_end)

        if np.isscalar(eval_box_search_end):
            if eval_box_search_end < self.score:
                self.score=eval_box_search_end
                self.counter=0
            else:
                self.counter +=1


    def on_train_end(self, LSS):
        """
        This is where we print and save the log files
        and important parameters.
        """
        append_to_log_file(
            LSS.log_output,
            "\n"+40*"="+"\nEnd of training!"+"\n"+40*"=",
            verbose=LSS.verbose)

        # Save history as pickled file
        print_message=\
            "\nSaving history to pickled file '" +self.save_history_as+".pickle'"

        append_to_log_file(
            LSS.log_output, print_message,
            verbose=LSS.verbose)

        with open(self.save_history_as +".pickle", 'wb') as save:
            pickle.dump(LSS.history, save, protocol=pickle.HIGHEST_PROTOCOL)

        save_as_txt(LSS.best_x,LSS.parameters_output+"_x")
        save_as_npy(LSS.best_x,LSS.parameters_output+"_x")
        save_as_npy(
            tf.concat(LSS.min_active_agents_x,axis=1),
            LSS.parameters_output+"_min_x")
        save_as_txt(LSS.best_y,LSS.parameters_output+"_y")
        save_as_npy(LSS.best_y,LSS.parameters_output+"_y")
        save_as_npy(
            tf.concat(LSS.min_active_agents_y,axis=1),
            LSS.parameters_output+"_min_y")

        # Number of costly evaluations
        append_to_log_file(
            LSS.log_output,"\nNumber of external evaluations :" +
            str(LSS.evaluation_counter)+"\nNumber of ML fittings : " +
            str(LSS.fitting_counter()), verbose=LSS.verbose)

        # Where best parameters are printed -- y
        append_to_log_file(
            LSS.log_output,"\nLoss function at the best parameter value : " +
            str(LSS.best_y.numpy()) +", printed to the file \n'" +
            str(LSS.parameters_output) +
            "_y.txt', and also stored as an npy file '" +
            str(LSS.parameters_output)+"_y.npy '", verbose=LSS.verbose)

        # Where best parameters are printed -- x
        append_to_log_file(
            LSS.log_output,"\nBest parameter_x value : " +
            str(LSS.best_x.numpy()) + ", printed to the file \n'" +
            str(LSS.parameters_output) +
            "_x.txt', and also stored as an npy file '" +
            str(LSS.parameters_output)+"_x.npy '", verbose=LSS.verbose)

        if LSS.classical:
            where_min=tf.where(
                np.ravel(LSS.best_y.numpy()==tf.reduce_min(LSS.best_y)))
            minimum_x=tf.gather(LSS.best_x, indices=where_min)
            minimum_y=tf.gather(LSS.best_y, indices=where_min)
            
            append_to_log_file(
                LSS.log_output,
                "\n\nBest among all parameters takes value at: \nX=" +
                str(minimum_x.numpy())+" and Y : " +
                    str(minimum_y.numpy()), verbose=LSS.verbose)
                    
        append_to_log_file(
            LSS.log_output, 50 * "*", verbose=LSS.verbose)
        
        LSS.running_time=time.time() - LSS.running_time
        # Elapsed time
        append_to_log_file(
            LSS.log_output,"\nElapsed time during full search : " +
            str(LSS.running_time/3600)+" hours", verbose=LSS.verbose)


########################################################################
# COOLING SCHEDULES
########################################################################

class ShrinkingSchedules():
    """
    TODO
    """
    def __init__(self):
        pass

    def algebraic_decay(
        self, T_init, T_end, N_steps=1,
        kappa=1, decay_step_every=1):

        vec=np.linspace(max(T_init, T_end), min(T_init, T_end), N_steps)
        vector=kappa / vec
        def cooling_sc(epoch, concentration_parameter=0, max_amplitude=0):
            if max_amplitude==0:
                aux=(1 - concentration_parameter) *\
                    vector [min(epoch // decay_step_every, N_steps-1)] +\
                        concentration_parameter /(4*max(T_init, T_end))
            else:
                aux=(1 - concentration_parameter) *\
                    vector [min(epoch, N_steps-1)] +\
                        concentration_parameter /(4*max(max_amplitude,1e-6))
            return aux

        return cooling_sc

    def linear_decay(
        self, T_init, T_end, N_steps=1,
        kappa=1, decay_step_every=1):

        vec=np.linspace(
            max(T_init, T_end), min(T_init, T_end), N_steps)

        vector=kappa * vec
        def cooling_sc(epoch, concentration_parameter=0):
            aux=(1 - concentration_parameter) *\
                vector [min(epoch // decay_step_every, N_steps-1)] +\
                    concentration_parameter * max(T_init, T_end)
            return aux

        return cooling_sc

    def constant(self, value):
        def cooling_sc(epoch):
            return value
        
        return cooling_sc

    def rounds(
        self, rounds_initial, rounds_end,
        rounds_box_prunning=1, decay_step_every=1):

        """rounds_initial and rounds_end are integers"""
        vector=np.linspace(
            rounds_initial, rounds_end,
            num=rounds_box_prunning, dtype=np.int32)

        def cooling_sc(epoch):
            return vector [min(epoch // decay_step_every, rounds_box_prunning-1)]

        return cooling_sc

########################################################################
# MANAGERS
########################################################################

# BEGIN BOX MANAGER

class BoxManager():
    """
    TODO
    """
    def __init__(self, initial_box):
        self.initial_box=initial_box

    ###########
    def shrink_box(self, Box, beta_shrink, best, threshold_factor=1/10):
        """TODO
        """
        l_lower=best - Box [0]
        l_upper=Box [1] - best
        l_box=Box [1] - Box [0]

        threshold=tf.reshape(threshold_factor * l_box,(1,-1))
        a=tf.minimum(tf.abs(l_lower),tf.abs(l_upper))
        
        if any(np.ravel(a.numpy() < threshold.numpy())):
            logger.info("Too close to the edge")
            # if too close to the boundary, prune initial box
            return self.create_box(
                beta_shrink * l_upper,beta_shrink * l_lower,
                best, self.initial_box)
        else:
            return self.create_box(
                beta_shrink * l_upper,beta_shrink * l_lower, best, Box)

    # Defining shrinking schedules
    def create_box(self, length_to_upper, length_to_lower, center, box):
        """
        TODO
        """
        l_low, l_up=box [0], box [1]
        l_low_new=tf.reshape(
            tf.maximum(center - length_to_lower, l_low),(1, -1))
        l_up_new=tf.reshape(
            tf.minimum(center + length_to_upper, l_up), (1, -1))
        
        return tf.concat((l_low_new, l_up_new), axis=0)

    def is_inside_the_box(self, X, Box):
        """Check if X is inside the box.
        return a vector with the indices inside the box. 
       (empty vector if no entry in X is inside the box)."""
        m_left=tf.reduce_all(X >=Box [0], axis=1, keepdims=True)
        m_right=tf.reduce_all(X <=Box [1], axis=1, keepdims=True)
        m_all=tf.concat((m_left, m_right), axis=1)
        return  np.ravel(tf.where(np.ravel(tf.reduce_all(m_all, axis=1))))

    def keep_inside_box(self, LSS, when=["active"]):
        """
        TODO
        """
        if "active" in when:
            x_old, y_old, weights_old=\
                LSS.active_agents_x, LSS.active_agents_y,\
                    LSS.active_param_value_fct

            where_in=self.is_inside_the_box(x_old, LSS.box)
            LSS.active_agents_x=tf.gather(x_old, indices=where_in)
            LSS.active_agents_y=tf.gather(y_old, indices=where_in)
            LSS.active_param_value_fct=tf.gather(weights_old, indices=where_in)

        if "past" in when:
            x_old, y_old, weights_old=\
                LSS.past_agents_x, LSS.past_agents_y,\
                    LSS.past_param_value_fct

            where_in=self.is_inside_the_box(x_old, LSS.box)
            LSS.past_agents_x=tf.gather(x_old, indices=where_in)
            LSS.past_agents_y=tf.gather(y_old, indices=where_in)
            LSS.past_param_value_fct=tf.gather(weights_old, indices=where_in)

    def prune_box(self, LSS, i, when=["active"], box=[]):
        """
        Pruned box is always smaller than the box cut in half in each dimension
       (That is, the box has maximum volume Old_vol/2^dim, where dim is its
        underlying dimension.)
        
        In the end, it save the points to be used in the next search(as initial_points).
        """
        if LSS.is_model:
            logger.info("Pruning model box: ", i)
        else:
            logger.info(f"Pruning: {i}")

        if not LSS.classical:
            if LSS.is_model:
                LSS.box=box
            else:
                LSS.box=self.shrink_box(LSS.box, LSS.box_shrinking(i), LSS.best_x)

        self.keep_inside_box(LSS, when=when)

        if not LSS.is_model:
            if LSS.classical or LSS.past_agents_x.shape[0] < 3:
                y_pred=[]
            else:
                y_pred=LSS.reconstruct_predictor()

            # Used to restart pred samples
            LSS.initial_points=[(
                LSS.active_agents_x, LSS.active_agents_y,
                LSS.active_param_value_fct, y_pred, []
                )]

    def update_box_history(self, LSS, i):
        """
        TODO
        """
        LSS.box_history ["Box_" + str(i)]={
            "box" : np.copy(LSS.box), "pred_samples" : LSS.pred_samples,
            "min_active_agents_x" : np.copy(LSS.min_active_agents_x),
            "min_active_agents_y" : np.copy(LSS.min_active_agents_y),
            "active_param_value_fct" : np.copy(LSS.active_param_value_fct),
            "best_y": LSS.best_y, "best_x": LSS.best_x }    
        
        if LSS.predict_and_plot and (not LSS.classical):
            # In this case, you plot the potential V(\cdot) on the interval
            # defined by the box.
            x_box=np.linspace(*LSS.box,num=50)
            y_pred=LSS.predictor.predict(x_box)
            LSS.box_history ["Box_" + str(i)].update(
                {"Potential-fitting": (x_box, y_pred)}
            )


#########################################
# TRAIN - TEST SPLIT AND GENERATE CONFIGURATION
#########################################
class GenerateConfigurations():
    """
    In LSS_callbacks_and_schedules.

    Generates a random set of traning and test set of configurations.

    Generates N_configurations, of which test_size % will become test size, 
    and other val_data % will become valiadation data.

    """
    def __init__(
        self, n_configurations,
        val_size=.1, test_size=.2,
        random_state=None):
        """
        TODO
        """
        self.n_configurations=n_configurations
        self.test_size=test_size
        self.val_size=val_size
        self.random_state=random_state
        self.count_measurements=0

        self.train_test_split()

    def train_test_split(self):
        """
        Train, val, test splitting.
        """
        aux=np.arange(self.n_configurations, dtype=np.int64)

        # Generate test set
        self.train_plus_val, self.test=train_test_split(
            aux, test_size=self.test_size,
            random_state=self.random_state)

        # Generate validation set and traning set
        self.train, self.validation_data=train_test_split(
            self.train_plus_val, test_size=self.val_size,
            random_state=self.random_state)

        # Keep lengths
        self.len_train=len(self.train)

    def new_configuration(self):
        """
        TODO
        """
        new_config=self.train [self.count_measurements % self.len_train]
        self.count_measurements +=1
        
        return new_config
    