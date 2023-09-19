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

import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from loguru import logger
try: 
    from scikeras.wrappers import KerasRegressor
except:
    from keras.wrappers.scikit_learn import KerasRegressor
from .LSS_print import append_to_log_file
from .LSS_connector import run_and_retrieve, sample_from_box
from .LSS_callbacks_and_schedules import ShrinkingSchedules, BoxManager
from .LSS_exploratory_policies import exploratory_policy, ProbabilityVecBuilder

################################################################################


def evaluate_and_store(
    LSS, sample_x, y_eval=[], epoch=0):
    """
    In LSS_model.py.
    
    Add 'sample_x' to the sample, finally evaluating them
    by using 'run_and_retrieve'.
    In the end, truncates to conform to length constraints.

    Evaluations only happen if y_eval !=[]
    This function also checks if number of evaluations goes above the budget
    """
    if not LSS.classical:
        logger.info("Landscape fitting!!")
        # There's need to remove repeated entries
        # unique_sample_x has all elements in sample_x, without repetition

        # ork well is sample is tensorflow vector
        try:
            unique_sample_x=np.unique(sample_x, axis=0)
        except:
            unique_sample_x=np.unique(sample_x.numpy(), axis=0)

        if not LSS.is_model:
            logger.info("Is not model!")
            # First check who has not been evaluated yet
            unique_already_evaluated_idx=[]
            for x in unique_sample_x:
                unique_already_evaluated_idx.append(
                    any(tf.reduce_all(x==LSS.past_agents_x, axis=1)))

            # These are the only ones that need to go to past_agents_x list
            # Because empty lists ate not in int64 format
            unique_already_evaluated_idx=np.ravel(
                np.array(unique_already_evaluated_idx, dtype=np.bool8)
                )
            unique_not_yet_evaluated_idx=np.ravel(
                tf.squeeze(tf.where(np.invert(unique_already_evaluated_idx)))
                )
            # It has to be after
            unique_already_evaluated_idx=np.ravel(
                tf.squeeze(tf.where(unique_already_evaluated_idx))
                )
            # Redefine unique_x
            unique_sample_x=tf.gather(
                unique_sample_x, indices=unique_not_yet_evaluated_idx
                )

        #The next part is only necessary if unique_sample_x is non-empty
        if len(unique_sample_x) > 0:
            # Retrieve Y
            if (len(y_eval) == 0) and (not LSS.is_model):
                logger.info("retrieving data by run_and_retrieve")
                # Updating parameters not yet seen
                if len(unique_sample_x) + LSS.evaluation_counter > LSS.evaluation_budget:
                    LSS.on_budget=False
                if LSS.on_budget:
                    unique_sample_y=tf.cast(
                        run_and_retrieve(
                            LSS, LSS.input_name,
                            LSS.output_name,
                            LSS.program_name,
                            x_active_agents=unique_sample_x
                            ),
                        dtype=tf.float32)
            else:
                if LSS.is_model:
                    logger.info("Is model!")
                    unique_sample_y=y_eval
                else:
                    where_in_y_eval=[]
                    for x in unique_sample_x:
                        where=tf.where(tf.reduce_all(x==sample_x, axis=1))[0,0]
                        where_in_y_eval.append(where)

                    # Because empty lists ate not in int64 format
                    where_in_y_eval=np.array(where_in_y_eval, dtype=np.int64)
                    unique_sample_y=tf.gather(y_eval, indices=where_in_y_eval)

            # ----------------------------------------------------
            if LSS.on_budget:
                # UPDATES
                append_to_log_file(
                    LSS.log_output, "\nUpdates -- past",
                    verbose=LSS.verbose)
                # We are ready to insert the new elements to the
                # past_agents_ list
                add_how_many_past=unique_sample_x.shape[0]
                update_entry(
                    LSS, unique_sample_x, when=["past"], extension="x")
                update_entry(
                    LSS, unique_sample_y, when=["past"], extension="y")
                update_entry(
                    LSS, tf.ones(shape=(add_how_many_past,1)),
                    when=["past"], extension="param_value")

                # TIME TO FEED THE DATA TO THE MODEL!!
                if not LSS.is_model:
                    logger.info("Feed to the model")
                    LSS.family_of_models.feed_and_store(
                        unique_sample_x, unique_sample_y)

    # For both models, classical or not
    if (len(y_eval) == 0):
        if LSS.classical:
            #########
            if\
                sample_x.shape[0] + LSS.evaluation_counter > LSS.evaluation_budget:
                    LSS.on_budget=False

            if LSS.on_budget:        
                # EVALUATION COUNTER!!
                sample_y=tf.cast(
                    run_and_retrieve(
                        LSS, LSS.input_name, LSS.output_name,
                        LSS.program_name,x_active_agents=sample_x),
                        dtype=tf.float32)
        else:
            logger.info("Redefining sample_y")
            # Now we simply retrieve all the data
            # we need from the past_agents_list
            where_in_past_exp_idx=[]
            for x in sample_x.numpy():
                try:
                    # In case LSS.past_agents_x is empty
                    where=tf.where(
                        tf.reduce_all(x==LSS.past_agents_x, axis=1))[0,0]
                    where_in_past_exp_idx.append(where)
                except:
                    # If you don't put it here it complains about indentation
                    pass
            # Because empty lists ate not in int64 format
            
            where_in_past_exp_idx=np.array(
                where_in_past_exp_idx, dtype=np.int64)
            sample_y=tf.gather(
                LSS.past_agents_y, indices=where_in_past_exp_idx)
    else:
        # Classical==True always fall here
        # as well as if you are feeding the model
        sample_y=tf.Variable(y_eval)

    if LSS.on_budget:
        # ...
        append_to_log_file(
            LSS.log_output, "\nUpdates -- active",
            verbose=LSS.verbose)
        # ... the lists of active and agents
        add_how_many_active=sample_x.shape [0]
        update_entry(LSS, sample_x, when=["active"], extension="x")
        update_entry(LSS, sample_y, when=["active"], extension="y")
        update_entry(
            LSS, tf.ones(shape=(add_how_many_active,1)),
            when=["active"], extension="param_value")

        # ... active weights(USED FOR POLICY)
        LSS.active_param_value_fct=param_value_update(
            LSS.active_param_value_fct, LSS.active_agents_y,
            alpha_wgt_updt=LSS.alpha_wgt_updt, probability=False,
            eps=1/tf.pow(len(LSS.active_agents_y), 2))

        if not LSS.classical:
            # .... past weights(USED FOR FITTING)
            LSS.past_param_value_fct=param_value_update(
                LSS.past_param_value_fct, LSS.past_agents_y,
                alpha_wgt_updt=LSS.alpha_wgt_updt, probability=False,
                eps=1/tf.pow(LSS.past_agents_y.shape[0], 2))

        LSS.keep_inside_box(when=["active"])
        # The next quantity is used in the exploration policy.
        # See 'new_states'
        LSS.max_amplitude=\
            tf.reduce_max(LSS.past_agents_y).numpy()\
                - tf.reduce_min(LSS.past_agents_y).numpy()
################################################################################


def populate(LSS, initial_points, epoch):
    """
    In LSS_model.py.
    
    'populate' receives a tuple 
    
    * initial_points: list or tensor
    * LandscapeSketch 

    It populates active_agents_x, 
    active_agents_y and weights with initial values

    """
    logger.info("Populating the box!")
    if tf.is_tensor(initial_points):  ##3 In case fed data is just a vector
        logger.info("is tensor!")

        # ive agents in casae initial points has been gven
        if LSS.n_min_active_agents(0) !=initial_points.shape[0]:
            logger.info("Redefining min active agents")
            S=ShrinkingSchedules()
            LSS.n_min_active_agents=S.constant(initial_points.shape[0])
            if LSS.classical:
                logger.info("Redefining max active agents")
                # If classical simulated annealing, then overwrite N_max_active agents
                LSS.n_max_active_agents=LSS.n_min_active_agents

        # HALTING!!!
        evaluate_and_store(LSS, initial_points, epoch=0)
        truncate_agents(LSS, epoch)   # Truncate active agents
        LSS.keep_inside_box(when=["past"])  # Eliminate elements out of the box
        LSS.initial_points=[
            LSS.active_agents_x,
            LSS.active_agents_y,
            LSS.active_param_value_fct,
            None,None]
        LSS.pred_samples=LSS.initial_points
    else:
        logger.info("Not a tensor")
        LSS.active_agents_x=tf.Variable(np.empty((0, LSS.dim)), dtype=tf.float32)
        LSS.active_agents_y=tf.Variable(np.empty((0, 1)), dtype=tf.float32)
        LSS.active_param_value_fct=tf.Variable(np.empty((0, 1)), dtype=tf.float32)

        if epoch==0 or (not initial_points):

            sample_x=sample_from_box(LSS.box, LSS.n_min_active_agents(0))
            # HALTING!!!
            evaluate_and_store(LSS, sample_x, epoch=0)  # will run_and_retrieve
            truncate_agents(LSS, epoch)
            LSS.keep_inside_box(when=["past"])  # Eliminate elements out of the box
        else:
            logger.info("Initial points were given to the LSS")
            act_expl_x, act_expl_y, _, _, _=LSS.initial_points[0]
            evaluate_and_store(LSS, act_expl_x, y_eval=act_expl_y, epoch=0)
            LSS.keep_inside_box(when=["past"])  # Eliminate elements out of the box
            # IDE BOX?
            LSS.pred_samples=LSS.initial_points
################################################################################


def update_entry(LSS, update_value, when, extension="param_value"):
    """
    In LSS_model.py.
    """
    if "active" in when:
        if  extension=="x":
            LSS.active_agents_x=tf.concat(
               (LSS.active_agents_x, update_value), axis=0)
        elif extension=="y":
            LSS.active_agents_y=tf.concat(
               (LSS.active_agents_y, update_value), axis=0)
        elif extension=="param_value":
            LSS.active_param_value_fct=tf.concat(
               (LSS.active_param_value_fct, update_value), axis=0)
    if "past" in when:
        if  extension=="x":
            LSS.past_agents_x=tf.concat(
               (LSS.past_agents_x, update_value), axis=0)
        elif extension=="y":
            LSS.past_agents_y=tf.concat(
               (LSS.past_agents_y, update_value), axis=0)
        elif extension=="param_value" :
            LSS.past_param_value_fct=tf.concat(
               (LSS.past_param_value_fct, update_value), axis=0)


def truncate_agents(LSS, epoch):
    """
    In LSS_model.py.
    """
    if LSS.truncate=="weights":
        keep=eliminate_by_weights(LSS.active_param_value_fct)
        keep=keep [:LSS.n_max_active_agents(epoch)]
        LSS.active_param_value_fct=\
            tf.gather(LSS.active_param_value_fct, indices=keep)
        LSS.active_agents_x=tf.gather(LSS.active_agents_x, indices=keep)
        LSS.active_agents_y=tf.gather(LSS.active_agents_y, indices=keep)
    else:
        LSS.active_param_value_fct=\
            LSS.active_param_value_fct [-LSS.n_max_active_agents(epoch):]
        LSS.active_agents_x=LSS.active_agents_x [-LSS.n_max_active_agents(epoch):]
        LSS.active_agents_y=LSS.active_agents_y [-LSS.n_max_active_agents(epoch):]
        

################################################################################
# WEIGHT UPDATE
def param_value_update(
    state_value_ftn, Y, alpha_wgt_updt, eps=1e-2, probability=True):
    """
    In LSS_model.py.
    
    Update weights using temporal differencing.
    """
    prob_build=ProbabilityVecBuilder()
    r_theta=tf.reshape(prob_build.min(np.ravel(Y), eps_policy=eps),(-1, 1))
    if not probability:
        max_r_theta=tf.reduce_max(r_theta)
        if max_r_theta > 0:
            r_theta=(1 / max_r_theta) * r_theta
    
    return (1 - alpha_wgt_updt) * state_value_ftn + alpha_wgt_updt * r_theta


def eliminate_by_weights(weights):
    """
    In LSS_model.py.
    
    Find the indexes of weights, in descending order(0 is highest; -1 is the lowest)
    See https://pythonguides.com/python-sort-list-of-tuples/ 
    """
    aux=tf.math.top_k(tf.transpose(weights), k=weights.shape[0])
    # RMK: aux is ranked from higher to lower values
    return np.ravel(aux[1].numpy())


# Building wrappers
# https://github.com/adriangb/scikeras/issues/144
#
# Can I send callbacks to a KerasClassifier?
# https://stackoverflow.com/questions/42492824/...
# ...can-i-send-callbacks-to-a-kerasclassifier
#
# Example
# estimators.append(
#('mlp', KerasClassifier(build_fn=create_baseline, nb_epoch=300, batch_size=16,
#  verbose=0, callbacks=[...your_callbacks...])))
################################################################################

def landscape_step(
    LSS, sample_low_temp, sample_high_temp, y_eval=[],
    box_search_epoch=0, sub_epoch=0, eps_policy=1e-2):
    """
    In LSS_model.py.

    Fit the model and evaluate using ....
    This function is responsive for keeping track of the evaluations, 
    not letting they go beyond the prefixed budget

    eps_policy : float, used to assign the ration of the probability
    of the lowest and largest probability in the p policy measure.
    """

    if LSS.classical or LSS.past_agents_x.shape[0] < 3:
        # We get here either if there's not enough past data, or in case of
        # classical simulated annealing
        if LSS.past_agents_x.shape[0] < 3 and not LSS.classical:
            print_message=\
                "Past agents' list still too small!"\
                    +" Resorting to classical Simulated annealing instead"
            append_to_log_file(
                LSS.log_output, print_message,verbose=LSS.verbose)
        sample_x=sample_low_temp
    else:
        # Sketch before stepping
        #
        # One needs to deal with old and new evaluations separately due to the
        # queue data-structure
        f_predict=LSS.predictor.predict
        y_old=np.ravel(LSS.active_agents_y.numpy())
        y_new_low_temp=f_predict(sample_low_temp)
        y_new_high_temp=f_predict(sample_high_temp)
        
        # DGET
        # how many evaluations we are left with.
        left_on_budget=LSS.evaluation_budget - LSS.evaluation_counter

        if left_on_budget > 0:
                
            # How many we'll in fact use.
            number_samples=min(LSS.n_expensive_evltns(sub_epoch), left_on_budget)

            indices_low_temp,\
                indices_high_temp,\
                    exploration_index,\
                        n_low_temp, n_high_temp=exploratory_policy(
                            LSS.active_agents_x,
                            sample_low_temp,
                            sample_high_temp,
                            y_old,
                            y_new_low_temp,
                            y_new_high_temp,
                            LSS.eps_low_temp_or_high_temp,
                            number_samples,
                            relaxation_factor=LSS.relaxation_factor,
                            eps_policy=eps_policy)
            # Gather and concatenate
            aux_low_temp=tf.gather(sample_low_temp, indices=indices_low_temp)
            aux_high_temp=tf.gather(sample_high_temp, indices=indices_high_temp)
            sample_x=tf.concat((aux_low_temp,aux_high_temp), axis=0)
            LSS.exploration_index=exploration_index

            # ster number of explorations into the history dict
            LSS.history[str(box_search_epoch)].update({
                "Epoch "+str(sub_epoch) +" - low_temp evals" : n_low_temp,
                "Epoch "+str(sub_epoch) +" - high_temp evals" : n_high_temp
            })
        else:
            print("Running out of budget, in landscape_step")
            LSS.on_budget=False

    # evaluation step: \theta \mapsto E(\theta)!!
    if LSS.on_budget:
        # The point is that classical is always stored, because sample_x has been
        # evaluated in advance, whereas in the non-classical case we
        # computed the number of remaining staes to evaluate and none are left
        # if LSS.budget=False

        evaluate_and_store(LSS, sample_x, y_eval=y_eval, epoch=sub_epoch)
        truncate_agents(LSS, sub_epoch)
################################################################################


class LandscapeSketch():
    """
    In LSS_model.py.
    
    """
    def __init__(
        self, LSS, cross_val=3, eps_multi_armed=.1,
        ml_epochs_gridsearch=10, patience=10):

        self.is_model=True
        self.eps_multi_armed=eps_multi_armed
        logger.info("Defining family of models!")
        self.couple_model_to_landscape_fitting(LSS)

        # self.classical is always false, since it won't be used for
        # classical calculations
        self.classical=False
        self.dim=self.initial_box.shape [1]
        self.box=np.copy(self.initial_box)

        self.box_manager=BoxManager(self.initial_box)
        self.badges_number=0
        self.cross_val=min(cross_val,3)
        ###TRACKERS
        # active ------------------------------
        self.active_agents_x=tf.Variable(
            np.empty((0, self.dim)), dtype=tf.float32)
        self.active_agents_y=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.active_param_value_fct=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        # past ------------------------------
        self.past_agents_x=tf.Variable(
            np.empty((0, self.dim)), dtype=tf.float32)
        self.past_agents_y=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.past_param_value_fct=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        # Creating the model
        self.predictor=None
        self.ml_epochs_gridsearch=ml_epochs_gridsearch
        self.patience=patience

        # For statistical log
        self.verbose=LSS.verbose
        self.number_fits=0
        self.best_index_hist={}
        self.alpha_multi_bandit=.2
        self.on_budget=True
        self.log_output=LSS.log_output

    def couple_model_to_landscape_fitting(self, LSS):
        """
        We won't need the input_name or so because we don't want 
        to do these computations again
        """

        logger.info("Coupling model to Landscape fitting")
        self.initial_box=LSS.initial_box
        self.truncate=LSS.truncate
        self.with_configurations=LSS.with_configurations
        self.alpha_wgt_updt=LSS.alpha_wgt_updt

    def clean_up_active_box(self):
        """
        To be called when a new box is started"""
        logger.info("Cleaning up active box")
        # Update fitting sample before prunning
        self.active_agents_x=self.past_agents_x
        self.active_agents_y=self.past_agents_y
        self.active_param_value_fct=self.past_param_value_fct
        self.keep_inside_box(when=["active"])  #active box only!

    def create_model(
        self, box, verbose=False, which_models=["svr"],
        wrapped={}, kFold=False, refit=False):

        if 'svr' in which_models:

            self.model=gridsearchcv(
                box=box,
                cross_val=self.cross_val,
                verbose=verbose,
                which_models=which_models,
                wrapped=wrapped,
                kFold=kFold,
                refit=refit)
        else:
            self.model=GridSearchcvKeras(
                box=box,
                cross_val=self.cross_val,
                verbose=verbose,
                which_models=which_models,
                wrapped=wrapped,
                ml_epochs_gridsearch=self.ml_epochs_gridsearch,
                patience=self.patience,
                kFold=kFold,
                refit=refit,
                dimension=self.dim)

    def fit(
        self, X=None, y=None, sample_weight=None,
        epochs=40, verbose=True, which_models=['svr'],
        multi_armed=False, gridsearch=False, estimator__batch_size=4):
        """
        TODO
        """
        print("\n\nEpochs now", epochs)

        if (not X)  or (not y):
            X=self.active_agents_x
            y=self.active_agents_y
            sample_weight=self.active_param_value_fct

        if self.number_fits==0 or gridsearch:
            # You reach this part only in the beginning of box searches
            # That's when the models are fully retrained
            if self.number_fits==0:
                append_to_log_file(
                    self.log_output,
                    "First full gridsearch",
                    verbose=self.verbose)
            elif gridsearch:
                append_to_log_file(
                    self.log_output,
                    "Full gridsearch - refitting the model",
                    verbose=self.verbose)

            # FIT COUNTER!!
            self.create_model(
                box=self.initial_box,
                verbose=verbose,
                which_models=which_models,
                kFold=True, refit=False)

            if (len(sample_weight) == 0):
                self.model.fit(X.numpy(),np.ravel(y.numpy()))
            else:
                self.model.fit(X.numpy(),np.ravel(y.numpy()),
                estimator__sample_weight=np.ravel(sample_weight))

            # IMPORTANT!!!!
            #  sklearn is a maximizer, so cost function  is negative
            if not multi_armed or self.number_fits==0:
                self.rewards=np.copy(self.model.cv_results_['mean_test_score'])
                self.length_rewards=len(self.model.cv_results_['mean_test_score'])
                self.all_params=copy.deepcopy(self.model.cv_results_['params'])
                #self.all_params=wrap_params(self.all_params)
            elif multi_armed:
                # temporal differencing the reward
                append_to_log_file(
                    self.log_output,
                    "\nTemporal differencing the rewards",
                    verbose=self.verbose)
                self.rewards=self.model.cv_results_['mean_test_score']

            self.number_fits +=len(self.model.cv_results_['params']) * self.cross_val

        aux_max=np.where(self.rewards==np.max(self.rewards))
        max_index=np.random.choice(np.ravel(aux_max))

        if multi_armed:
            # print the model has been fitted once, so
            #  we have a record of it's rewards.
            logger.info(f"Multi-armed, p={1./self.length_rewards}")
            eps=np.random.binomial(n=1, p=1./self.length_rewards)
            # Not the best way to do this, but we want to store both
            if eps==1:
                # Have to subtract 1, otherwise max is included!!!
                # It took a while to track this error.
                append_to_log_file(
                    self.log_output,
                    "\nIn multi-armed, non-optimal choice.",
                    verbose=self.verbose)
                index=np.random.randint(0, self.length_rewards-1)
            else:
                append_to_log_file(
                    self.log_output,
                    "\nIn multi-armed, optimal choice",
                    verbose=self.verbose)
                index=max_index
        else:
            index=max_index

        retrieve_params=self.all_params [index]
        name=str(retrieve_params['estimator']).split("(")[0]

        logger.info(f"Retrieve params {retrieve_params}, name : {name}")
        
        # Notice that by calling gridsearch with a wrapped dict you'll
        # obtain just a single model
        if multi_armed:
            append_to_log_file(
                    self.log_output,
                    "\nMulti-armed -- fitting selected",
                    verbose=self.verbose)
            retrieve_params_cp=copy.deepcopy(retrieve_params)
            if "mlp" in which_models or "mlp_low_temp" in which_models:
                wrap_params([retrieve_params_cp])

            self.create_model(
                self.box, verbose=False, which_models=which_models,
                wrapped=retrieve_params_cp,
                kFold=False,
                refit=True)
            self.predictor=self.model
        else:
            append_to_log_file(
                    self.log_output,
                    "\nNot multi-armed -- fitting best",
                    verbose=self.verbose)
            # Just retrieve the best
            if "svr" in which_models:
                self.predictor=self.model.best_estimator_
            else:
                retrieve_params_cp=copy.deepcopy(retrieve_params)
                wrap_params([retrieve_params_cp])
                self.create_model(
                    self.box, verbose=False, which_models=which_models,
                    wrapped=retrieve_params_cp,
                    kFold=False,
                    refit=True)
                self.predictor=self.model

        # Update the rewards
        hist_update={
            "Number of parameters" : self.length_rewards,
            "Best index" : max_index,
            "Chosen index": index,
            "Best parameters" : retrieve_params,
            "Best score" : -self.rewards [index]
            }

        print_new_config=" "
        for rew_ in self.rewards:
            print_new_config +=str(rew_) + " "
        append_to_log_file(
            self.log_output,
            "\nAll rewards :" + print_new_config,
            verbose=self.verbose)
        # r both
        self.best_index_hist [str(self.number_fits)]=hist_update
        append_to_log_file(
            self.log_output,
            "\nBest estimator: "+str(hist_update),
            verbose=self.verbose)

        if (len(sample_weight) == 0):
            sample_now=np.ones(len(y), dtype=np.float32)
        else:
            sample_now=np.ravel(sample_weight)

        if 'SVR'in name:
            self.predictor.fit(
                X.numpy(), np.ravel(y.numpy()),
                estimator__sample_weight=sample_now)
            X_pred=self.predictor.predict(X.numpy())
            score=-mean_squared_error(X_pred, y.numpy())
        else:
            # ANN!!
            self.predictor.fit(
                X.numpy(),np.ravel(y.numpy()),
                estimator__sample_weight=np.ravel(sample_weight),
                estimator__epochs=epochs, estimator__batch_size=estimator__batch_size,
                estimator__verbose=verbose)
            print("scoring")
            score=- self.predictor.score(
                X.numpy(),np.ravel(y.numpy()))   # IN mlps this quantity is positive,
                # the minus sign hence

        print("score=", score)
        append_to_log_file(
                    self.log_output,
                    "\nscore :" + str(score),
                    verbose=self.verbose)
        self.number_fits +=1 # FIT COUNTER!!

        if multi_armed:
            # Update just the index chosen
            # Note that if multi_armed=False then rewards will never get updated,
            # and each within box epoch will always chose the same models, namely,
            # those with maximum score.
            append_to_log_file(
                    self.log_output,
                    "\nMulti-armed, updating index : "+ str(index),
                    verbose=self.verbose)
            self.rewards[index]=\
                self.rewards[index] +\
                    self.alpha_multi_bandit *\
                       (score - self.rewards[index])

    def predict(self, X):
        try:
            x_pred=X.numpy()
        except:
            x_pred=X
        
        return self.predictor.predict(x_pred)

    #BOX MANAGER
    def update_box_history(self, i): #  details
        """ Update box  """
        self.box_manager.update_box_history(self, i)
    
    def prune_box(self, i, when=["active"], box=[]):
        """ Prune box """
        self.box_manager.prune_box(self, i, when=when, box=box)
    
    def keep_inside_box(self, when=["active"]):
        """ Check if given points are inside the new box """
        self.box_manager.keep_inside_box(self, when=when)
            
    def feed_and_store(self, X, y):
        """Feed data to the model."""
        evaluate_and_store(self, X, y)

################################################################################
# AUXILIARY THINGS FOR ML/LANDSCAPE SKETCHING
################################################################################

def wrap_params(wrapped):
    """
    In LSS_model.py.
    
    """
    logger.info("Creating a dict from wrapped")
    for i,_ in enumerate(wrapped):
        for key in wrapped[i].keys():
            if key=='estimator':
                try:
                    name=str(wrapped[i] [key]).split("(")[0]
                except:
                    name=str(wrapped[i] [key])
                if 'SVR' in name:
                    wrapped[i] [key]='SVR'
                else:
                    wrapped[i] [key]='MLP'
            else:
                wrapped[i] [key]=[wrapped[i] [key]]
    return wrapped

            
class  Normalize(BaseEstimator, TransformerMixin):
    """
    In LSS_model.py.
    
    See https://towardsdatascience.com/
        pipelines-custom-transformers-in-scikit-learn-the
            -step-by-step-guide-with-python-code-4a7d9b068156
    or see Hands-on, page 68
    """
    def __init__(self, box):
        self.box=box

    def fit(self, x, y=None):
        
        return self

    def transform(self,x, y=None):
        X_=np.copy(x)
        
        return (X_ - self.box[0]) /(self.box[1] - self.box[0])


class NormalizationLayer(tf.keras.layers.Layer):
    """
    In LSS_model.py.

    Normalization layer based on Keras model. 
    """
    def __init__(self, box):
        super().__init__()
        self.box=box

    def call(self, X):
        
        return (X - self.box[0]) /(self.box[1] - self.box[0])

    def get_config(self):
        return {**super().get_config(), "box" : self.box}


################################################################################
def create_ann_model(
    hidden_layer_sizes, activation, box, name=""):
    """
    In LSS_model.py.
    
    """
    model=keras.Sequential(name="mlp_"+name)
    model.add(keras.layers.Input(shape=[1]))
    #model.add(NormalizationLayer(box))
    for i,_ in enumerate(activation):
        model.add(
            keras.layers.Dense(
            hidden_layer_sizes[i], 
            activation=activation[i],
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l2(1e-4)))

    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam)
    return model


################################################################################
def model_scikit_wrap(
    model, box, hidden_layer_sizes=[7,1], activation=["elu", "linear"],
    patience=10, restore_best_weights=True, warm_start=True,
    verbose=True, epochs=10):
    
    return KerasRegressor(
        model=model, # Also model=model build_fn
        box=box,
        warm_start=warm_start,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=restore_best_weights,
                monitor="loss")],
        epochs=epochs,
        verbose=verbose)

def gridsearchcv(
    box, cross_val=3, verbose=False,
    which_models=["svr"], wrapped={}, kFold=False,
    refit=True):
    """
    In LSS_model.py.
    
    Create a model by doing grid search

    refit is useless, 
    """
    # id_search using many estimators, see
    #https://stackoverflow.com/questions/51629153/...
    # ...  more-than-one-estimator-in-gridsearchcvsklearn
    # https://stackoverflow.com/questions/50265993/...
    # ...alternate-different-models-in-pipeline-for-gridsearchcv
    cross_val=min(cross_val, 3)
    norm=Normalize(box)
    svr=SVR()
    #mlp=MLPRegressor()  # does not accept sample_weights...
    #mlp=model_scikit_wrap(create_ann_model, box, verbose=verbose)
    pipe=Pipeline(steps=[('norm', norm),('estimator', None)])

    if not wrapped:
        print("first wrapping")
        parameters_mlp={}
        parameters_mlp_low_temp={}
        #['rbf'],#, 'sigmoid'],
        parameters_svr=dict(
            estimator=[svr],
            estimator__kernel=['linear'],
            estimator__gamma=[.1,1,10, 100, 1000],
            estimator__C=[.001, .01, .1,1,10, 100, 1000]
        )
        parameters_all=[]

        if "mlp" in which_models:
            parameters_all.append(parameters_mlp)
        if "svr" in which_models:
            parameters_all.append(parameters_svr)
        if "mlp_low_temp" in which_models:
            parameters_all.append(parameters_mlp_low_temp)

    # if kFold=False, then there's no need for grid_search
    #print("param_all", parameters_all)
    if kFold:
        grid=GridSearchCV(
            estimator=pipe,
            param_grid=parameters_all,
            cv=cross_val,
            scoring='neg_mean_squared_error',
            return_train_score=True,
            refit=True,
            verbose=verbose)#, n_jobs=-1, )
    else:
        if "svr" in which_models:
            svr=SVR(
                kernel=wrapped['estimator__kernel'],
                gamma=wrapped['estimator__gamma'],
                C=wrapped['estimator__C'])

            grid=Pipeline(steps=[('norm', norm),('estimator', svr)])
            
    return grid


################################################################################
# ANN using keras.
################################################################################


def create_ann_model_keras(
    box,
    dimension,
    hidden_layer_sizes,
    activation,
    learning_rate,
    name=""):
    """
    In LSS_model.py.
    
    """
    inputs=keras.layers.Input(shape=[dimension])
    normalization_layer=NormalizationLayer(box)
    X=normalization_layer(inputs)
    
    for i,_ in enumerate(activation):
        X=keras.layers.Dense(
            hidden_layer_sizes[i],
            activation=activation[i],
            kernel_regularizer=keras.regularizers.l2(1e-4),
            bias_regularizer=keras.regularizers.l2(1e-4))(X)

    ##3 Last layers, no activation
    outputs=keras.layers.Dense(
        1,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        bias_regularizer=keras.regularizers.l2(1e-4))(X)

    model=keras.Model(inputs=[inputs], outputs=[outputs], name="ann_"+name)
    
    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    
    return model

def copy_model_and_weights(model_now,box,dimension, params):
    """
    It assumes that the model is a Normalization ANN
    """
    print("Copying the model")
    hidden_layer_sizes=params['hidden_layer_sizes']
    activation=params['activation']
    learning_rate=params ['learning_rate']

    # Setting up similar architecture
    copy_model=create_ann_model_keras(
        box, dimension,
        hidden_layer_sizes, activation,
        learning_rate, name=model_now.name +"_copy")

    # coppying layers
    for i in range(2,len(model_now.layers)):
        copy_model.layers[i].set_weights(model_now.layers[i].get_weights())

    return copy_model


def many_anns(box,dimension, parameters):
    """
    parameters is a list of dictionaries
    """
    models=[]
    params_dict=[]
    for params in parameters:
        estimator=params['estimator']
        hidden_layer_sizes=params['hidden_layer_sizes']
        activation=params['activation']
        learning_rate=params ['learning_rate']

        for i, hid_lay_siz in enumerate(hidden_layer_sizes):
            for actv in activation:
                for lrng_rt in learning_rate:
                    models.append(
                        create_ann_model_keras(
                            box, dimension, hid_lay_siz, actv, lrng_rt, name="_"+str(i))
                    )
                    params_dict.append(
                        dict(
                            estimator=estimator,
                            learning_rate=lrng_rt,
                            hidden_layer_sizes=hid_lay_siz,
                            activation=actv
                        )
                    )
    return models, params_dict


class GridSearchcvKeras():
    """
    In LSS_model.py.
    
    Create a model by doing grid search
    """
    # For grid_search using many estimators, see
    # https://stackoverflow.com/questions/51629153/...
    # ...  more-than-one-estimator-in-gridsearchcvsklearn
    # https://stackoverflow.com/questions/50265993/...
    # ...alternate-different-models-in-pipeline-for-gridsearchcv
    def __init__(
        self, box, cross_val=3, verbose=True,
        which_models=["mlp"], wrapped={},
        patience=10, restore_best_weights=True,
        ml_epochs_gridsearch=10, kFold=False,
        refit=False, dimension=1
    ):

        self.dim=dimension
        self.cross_val=min(cross_val, 3)
        self.refit=refit
        self.verbose=verbose
        self.kFold=kFold
        self.kf=KFold(n_splits=self.cross_val)
        self.epochs_grid_fit=10
        self.ml_epochs_gridsearch=ml_epochs_gridsearch
        self.early_stopping=keras.callbacks.EarlyStopping(
                patience=patience,
                restore_best_weights=restore_best_weights,
                monitor="loss")
        # Normalizer
        self.cv_results_={}
        self.params=[]

        # ANN
        if not wrapped:
            parameters_mlp=dict(
                    estimator="mlp",
                    hidden_layer_sizes=[[5], [6], [7],[8]],
                    activation=[["elu"]],
                    learning_rate=\
                        np.array([1e-3, 1e-4, 1e-1], dtype=np.float32)
            )
            parameters_mlp_low_temp=dict(
                    estimator="mlp_low_temp", 
                    hidden_layer_sizes=[[5,5]],
                    activation=[["elu","elu"]],
                    learning_rate=\
                        np.array([1e-3, 1e-4, 1e-1], dtype=np.float32)
            )
            parameters_all=[]

            if "mlp" in which_models:
                parameters_all.append(parameters_mlp)
            if "mlp_low_temp" in which_models:
                parameters_all.append(parameters_mlp_low_temp)

            ann_models, params=many_anns(box, dimension, parameters_all)
        else:
            ann_models, params=many_anns(box, dimension, [wrapped])

        self.ann_models=ann_models
        self.box=box
        self.cv_results_['params']=params

    def fit(
        self,
        X, y, estimator__sample_weight=[],
        estimator__batch_size=4,
        estimator__epochs=12,
        estimator__verbose=True
    ):
        """
        From https://scikit-learn.org/stable/modules/\
        generated/sklearn.model_selection.KFold.html
        """
        # Normalize FIRST!!!
        x_data_, y_resp_=X, np.ravel(y)

        if self.kFold:            
            split_x_y=self.kf.split(x_data_)
            col=self.cross_val
            scores=np.zeros((len(self.ann_models), col))

            if (len(estimator__sample_weight) == 0):
                estimator__sample_weight=np.ones(len(y_resp_), dtype=np.float32)

            for i, index in enumerate(split_x_y):

                train_index, test_index=index
                x_train=tf.gather(x_data_, indices=train_index)
                x_test=tf.gather(x_data_, indices=test_index)
                y_train=y_resp_[train_index]
                y_test=y_resp_[test_index]
                sample_weights_train=estimator__sample_weight[train_index]
                sample_weights_test=estimator__sample_weight[test_index]

                for model_idx, model_now in enumerate(self.ann_models):

                    # COPY THE MODEL!
                    fit_the_clone=copy_model_and_weights(
                        model_now,self.box, self.dim,
                        self.cv_results_['params'][model_idx])

                    fit_the_clone.compile(
                        optimizer=model_now.optimizer,
                        loss=model_now.loss)
                    fit_the_clone.optimizer.learning_rate=model_now.optimizer.learning_rate.numpy()
                    fit_the_clone.fit(
                        x_train, y_train, sample_weight=sample_weights_train,
                        batch_size=estimator__batch_size,
                        epochs=self.ml_epochs_gridsearch,
                        verbose=estimator__verbose)
                    score=fit_the_clone.evaluate(
                        x_test, y_test, sample_weight=sample_weights_test)                
                    scores[model_idx, i]=score

            # NOTE We store the negative of the scores, because sklearn uses the max
            self.cv_results_['score']=- scores
            self.cv_results_['mean_test_score']=np.mean(self.cv_results_['score'], axis=1)
            self.best_score_=np.max(self.cv_results_['mean_test_score'])
            self.best_index_=np.random.choice(
                np.ravel(np.where(self.best_score_==self.cv_results_['mean_test_score']))
                )
            self.best_params_=self.cv_results_['params'][self.best_index_]
            self.best_estimator_=self.ann_models[self.best_index_]
        else:
            self.best_estimator_=self.ann_models[0]

        if self.refit:
            print("Refitting best model")
            self.best_estimator_.fit(
            x_data_, y_resp_, sample_weight=estimator__sample_weight,
                epochs=estimator__epochs,
                batch_size=estimator__batch_size,
                callbacks=[self.early_stopping])

    def predict(self, x_input):
        """
        Only called for single models
        """
        # You have to ravel other
        return np.ravel(self.best_estimator_.predict(x_input))

    def score(self, x_input, y_response):
        """Only called for single models"""
        return self.best_estimator_.evaluate(x_input,y_response)
