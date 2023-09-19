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
from loguru import logger
from .LSS_print import print_details
from .LSS_connector import\
    append_to_log_file,\
        read_parameter_files
from .LSS_exploratory_policies import\
    simulated_annealing,\
        adjust_sigma_sim_an,\
            epsilon_low_temp_adjust
from .LSS_model import\
    LandscapeSketch,\
        populate,\
            landscape_step,\
                evaluate_and_store,\
                    truncate_agents
from .LSS_callbacks_and_schedules import\
    ShrinkingSchedules,\
        BoxManager,\
            EvaluationCallback,\
                GenerateConfigurations
from .Deep_learning_MD_lib import Extract_v2

##############################################################
##############################################################

#1) Using subprocess with shell=True, when should instead be shell=False
# https://stackoverflow.com/questions/29889074/how-to-wait-for-first-command-to-finish
# https://stackoverflow.com/questions/2837214/
#       python-popen-command-wait-until-the-command-is-finished
# https://stackoverflow.com/questions/23611396/
#       python-execute-cat-subprocess-in-parallel/23616229#23616229 

#2) On Python threading multiple bash subprocesses?
# https://stackoverflow.com/questions/14533458/
#       python-threading-multiple-bash-subprocesses/14533902#14533902
# https://stackoverflow.com/questions/61763029/
#       tensorflow-tf-reshape-causes-gradients-do-not-exist-for-variables
# https://www.programcreek.com/python/example/69298/subprocess.getoutput

################################################################################
################################################################################
#  MAIN FUNCTION -- second layer
#################################################################################
################################################################################
"""
Given several sampled points we'd like to generate a next batch of possible minima.
"""

class LandscapeSketchandStepLayerzero():
    """
    In LSS.py

    This class is responsible for picking up points, 
    evaluating them by simulated annealing, 
    and later using the values to interpolate them.
    """
    def __init__(
        self,
        input_name,
        output_name,
        program_name,
        wrapped_schedules,
        parameters,
        log_output="log_landscape_fiting.txt",
        alpha_wgt_updt=.2,
        bm_step_size_low_temp=20.,
        bm_step_size_high_temp_1=10.,
        bm_step_size_high_temp_2=5.,
        rounds_box_prunning=2,             # EPOCHS
        initial_points=[],
        truncate="chopped",
        relaxation_factor=1, 
        with_configurations=False,
        n_max_measurements=np.inf,
        classical=False,
        ml_epochs_gridsearch=5,
        ml_epochs_refit=30,
        multi_armed=True,
        evaluation_budget=np.inf,
        validation_data=(None, None),
        md_validate=True,
        verbose=False,
        patience=np.inf,
        gridsearch_steps=1,
        save_history_as="history",
        n_expensive_evltns=np.inf,
        eps_multi_armed=.1,
        cross_val=3,
        average_method="mean",
        hungry_for_improvement=5,
        predict_and_plot=False):
        ########################################################################
        # ATTRIBUTES
        self.is_model=False
        if 'initial_box' in parameters.keys():
            self.initial_box=parameters['initial_box']
        else:
            self.initial_box=read_parameter_files()
        self.box=self.initial_box
        self.adjustment_rate=1. # will be used to adjust model
                                # if no improvement is seen
        # Used for adjustment of clustering based on improvement
        self.hungry_for_improvement=hungry_for_improvement
        self.save_history_as=save_history_as
        self.patience=patience
        self.classical=classical
        self.verbose=verbose
        self.gridsearch_steps=gridsearch_steps
        self.multi_armed=multi_armed
        self.ml_epochs_gridsearch=ml_epochs_gridsearch
        self.ml_epochs_refit=ml_epochs_refit
        self.validation_data=validation_data
        self.md_validate=md_validate
        self.rounds_box_prunning=rounds_box_prunning
        self.cross_val=cross_val
        self.input_name=input_name
        self.output_name=output_name
        self.program_name=program_name
        self.log_output=log_output
        self.parameters_output="parameters_output"
        self.eps_multi_armed=eps_multi_armed
        self.which_models=["svr"]
        self.average_method=average_method
        self.predict_and_plot=predict_and_plot

        if 'hungry_for_improvement' in parameters.keys():
            self.hungry_for_improvement=parameters['hungry_for_improvement']
        #-------
        if 'save_history_as' in parameters.keys():
            self.save_history_as=parameters['save_history_as']
        #-------
        if 'patience' in parameters.keys():
            self.patience=parameters['patience']
        
        self.callback=EvaluationCallback(self.patience, self.save_history_as)
        #-------
        if 'classical' in parameters.keys():
            self.classical=parameters['classical']
        #-------
        if 'verbose' in parameters.keys():
            self.verbose=parameters['verbose']
        #-------
        if 'gridsearch_steps' in parameters.keys():
            self.gridsearch_steps=parameters['gridsearch_steps']
        #-------
        if 'multi_armed' in parameters.keys():
            self.multi_armed=parameters['multi_armed']
        #-------
        if 'ml_epochs_gridsearch' in parameters.keys():
            self.ml_epochs_gridsearch=parameters['ml_epochs_gridsearch']
        #-------
        if 'ml_epochs_refit' in parameters.keys():
            self.ml_epochs_refit=parameters['ml_epochs_refit']
        #-------
        if 'validation_data' in parameters.keys():
            self.validation_data=parameters['validation_data']
        #-------
        if 'md_validate' in parameters.keys():
            self.md_validate=parameters['md_validate']
        #-------
        if 'rounds_box_prunning' in parameters.keys():
            self.rounds_box_prunning=parameters['rounds_box_prunning']
        #-------
        if 'cross_val' in parameters.keys():
            self.cross_val=parameters['cross_val']
        # QUERY AND OUTPUT
        if 'input_name' in parameters.keys():
            self.input_name=parameters['input_name']
        #-------
        if 'output_name' in parameters.keys():
            self.output_name=parameters['output_name']
        #-------
        if 'program_name' in parameters.keys():
            self.program_name=parameters['program_name']
        #-------
        if 'log_output' in parameters.keys():
            self.log_output=parameters['log_output']
        #-------
        if 'parameters_output' in parameters.keys():
            self.parameters_output=parameters['parameters_output']
        #-------
        if 'eps_multi_armed' in parameters.keys():
            self.eps_multi_armed=parameters['eps_multi_armed']
        #-------
        if 'which_models' in parameters.keys():
            self.which_models=parameters['which_models']
        #-------
        if 'average_method' in parameters.keys():
            self.average_method=parameters['average_method']  # Max or mean
        #-------
        if 'predict_and_plot' in parameters.keys():
            self.predict_and_plot=parameters['predict_and_plot']
        # SIMULATED ANNEALING PARAMETERS - Constant during box_sessions
        self.bm_step_size_low_temp=bm_step_size_low_temp
        self.bm_step_size_high_temp_1=bm_step_size_high_temp_1
        self.bm_step_size_high_temp_2=bm_step_size_high_temp_2
        self.with_configurations=with_configurations

        if 'bm_step_size_low_temp' in parameters.keys():
            self.bm_step_size_low_temp=parameters['bm_step_size_low_temp']
        #-------
        if 'bm_step_size_high_temp_1' in parameters.keys():
            self.bm_step_size_high_temp_1=parameters['bm_step_size_high_temp_1']
        #-------
        if 'bm_step_size_high_temp_2' in parameters.keys():
            self.bm_step_size_high_temp_2=parameters['bm_step_size_high_temp_2']
        #-------
        if 'with_configurations' in parameters.keys():
            self.with_configurations=parameters['with_configurations']
        # SCHEDULES
        shrinking_sch=ShrinkingSchedules()
        self.box_shrinking=shrinking_sch.constant(1.)  # No shrinking
        if not self.classical and 'box_shrinking' in wrapped_schedules.keys():
            self.box_shrinking=wrapped_schedules['box_shrinking']

        # Vary within box_session
        self.alpha_wgt_updt=alpha_wgt_updt
        self.beta_accept_low_temp=wrapped_schedules['beta_accept_low_temp']
        self.beta_accept_high_temp=wrapped_schedules['beta_accept_high_temp']
        self.eps_low_temp_or_high_temp=.5
        self.sigma_sim_an_high_temp=0.
        # Adjust sigma in the simulated annealing
        adjust_sigma_sim_an(
            self, bm_step_size_low_temp=self.bm_step_size_low_temp,
            bm_step_size_high_temp_1=self.bm_step_size_high_temp_1,
            bm_step_size_high_temp_2=self.bm_step_size_high_temp_2)
        # --------------------------------------------------------------------
        # COUNTERS AND PARAMETERS
        # This will be set as a cap on the number of total evaluations
        self.evaluation_budget=evaluation_budget
        self.on_budget=True
        self.dim=self.initial_box.shape[1]
        self.n_max_measurements=n_max_measurements
        self.truncate=truncate
        # quantities get overwritten if classical=True
        self.rounds_sim_an_high_temp=wrapped_schedules['rounds_sim_an_high_temp']
        self.rounds_sim_an_low_temp=wrapped_schedules['rounds_sim_an_low_temp']
        self.n_min_active_agents=wrapped_schedules['n_min_active_agents']
        self.rounds_within_box_search=wrapped_schedules['rounds_within_box_search']
        #
        if classical:
            # If classical==True, then overwrite N_max_active agents
            self.n_max_active_agents=wrapped_schedules['n_min_active_agents']
        else:
            self.n_max_active_agents=wrapped_schedules['n_max_active_agents']
        if 'n_expensive_evltns' in wrapped_schedules.keys():
            self.n_expensive_evltns=wrapped_schedules['n_expensive_evltns']
        else:
            self.n_expensive_evltns=n_expensive_evltns
        # --------------------------------------------------------------------
        # TRACKERS
        self.active_agents_x=tf.Variable(
            np.empty((0, self.dim)), dtype=tf.float32)
        self.active_agents_y=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.active_param_value_fct=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.past_agents_x=tf.Variable(
            np.empty((0, self.dim)), dtype=tf.float32)
        self.past_agents_y=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.past_param_value_fct=tf.Variable(
            np.empty((0, 1)), dtype=tf.float32)
        self.min_active_agents_x=[]
        self.min_active_agents_y=[]
        self.pred_samples=[]
        self.track_evaluation_counter=[]
        self.evaluation_counter=0
        self.max_amplitude=0.
        # --------------------------------------------------------------------
        # INTERFACE RELATED -- MANAGERS
        self.configurations_manager=None
        self.extractor=None
        self.box_manager=BoxManager(self.initial_box)
        # LOGS
        self.history={}
        # --------------------------------------------------------------------
        self.configurations_manager=GenerateConfigurations(
        n_configurations=1000, test_size=.2,
        val_size=.1, random_state=23)

        if self.with_configurations:
            # Create a configuration manager
            self.configurations_manager=GenerateConfigurations(
            n_configurations=1000, test_size=.2,
            val_size=.1, random_state=23)

            # Create ab initio matrix extractor
            E=Extract_v2()
            E.retrieve_ab_initio_matrix()
            E.split_ab_initio(self.configurations_manager.test)
            self.extractor=E

            # Correcting validation data
            # NOTE: this part suits the MD simulation we are doing,
            # therefore it only contains one entry, denoting the
            # atomic configurations.
            if self.md_validate:
                self.validation_data=\
                    self.configurations_manager.validation_data, None
        # --------------------------------------------------------------------
        self.family_of_models=LandscapeSketch(
            self,
            eps_multi_armed=self.eps_multi_armed,
            ml_epochs_gridsearch=self.ml_epochs_gridsearch,
            patience=self.patience)
        self.exploration_index=[]
        self.relaxation_factor=relaxation_factor
        # ------------------------------------
        # INITIALIZATIONS
        populate(self, initial_points, epoch=0)
        # ------------------------------------
        self.best_x=np.inf
        self.best_y=np.inf
        self.rounds_without_improvement=0
        if classical:
            self.best_x, self.best_y=\
                np.inf * np.ones(self.active_agents_x.shape),\
                    np.inf * np.ones(self.active_agents_x.shape)
            self.rounds_without_improvement=np.zeros(self.best_y.shape)
            
        logger.info("Initializations are done!")
        if classical or self.past_agents_x.shape[0] < 3:
            y_pred=[]
        else:
            y_pred=self.reconstruct_predictor()

        self.initial_points=[
            self.active_agents_x,
            self.active_agents_y,
            self.active_param_value_fct,
            None,
            None
            ]
        self.best_y_old=self.best_y
        self.update_lists(y_pred)
        append_to_log_file(
            self.log_output, "\nLSS uses multi-armed :" +str(self.multi_armed),
            verbose=self.verbose)
        append_to_log_file(
                    self.log_output, "\nMD validate :" +str(self.md_validate),
                    verbose=self.verbose)
        ########################################################################

    def sim_annealing_param_auto_adjustments(self):
        """ Measure of concentration! if close fo zero, good.
        If close to 1, very concentrated"""

        if self.classical or self.past_agents_x.shape[0] < 3:
            if self.past_agents_x.shape[0] < 3 and not self.classical:
                append_to_log_file(
                    self.log_output,
                    "Past agents' list still too small!"\
                        +" Resorting to classical Simulated annealing instead",
                        verbose=self.verbose)
            self.eps_low_temp_or_high_temp=1.
        else:
            # Need to adjust rate based on patience.
            logger.info(
                "\nHungry improvement ",
                self.hungry_for_improvement, self.rounds_without_improvement
                )
            if self.hungry_for_improvement < self.rounds_without_improvement:
                self.adjustment_rate *=.95
                self.adjustment_rate=max(self.adjustment_rate,.8)
            else:
                self.adjustment_rate=min(self.adjustment_rate / .95, 1.)

            append_to_log_file(
                    self.log_output,
                    "Adjustment rate=" +str(self.adjustment_rate),
                    verbose=self.verbose)
            self.eps_low_temp_or_high_temp=epsilon_low_temp_adjust(
                target=self.best_x,
                adjustment_rate=self.adjustment_rate,
                x_data=self.active_agents_x, box=self.box,
                average_method=self.average_method)
            
        self.relaxation_factor=1 - self.eps_low_temp_or_high_temp
        # Need to adjust sigma
        # Adjust sigma in the simulated annealing
        adjust_sigma_sim_an(
            self, eps=self.eps_low_temp_or_high_temp,
            bm_step_size_low_temp=self.bm_step_size_low_temp,
            bm_step_size_high_temp_1=self.bm_step_size_high_temp_1,
            bm_step_size_high_temp_2=self.bm_step_size_high_temp_2)

    def box_search(self, box_search_epoch):
        """
        This function is called only when
            LSS.evaluation_counter < LSS.evaluation_budget
        
        But assumption can be violated in the new_states function
        Begins a box search, running simmulated annealing for a fixed number of 
        steps.
        In the end it updates the list of parameters
        """

        if box_search_epoch > 0:
            # Populate box with previous agents
            # Unlike the initial case (box_search_epoch ==0),
            # in this case there are no costly evaluations to be made
            populate(self, self.initial_points, epoch=box_search_epoch)

        for sub_epoch in range(self.rounds_within_box_search(box_search_epoch)):
            # if you are here then
            # LSS.evaluation_budget > LSS.evaluation_counter
            # Therefore you can sample new elements
            queue_act_states_low_temp,\
                queue_act_states_high_temp,\
                    y_eval=self.new_states(box_search_epoch)
            # Note that y_eval is the true evaluation \theta \mapsto E(\theta)
            # when LSS.classical == True, which corresponds to the
            # Simulated Annealing case
            
            # Notice also that if LSS.classical==False, then
            # LSS.evaluation_budget > LSS.evaluation_counter
            # because no evaluation has been made.
            
            # High eps_policy means more exploration.
            hist_aux=max(
                self.past_agents_x.shape[0],
                self.active_agents_x.shape[0]
                )
            eps_policy=\
               (1 - self.eps_low_temp_or_high_temp)/hist_aux + self.eps_low_temp_or_high_temp *.5

            # Halting!!!
            # If self.classical=False, then recall that
            # LSS.evaluation_budget > LSS.evaluation_counter still holds.
            # In this case, new elements are added to the model through
            # exploratory policy. The budget tracking is then done by
            # the next function.
            # Number of costly_evaluations is defined here!
            landscape_step(
                self, queue_act_states_low_temp, queue_act_states_high_temp,
                y_eval=y_eval,
                box_search_epoch=box_search_epoch,
                sub_epoch=sub_epoch, eps_policy=eps_policy)

            # If there is enough past experience, we are now
            # able to reconstruct the predictor. That's why interpolation is
            # left to the end.
            if self.classical or self.past_agents_x.shape[0] < 3:
                if self.past_agents_x.shape[0] < 3 and not self.classical:
                    append_to_log_file(
                        self.log_output,
                        "Past agents' list still too small!"\
                        +" Resorting to classical Simulated annealing instead",
                        verbose=self.verbose)
                y_pred=[]
            else:
                # The model does a full grid search in the beginning
                # of each box_search andevery box_search_epoch epoch
                y_pred=\
                    self.reconstruct_predictor(
                        gridsearch=\
                           (box_search_epoch % self.gridsearch_steps==0) and(sub_epoch==0))

            self.update_lists(
                y_pred,
                epoch=sub_epoch,
                logs=self.history[str(box_search_epoch)])

            # check budget condition
            if not self.on_budget:
                break
        
    def new_states(self, box_search_epoch, x_agents=None):
        """
        Adjust parameters before box_search.

        Run simulated annealing and add the points to the vector of sampled points 
        """
        # Parameter auto-adjustments
        self.sim_annealing_param_auto_adjustments()

        # 'beta_accept_high_temp' and beta_accept_low_temp are cooling schedules
        # but high_temperature is subject to parameterization by the concentration
        # factor.
        beta_accept_high_temp=self.beta_accept_high_temp(
                box_search_epoch,
                concentration_parameter=self.eps_low_temp_or_high_temp,
                max_amplitude=self.max_amplitude)

        beta_accept_low_temp=self.beta_accept_low_temp(box_search_epoch)
        logger.info("\nAdjusting parameters before simulated annealing.")

        if not x_agents:
            sample_low_temp=np.copy(self.active_agents_x)
            sample_high_temp=np.copy(self.active_agents_x)
        else:
            sample_low_temp=np.copy(x_agents)
            sample_high_temp=np.copy(x_agents)

        if self.classical or self.past_agents_x.shape[0] < 3:

            if self.past_agents_x.shape[0] < 3 and not self.classical:
                # Not enough past history to landscape sketch
                append_to_log_file(
                    self.log_output,
                    "Past agents' list still too small!"\
                        +" Resorting to classical Simulated annealing instead",
                        verbose=self.verbose)
            rounds_sim_an_high_temp_this_epoch=0
            rounds_sim_an_low_temp_this_epoch=1
        else:
            rounds_sim_an_high_temp_this_epoch=\
                self.rounds_sim_an_high_temp(box_search_epoch)
            rounds_sim_an_low_temp_this_epoch=\
                self.rounds_sim_an_low_temp(box_search_epoch)

        # Register on history
        hist_update={
          "rounds_sim_an_high_temp_this_epoch" : rounds_sim_an_high_temp_this_epoch,
          "rounds_sim_an_low_temp_this_epoch" : rounds_sim_an_low_temp_this_epoch
        }
        self.history[str(box_search_epoch)].update(hist_update)

        # Register on log
        append_to_log_file(
            self.log_output, str(hist_update), verbose=self.verbose)

        y_eval=[]

        # HIGH TEMPERATURE SIMULATED ANNEALING
        for _ in range(rounds_sim_an_high_temp_this_epoch):
            # HALTING due to budget!!!
            sample_high_temp_new, _=simulated_annealing(
                self, sample_high_temp, beta_accept=beta_accept_high_temp,
                sigma_sim_an=self.sigma_sim_an_high_temp,
                classical=self.classical)

            if sample_high_temp_new == None: # in case it it 'None'
                logger.info("You went above the budget!")
                break
            else:
                sample_high_temp=sample_high_temp_new

        # LOW TEMPERATURE SIMULATED ANNEALING
        for _ in range(rounds_sim_an_low_temp_this_epoch):
            # HALTING!!!
            sample_low_temp_new, y_eval=simulated_annealing(
                self, sample_low_temp, beta_accept=beta_accept_low_temp,
                sigma_sim_an=self.sigma_sim_an_low_temp,
                classical=self.classical)

            if sample_low_temp_new == None: # in case it is None
                logger.info("You went above the budget!")
                break
            else:
                sample_low_temp=sample_low_temp_new

        # NOTE if classical==True, then y_eval is evaluated by run_and_retrieve,
        # otherwise it is[]

        return sample_low_temp, sample_high_temp, y_eval

    def reconstruct_predictor(self, gridsearch=False, batch_size=4):
        """
        Fit the model on active_agents_x and active_agents_y
        """
        append_to_log_file(
            self.log_output, "Reconstructing model", verbose=self.verbose)

        self.family_of_models.fit(
            which_models=self.which_models,
            epochs=self.ml_epochs_refit,
            multi_armed=self.multi_armed,
            gridsearch=gridsearch,
            estimator__batch_size=batch_size
            )
        self.predictor=self.family_of_models.predictor
        y_pred=None
        
        return y_pred

    def update_lists(self, y_pred, epoch=0, logs={}):
        """
        Update best values, min_trackers, and history

        It is here that the model gets truncated
        (by re introduction of best_min to active agents)
        """
        if self.classical:
            where_y_min=tf.where(
                self.active_agents_y < self.best_y,
                self.active_agents_y, self.best_y)
            where_x_min=tf.where(
                self.active_agents_y < self.best_y,
                self.active_agents_x, self.best_x)

            self.best_x, self.best_y=where_x_min, where_y_min
            self.min_active_agents_x.append(self.best_x)
            self.min_active_agents_y.append(self.best_y)
            self.track_evaluation_counter.append(self.evaluation_counter)

            # Tracking improvement
            self.rounds_without_improvement=tf.where(
                self.best_y_old > self.best_y,
                self.rounds_without_improvement+1, 0)
            self.best_y_old=self.best_y
        else:
            y_min_candidate=tf.reduce_min(self.active_agents_y, keepdims=True)
            where_y_min=tf.argmin(self.active_agents_y)
            x_min_candidate=tf.gather(
                self.active_agents_x, indices=where_y_min)

            if tf.squeeze(y_min_candidate).numpy() < tf.squeeze(self.best_y).numpy():
                self.best_x, self.best_y=x_min_candidate, y_min_candidate

            # Tracking improvement
            logger.info("Best old, Best new", self.best_y_old, self.best_y)
            if self.best_y_old > self.best_y:
                logger.info("Zeroying")
                self.rounds_without_improvement=0
            else:
                self.rounds_without_improvement +=1

            self.best_y_old=self.best_y
            self.min_active_agents_y.append(self.best_y.numpy())
            self.min_active_agents_x.append(self.best_x.numpy())
            self.track_evaluation_counter.append(self.evaluation_counter)

            # Keep minimum among the active agents
            # if all(self.best_x not in self.active_agents_x):
            # This works well with numpy, not with tensorflow
            if tf.reduce_all(tf.reduce_any(self.best_x !=self.active_agents_x, axis=1)):
                # HALTING!!!
                evaluate_and_store(
                    self, self.best_x, tf.reshape(self.best_y,(1,1)),
                    epoch=epoch)
                truncate_agents(self, epoch)

        self.pred_samples.append(
            (self.active_agents_x, self.active_agents_y,
            self.active_param_value_fct, y_pred, self.exploration_index)
            )

        # Updating logs
        if logs:
            hist_update={
            "Epoch "+str(epoch) +" - eps_low_temp_or_high_temp" : 
                self.eps_low_temp_or_high_temp, 
            "Epoch "+str(epoch) +" - relaxation_factor" : 
                self.relaxation_factor,
            "Epoch "+str(epoch) +" - sigma_sim_an_high_temp" :
                np.squeeze(self.sigma_sim_an_high_temp.numpy()),
            "Epoch "+str(epoch) +" - sigma_sim_an_low_temp" :
                np.squeeze(self.sigma_sim_an_low_temp.numpy()),
            "Epoch "+str(epoch) +" -(best_x, best_y)" :
                tf.concat((self.best_x, self.best_y), axis=1).numpy(),
            "Epoch "+str(epoch) +" - rounds_without_improvement" :
                self.rounds_without_improvement
            }
            logs.update(hist_update)
    # FROM BOX MANAGER
    def update_box_history(self, i): # SAVE details
        """ Update box  """
        self.box_manager.update_box_history(self, i)

    def prune_box(self, i, when=["active"]):
        """ Prune box """
        self.box_manager.prune_box(self, i, when=when)

    def keep_inside_box(self, when=["active"]):
        """ Check if given points are inside the new box """
        self.box_manager.keep_inside_box(self, when=when)

    def new_configuration(self):
        """ Generate configurations """
        
        return self.configurations_manager.new_configuration()

    def fitting_counter(self):
        """Count the number of times the model has been fit"""
        
        return self.family_of_models.number_fits

################################################################################
################################################################################
#  MAIN FUNCTION
################################################################################
################################################################################

class LandscapeSketchandStep(LandscapeSketchandStepLayerzero):
    """
    In LSS.py 
    TODO

    """
    def __init__(
        self,
        input_name,
        output_name,
        program_name,
        wrapped_schedules,
        parameters={},
        initial_points=[]):

        # Checking schedules
        for key_sch in wrapped_schedules.keys():
            if wrapped_schedules[key_sch].__class__==int:
                raise ValueError(
                    "The quantity "\
                        +key_sch +" is not a schedule, but an int!!")

        logger.info("All schedules are valid!")
        self.running_time=0.0
        self.initial_points=initial_points
        super().__init__(
            input_name, output_name, program_name,
            wrapped_schedules,
            parameters,
            initial_points=initial_points)

        self.box_prunning_iter_so_far=0 # Number of searches carried out so far
        self.box_history={}

    def full_search(self):
        """TODO"""
        # CALLBACK
        self.callback.on_train_begin(self)
        print_details(self)
        if self.box_prunning_iter_so_far==0:
            number=self.rounds_box_prunning
        else:
            # In this case, if the model is called again, ony one round of 
            # searches is carried out
            number=1
            
        # We only iterate in the following way because we also want to call the
        # model after a few search rounds and have it to continue from where it
        # was left
        for i in range(
            self.box_prunning_iter_so_far, self.box_prunning_iter_so_far + number):

            # CALLBACK and search
            if self.callback.on_box_search_begin(self, box_search_epoch=i):
                # The model is on budget or Early stopping(if apply)
                # has not been reached
                print_message=\
                    "\n"+40*"="+"\nBegin "+str(i)+"-th box search"+"\n"+40*"="
                append_to_log_file(
                    self.log_output, print_message, verbose=self.verbose)
                # Do a box search and reconstruct the predictor.
                # This is done rounds_within_box_search times at each box_search
                self.box_search(i)
                # Saving further details, like the queue of active states
                self.update_box_history(i)
                ################################################################
                
                
                ################################################################
                # evaluate_at_validation_data(self, box_search_epoch=i)
                # Prunning past_agents, those used for fitting purposes.
                self.prune_box(i)
                self.family_of_models.prune_box(i, box=self.box)
                self.keep_inside_box(when=["past"])
                self.family_of_models.clean_up_active_box()
                # CALLBACK
                self.callback.on_box_search_end(self, box_search_epoch=i)
            else:
                if self.evaluation_counter >= self.evaluation_budget:
                    append_to_log_file(
                        self.log_output,
                        "\nNO MORE EVALUATION BUDGET!", verbose=True)
                else:
                    append_to_log_file(
                        self.log_output, "\nEARLY STOPPING!", verbose=True)
                break

        self.box_prunning_iter_so_far +=number
        # CALLBACK
        self.callback.on_train_end(self)
        