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

import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=[FutureWarning,Warning])
import numpy as np
from loguru import logger
import LIBS.LSS as LSS
# https://stackoverflow.com/questions/15777951/...
# ...how-to-suppress-pandas-future-warning
try: # In order to open and save dictionaries
    import cPickle as pickle
except ImportError:  # python 3.x
    import pickle

DIMENSIONS=int(sys.argv[1])
FOLDER_NOW=str(DIMENSIONS)
os.makedirs(FOLDER_NOW, exist_ok=True)
os.chdir(FOLDER_NOW)
INPUT_NAME="input_name"
OUTPUT_NAME="output_name"
PROGRAM_NAME="../query.py"

#########################################################
# WRAPPING SCHEDULES
#########################################################
ROUNDS_BOX_PRUNNING=5
S=LSS.ShrinkingSchedules()

cooling=S.algebraic_decay(
    T_init=1., T_end=.4,
    N_steps=20, kappa=1)
box_shrinking=S.constant(1)# S.linear_decay(1, .9, N_steps=5)
beta_accept_high_temp=S.algebraic_decay(
    T_init=.5, T_end=1, N_steps=ROUNDS_BOX_PRUNNING)
beta_accept_low_temp=S.algebraic_decay(
    T_init=5, T_end=4, N_steps=ROUNDS_BOX_PRUNNING)
n_max_active_agents=S.rounds(
    3, 3, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
n_min_active_agents=S.constant(3)
n_expensive_evltns=S.rounds(
    2,2, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
rounds_sim_an_low_temp=S.rounds(
    5, 3, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
rounds_sim_an_high_temp=S.rounds(
    5, 4, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
rounds_within_box_search=S.constant(20)

wrapped_schedules=dict(
    box_shrinking=box_shrinking,
    beta_accept_high_temp=beta_accept_high_temp,
    beta_accept_low_temp=beta_accept_low_temp,
    n_max_active_agents=n_max_active_agents,
    n_min_active_agents=n_min_active_agents,
    n_expensive_evltns=n_expensive_evltns,
    rounds_sim_an_low_temp=rounds_sim_an_low_temp,
    rounds_sim_an_high_temp=rounds_sim_an_high_temp,
    rounds_within_box_search=rounds_within_box_search
)
#########################################################################
# simulated annealing experiments
##########################################################################

N_EXPERIMENTS=50
logger.info(
    f"\nBeginning study in dimension {DIMENSIONS}"+
    "\n"+40*"-"+
    "\nToy problem using Simulated annealing"+
    "\n"+40*"-")

# Create initial box
X=.1 * tf.ones((N_EXPERIMENTS,DIMENSIONS), dtype=tf.float32)
initial_points=X
lower=tf.zeros(shape=(1, DIMENSIONS))
upper=tf.ones(shape=(1, DIMENSIONS))
initial_box=tf.concat((lower,upper), axis=0)

for step in[12.5, 25, 50]:
    log_output=f"log_output_toy_model_{DIMENSIONS}"+\
        f"D_sim_ann_step_{str(step).replace('.','__')}.txt"
    save_history_as=f"history_toy_model_{DIMENSIONS}"+\
        f"D_sim_ann_step_{str(step).replace('.','__')}"
    parameters=dict(
        bm_step_size_low_temp=step,
        bm_step_size_high_temp_1=25,
        bm_step_size_high_temp_2=12.5,
        truncate="brute",
        rounds_box_prunning=ROUNDS_BOX_PRUNNING,
        relaxation_factor=.5,
        classical=True,
        patience=np.inf,
        log_output=log_output,
        evaluation_budget=np.inf,
        save_history_as=save_history_as,
        initial_box=initial_box)

    F=LSS.LandscapeSketchandStep(
        INPUT_NAME, OUTPUT_NAME, PROGRAM_NAME,
        wrapped_schedules,
        parameters,
        initial_points=initial_points)

    F.full_search()
    F_min_sa=tf.concat(F.min_active_agents_y, axis=1).numpy()
    filename=\
        f"toy_model_{DIMENSIONS}D_100_sim_annealing_step_{str(step).replace('.','__')}.pickle"
    logger.info(f"\nSaving pickled file as  {filename}")

    with open(filename, 'wb') as save:
        pickle.dump(F_min_sa, save, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"\nDone with simulated annealing at step {step}")

#########################################################################
# Landscape sketch and step experiments - SVR
##########################################################################
logger.info("\n"+40*"-"+
            "\nToy problem using Landscape sketch and step method"+
            "\n"+40*"-")

##### SCHEDULES
ROUNDS_BOX_PRUNNING=5
S=LSS.ShrinkingSchedules()
cooling=S.algebraic_decay(
    T_init=1., T_end=.4, N_steps=20, kappa=1)
box_shrinking=S.constant(1)# S.linear_decay(1, .9, N_steps=5)
beta_accept_high_temp=S.algebraic_decay(
    T_init=.5, T_end=1, N_steps=ROUNDS_BOX_PRUNNING)
beta_accept_low_temp=S.algebraic_decay(
    T_init=5, T_end=4, N_steps=ROUNDS_BOX_PRUNNING)
n_max_active_agents=S.rounds(
    5, 5, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
n_min_active_agents=S.constant(3)
n_expensive_evltns=S.rounds(
    2,1, rounds_box_prunning=ROUNDS_BOX_PRUNNING, decay_step_every=5)
rounds_sim_an_low_temp=S.rounds(
    5, 3, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
rounds_sim_an_high_temp=S.rounds(
    5, 4, rounds_box_prunning=ROUNDS_BOX_PRUNNING)
rounds_within_box_search=S.constant(20)

wrapped_schedules=dict(
    box_shrinking=box_shrinking,
    beta_accept_high_temp=beta_accept_high_temp,
    beta_accept_low_temp=beta_accept_low_temp,
    n_max_active_agents=n_max_active_agents,
    n_min_active_agents=n_min_active_agents,
    n_expensive_evltns=n_expensive_evltns,
    rounds_sim_an_low_temp=rounds_sim_an_low_temp,
    rounds_sim_an_high_temp=rounds_sim_an_high_temp,
    rounds_within_box_search=rounds_within_box_search
)

for  ML_model in['svr', 'mlp']:
    for multi_armed in[True, False]:
        logger.info(
            "\n"+20*"-"+
            "\nStarting simulations"+
            f"\tML model : {ML_model} \t Multi armed : {multi_armed}")

        experiments={}
        parameters=dict(
            bm_step_size_low_temp=50,
            bm_step_size_high_temp_1=25,
            bm_step_size_high_temp_2=12.5,
            truncate="brute",
            rounds_box_prunning=ROUNDS_BOX_PRUNNING,
            relaxation_factor=.5,
            classical=False,
            patience=np.inf,
            multi_armed=multi_armed,
            ml_epochs_gridsearch=5,
            ml_epochs_refit=20,
            evaluation_budget=np.inf,
            which_models=[ML_model],
            initial_box=initial_box)

        for i in range(N_EXPERIMENTS):
            logger.info(
                f"""\n--------------------------
                \n\n\t Experiment #{i}
                \n--------------------------""")
            parameters['log_output']=f"log_output_toy_model_{DIMENSIONS}D_"\
                +f"{ML_model}_multi_armed_{multi_armed}_exp_{i}.txt"
            parameters['parameters_output']=\
                f"toy_model_{DIMENSIONS}D_parameters_{ML_model}_multi_armed_"+\
                    f"{multi_armed}_exp_{i}"
            parameters['save_history_as']=f"history_toy_model_{DIMENSIONS}D_lss_"\
                    +f"{ML_model}_multi_armed_{multi_armed}_exp_{i}"
            F=LSS.LandscapeSketchandStep(
                INPUT_NAME, OUTPUT_NAME, PROGRAM_NAME,
                wrapped_schedules,
                parameters,
                initial_points=initial_points)
            F.full_search()
            experiments[str(i)]=(
                tf.concat(F.min_active_agents_x,axis=0),
                tf.concat(F.min_active_agents_y, axis=0))

        filename=f"{N_EXPERIMENTS}toy_model_{DIMENSIONS}D_lss_"\
                + f"{ML_model}_multi_armed_{multi_armed}.pickle"
        logger.info(f"Saving pickled file as {filename}")

        with open(filename, 'wb') as save:
            pickle.dump(experiments, save, protocol=pickle.HIGHEST_PROTOCOL)

logger.info(f"\n Done with dimension {DIMENSIONS} !")
