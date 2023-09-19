################################################################################
# HOW TO USE
################################################################################
# This is library 
# with all the main functions used by the programs
# Deep_Learning_MD_main.py and Deep_Learning_MD_train_test.py
###
################################################################################

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import random
import pickle
import os

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
    logger.info("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
################################################################################

# TO READ PARAMETERS

def is_out_of_the_box(X, lower, upper):
    return any(X-lower <0) or any(upper -X <0)

def read_parameter_files():
    """
    Read parameters from txt file
    """
    lower=\
        tf.reshape(
        tf.constant(np.loadtxt("Box_lower_parameters.txt", dtype=np.float32)),
       (-1, 1))        
    upper=\
        tf.reshape(
        tf.constant(np.loadtxt("Box_upper_parameters.txt", dtype=np.float32)),
       (-1, 1))
        
    box=tf.concat([lower, upper], axis=1)
    return box


class CreateParameterFiles():
    
    def __init__(self, lower, upper, eps):
        
        # The fixed eps, that will be the maximum value of each entry
        
        self.lower=lower 
        self.upper=upper
        self.eps=eps
        
    def NN_output_to_parameter_files(self, k):
        
        for j in range(k):
            # Plus eps...
            filename_now="/parameters"+str(j + 1)+".txt"
            f_now=open(filename_now,"w")
            np.savetxt(f_now, output.numpy(), delimiter='\n')
            f_now.close()
        
################################################################################
# Compute finite differences and approximate the norm difference derivative
# w.r.t training parameters

class FiniteDifLinalg():

    def __init__(self, n_atoms, eps_left, eps_right, rdf=False):
        
        self.N=n_atoms  # N is the parameter dimensions
        #self.N_atoms=832
        self.Extractor=Extract(rdf=rdf)
        self.eps_left=eps_left
        self.eps_right=eps_right
    
    def update_epss(
        self, eps_left, eps_right):
        self.eps_left=eps_left
        self.eps_right=eps_right
    
    def l2_norm(self, C_theta, q):
        """
        Inputs are all numpy matrices.
        Due to loss of significan digits, the computations will be as 
        pointed out in the worklog remark
        """
        
        aux=C_theta - q
        N_atoms=C_theta.shape[0]
        
        I_1=tf.reduce_sum(tf.multiply(aux, aux))
        
        return (1 / N_atoms) * I_1


################################################################################
# TO FREEZE THE PARAMETERS
class Stop():
    def __init__(self):
        pass
    
    def freeze(
        self, model, lower, upper, x_initial, x_final,
        eps, hist_metric, picklename="frozen_parameters.p",
        nn_weights_name="NN_MD_frozen_weights.h5"
        ):
        parameters={
            "lower": lower,
            "upper": upper,
            "initial_parameters": x_initial, 
            "final_parameters": x_final, 
            "eps": eps,
            "hist_metric": hist_metric
        }

        logger.info(f"Saving parameters in pickle file {picklename}")    
        pickle.dump(parameters, open(picklename, "wb"))
        if model !=None:
            logger.info(f"Saving model weights in h5 file {nn_weights_name}")   
            model.save_model_weights(nn_weights_name)

    def unfreeze(self, model_included=True):
        """
        TODO
        """
        parameters=pickle.load(open("frozen_parameters.p", "rb" ))
        lower=parameters["lower"]
        upper=parameters["upper"]
        eps=parameters["eps"]
        hist_metric=parameters["hist_metric"]
        x_initial=parameters["initial_parameters"]
        x_final=parameters["final_parameters"]

        if model_included:
            M=CreateModel(lower=lower, upper=upper)
            M.load_model_weights()
            return M, lower, upper, eps, x_initial, x_final, hist_metric
        else:
            return None, lower, upper, eps, x_initial, x_final, hist_metric

################################################################################
# TO CREATE THE MODEL

class CreateModel():
    """
    For bias initialization, see
    https://keras.io/api/layers/initializers/
    """
    def __init__(self, lower, upper, p=0, Ising=False):
        """
        TODO
        """
        if Ising:
            N=lower.shape[0]
            self.N=N
            model=keras.models.Sequential(
           [NormalizationLayer(lower=lower, upper=upper, input_shape=(N,)),
            keras.layers.Dense(
                N, activation="relu", input_shape=(N,),
                bias_initializer=keras.initializers.RandomNormal
                ),
            keras.layers.Dropout(rate=p),
            #keras.layers.Dense(
            #    N, activation="relu", input_shape=(N,),
            #    bias_initializer=keras.initializers.RandomNormal
            #    ),
            #keras.layers.Dropout(rate=p),
            keras.layers.Dense(N, activation=tf.nn.tanh),
            DenormalizationLayer(lower=lower, upper=upper, input_shape=(N,))
            ])
            model.initializers=keras.initializers.RandomUniform
            model.compile() #  loss=keras.losses.MeanSquaredError(), optimizer="Nadam")
        else:  # Not Ising
            N=lower.shape[0]
            self.N=N
            model=keras.models.Sequential(
           [NormalizationLayer(lower=lower, upper=upper, input_shape=(N,)),
            keras.layers.Dense(
                N, activation="relu", input_shape=(N,)
                ),
            keras.layers.Dropout(rate=p),
            #keras.layers.Dense(
            #    N, activation="relu", input_shape=(N,)
            #    ),
            #keras.layers.Dropout(rate=p),
            keras.layers.Dense(N, activation=tf.nn.tanh),
            DenormalizationLayer(lower=lower, upper=upper, input_shape=(N,))
            ])
            model.initializers=keras.initializers.RandomUniform
            model.compile() #  loss=keras.losses.MeanSquaredError(), optimizer="Nadam")

        self.model=model

    def save_model_weights(self, model_name="NN_MD_frozen_weights.h5"):
        """
        TODO
        """
        self.model.save_weights(model_name)

    def load_model_weights(self, model_name="NN_MD_frozen_weights.h5"):
        """
        TODO
        """
        
        return self.model.load_weights(model_name)

    def finite_diff_chain_rule(self, finite_diff_vec, gradients):
        """
        gradients is a constant tensor, it cannot be modified.
        That's why we create a Variable copy of it"""
        Variable_gradient=[tf.Variable(a) for a in gradients]
       
        for grad in Variable_gradient:

            if len(grad.shape)==4: # is weights
                for j in range(self.N):
                    grad[0, j, :, :].assign(grad[0, j, :, :] * finite_diff_vec[j])
            if len(grad.shape)==3: # is bias
                for j in range(self.N):
                    grad[0, j, :].assign(grad[0, j, :] * finite_diff_vec)
        
        return Variable_gradient

################################################################################
class Normalization():
    def __init__(self, l, u):
        """ l and u are assumed to be row vectors """
        self.lower=tf.reshape(l,(1, -1))
        self.upper=tf.reshape(u,(1, -1))
        self.diff=(self.upper - self.lower)/2
        self.bias=(self.upper + self.lower)/2
        self.diff_inv=1 / self.diff
    
    def denormalization(self, x):
        """
        Maps the interval[-1, 1]^k to the interval[l, u]
        """
        answer=self.bias + x * self.diff 
        # enforcing lower and upper boundaries(has to be done due to floating point errors)
        answer=tf.reduce_max((answer, self.lower), axis=0)
        answer=tf.reduce_min((answer, self.upper), axis=0)
        
        return answer
    
    def Normalization(self, x):
        """
        Maps the interval interval[l, u] to the[-1, 1]^k 
        """
        
        return (x - self.bias) * self.diff_inv

################################################################################
class DenormalizationLayer(keras.layers.Layer):
    """
    Maps the interval[-1, 1]^k to the interval[l, u]
    """
    def __init__(self, lower, upper, **kwargs):
        """ l and u are assumed to be row vectors """
        super().__init__(**kwargs)
        self.norm=Normalization(l=lower, u=upper)
    
    def call(self, inputs):
        Z=inputs
        Z=self.norm.denormalization(Z)
        
        return Z   
    
    def  compute_output_shape(self, batch_input_shape):
        b1, b2=batch_input_shape
        return [b1, b2]

##############################################################################    
class NormalizationLayer(keras.layers.Layer):
    """
    Maps the interval interval[l, u] to the[-1, 1]^k 
    """
    def __init__(self, lower, upper, **kwargs):
        """ l and u are assumed to be row vectors """
        super().__init__(**kwargs)
        self.norm=Normalization(l=lower, u=upper)
    
    def call(self, inputs):
        Z=inputs
        Z=self.norm.Normalization(Z)

        return Z 
    
    def  compute_output_shape(self, batch_input_shape):
        b1, b2=batch_input_shape
        
        return [b1, b2]

################################################################################
class MetricDeepMD(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        """
        TODO
        """
        super().__init__(**kwargs)
        self.total=self.add_weight("total", initializer="zeros")
        self.count=self.add_weight("count", initializer="zeros")
    
    def update_state(self, dist, sample_weight=None):
        """
        TODO
        """
        self.total.assign_add(dist)
        self.count.assign_add(1)

    def result(self):
        """
        TODO
        """
        
        return self.total / self.count
    
    def get_config(self):
        """
        TODO
        """
        base_config=super().get_config()
        
        return base_config

################################################################################
# Extract force.dat

class Extract():

    def __init__(self, rdf=False):
        """
        TODO
        """
        self.rdf=rdf
    
    def retrieve_rdf(self, folder_name="ab_initio"):
        """
        TODO
        """
        if folder_name=="ab_initio":
            filename="../../ab_initio/rdf_FPMD.dat"
            
            df=pd.read_csv(
                filepath_or_buffer=filename,
                sep=" ", engine='python', skiprows=0,
                names=['x','R_x'], delimiter="\s+")
            matrix=df.to_numpy()
        else:
            filename=os.path.join(str(folder_name),"rdf.dat")
            df=pd.read_csv(
                filepath_or_buffer=filename,
                sep=" ", engine='python', skiprows=4,
                index_col=False, header=None, delimiter="\s+")
            matrix=df.to_numpy()[:,[1,2]]
        
        matrix_col0, matrix_rest=matrix[:,0], matrix[:,1]
    
        return tf.constant(matrix_col0,dtype=tf.float32, shape=(len(matrix_col0),1)),\
            tf.constant(matrix_rest,dtype=tf.float32, shape=(len(matrix_col0),1))

    def force_matrix_linalg(self, just_zero=False):
        """
        Given the path of a dat file, returns a tensor with it's
        content.
        """
        if self.rdf:
            
            max_range=1 if just_zero else 15
            
            # Using radial distribution function
            matrix_aux=tf.Variable(tf.zeros((1000,1,max_range)), dtype=tf.float32)
            
            for i in range(max_range):
            # READ CLASSICAL FOLDER
                x, y=self.retrieve_rdf(str(i))
                matrix_aux[:,:,i].assign(
                    tf.Variable(y, dtype=tf.float32, shape=(1000,1))
                    )
        else:
            if just_zero:
                max_range=1
            else: 
                max_range=15

            # OLD METHOD< FRO FORCE
            matrix_aux=tf.Variable(tf.zeros((832,3,max_range)), dtype=tf.float32)
                
            for i in range(max_range):
            # READ CLASSICAL FOLDER
                filename=str(i)+"/force.dat"
                df=pd.read_csv(
                    filepath_or_buffer=filename,
                    sep=" ", engine='python', skiprows=9,
                    names=['id','fx','fy', 'fz'], delimiter="\s+")
                df.drop(['id'],axis=1,inplace=True)
                matrix_aux[:,:,i].assign(
                    tf.Variable(df.to_numpy()[:,:], dtype=tf.float32, shape=(832,3))
                    )
            
        return matrix_aux


    def ab_initio_matrix(self):
        """
        TODO
        """
        if self.rdf:
            filename="../../ab_initio/rdf_FPMD.dat"
            df=pd.read_csv(
                filepath_or_buffer=filename,
                sep=" ", engine='python', skiprows=0,
                names=['x','R_x'], delimiter="\s+")
            matrix_aux=tf.Variable(
                df.to_numpy()[:,:,np.newaxis],
                dtype=tf.float32,shape=(1000,2,1)
                )
        else:
            df=pd.read_csv(
                filepath_or_buffer="../../ab_initio/force.dat",
                sep=" ", engine='python', skiprows=0,
                names=['fx','fy', 'fz'], delimiter="\s+"
                )
            matrix_aux=tf.Variable(
                df.to_numpy()[:,:,np.newaxis],
                dtype=tf.float32,shape=(832 *1000,3,1)
                )
            
        return matrix_aux

    
################################################################################
# Minibatches, evaluation on test sets, and cross validation
# Minibatches, evaluation on test sets, and cross validation
def create_minibatches(ab_initio_matrix, number_atoms_per_config=832):
    """
    TODO
    """
    
    ab_initio_length=ab_initio_matrix.shape[0]
    minibatches=np.arange(int(ab_initio_length / number_atoms_per_config))
    random.shuffle(minibatches)
    # SEE #https://www.geeksforgeeks.org/python-ways-to-shuffle-a-list/

    return minibatches

def train_test_split_configurations(ab_initio_matrix, test_size=0.2, random_state=None):
    """ 
    For train test split, see 
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    minibatches=create_minibatches(ab_initio_matrix)
    train, test=\
        train_test_split(
            minibatches, test_size=test_size, random_state=random_state
            )
    
    return list(train), list(test)

def K_fold_MD(train_minibatch, cross_val=5, random_state=None):
    """
    It will create the cross_val minibatches for cross_val.
    For more information, see 
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    minibatch is a LIST!!
    """
    
    ab_initio_length=len(train_minibatch)
    aux=np.arange(ab_initio_length)
    K_fold=KFold(n_splits=cross_val, random_state=random_state)
    train_minibatch_np=np.asarray(train_minibatch)
    Cross_fold_sets=[]
    
    for train_id, test_id in K_fold.split(aux):
        Cross_fold_sets.append(
           (train_minibatch_np[np.asarray(train_id)], train_minibatch_np[np.asarray(test_id)])
            )

    return Cross_fold_sets  # This is a generator

def K_fold_MD_non_shared(train_minibatch, cross_val=5, random_state=None):
    """
    It will create the cross_val minibatches for cross_val.
    For more information, see 
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    minibatch is a LIST!!
    """ 
    ab_initio_length=len(train_minibatch)
    aux=np.arange(ab_initio_length)
    np.random.seed(random_state)
    np.random.shuffle(aux)
    np.random.seed(None)
    splittings=np.array_str(aux, cross_val)
    
    train_minibatch_np=np.asarray(train_minibatch)
    Cross_fold_sets=[]
    
    for split in splittings:
        train_id, test_id=train_test_split(split, test_size=1./cross_val, random_state=random_state)
        Cross_fold_sets.append(
           (train_minibatch_np[np.asarray(train_id)], train_minibatch_np[np.asarray(test_id)])
            )

    return Cross_fold_sets  # This is a generator

def little_evaluation(ab_initio_matrix, test, force):
    """
    TODO
    """
    N_ATOMS=832
    E=Extract()
    C_theta=\
            E.force_matrix_linalg(just_zero=True)  # UPDATE matrix
    test_results=[]
    
    for configuration in test:
        range_atoms=np.arange(
            configuration * N_ATOMS,(configuration + 1) * N_ATOMS
            )
        q=tf.gather_nd(
            ab_initio_matrix, indices=range_atoms[:,np.newaxis]
            )
        
        test_results.append(force.l2_norm(C_theta, q))
    
    return tf.reduce_mean(test_results)

def exponential_lr_schedule(decay, every):
    """
    TODO
    """
    def lr_schedule_at_epoch(epoch):
        return decay **(int(epoch / every))
    
    return lr_schedule_at_epoch


# STATISTICAL ANALYSIS

def get_best_parameter(
    all_evaluations, dropout_rates, lr_rates_nn, decay_rates):
    """
    TODO
    """
    Collect=np.zeros((len(dropout_rates), len(lr_rates_nn), len(decay_rates)))

    for drop_num in enumerate(dropout_rates):
        for lr_num in enumerate(lr_rates_nn):
            for decay_num in range((decay_rates)):
                Collect[drop_num,lr_num, decay_num], _, _, _=all_evaluations[(drop_num,lr_num,decay_num)] 

    drop_num, lr_num, decay_num=np.argmin(Collect)
    _, best_dropout, best_lr, best_decay=all_evaluations[(drop_num, lr_num, decay_num)] 

    return best_dropout, best_lr, best_decay


class Extract_v2():
    """
    In LSS_connector.py.
    """
    def __init__(self):
        self.normalization_constant=23.061
    
    def force_matrix_linalg(self, how_many):
        """
        Given the path of a dat file, returns a tensor with it's
        content.
        """
        force_m=[]
        for i in range(how_many):
            filename="force_" + str(i) + ".dat"
            dataframe=pd.read_csv(
                filepath_or_buffer=filename,
                engine='python', skiprows=9,
                names=['id','fx','fy', 'fz'], delimiter="\s+")
            dataframe.drop(['id'],axis=1,inplace=True)
            force_m.append(dataframe.to_numpy(dtype=np.float32))

        return tf.constant(np.stack(force_m, axis=0))

    def retrieve_ab_initio_matrix(self):
        """
        TODO
        """
        df=pd.read_csv(
            filepath_or_buffer="../ab_initio/force.dat",
            engine='python', skiprows=0,
            names=['fx','fy', 'fz'], delimiter="\s+")
        matrix_aux=df.to_numpy()
        matrix_aux=np.split(matrix_aux, 1000, axis=0)
        matrix_aux=np.stack(matrix_aux, axis=0)
        matrix_aux=tf.constant(matrix_aux, dtype=tf.float32, shape=matrix_aux.shape)
        self.ab_initio_matrix=self.normalization_constant * matrix_aux

    def at_configurations(self, configurations):
        """TODO"""
        
        return  tf.gather(self.ab_initio_matrix, indices=configurations)

    def split_ab_initio(self, test):
        """
        Create ab_initio matrix that will be used for tests
        """
        self.ab_initio_matrix_test=self.at_configurations(test)

    def print_ab_initio_chunk(self, configurations, in_parts=False):
        """
        TODO
        """
        chunk=self.at_configurations(configurations)
        logger.info("Printing Ab initio chunk")
        print_parameters(chunk, in_parts=in_parts)
        logger.info("Done")
