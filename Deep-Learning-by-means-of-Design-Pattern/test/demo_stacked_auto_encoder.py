# -*- coding: utf-8 -*-
import numpy as np

# `StackedAutoEncoder` is-a `DeepBoltzmannMachine`.
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
# The `Concrete Builder` in Builder Pattern.
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
# Contrastive Divergence for function approximation.
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction

# Observed data points.
observed_arr = np.random.normal(loc=0.5, scale=0.2, size=(10000, 10000))
print("Observed data points:")
print(observed_arr)

# Setting objects for activation function.
activation_list = [
    LogisticFunction(binary_flag=False, normalize_flag=False), 
    LogisticFunction(binary_flag=False, normalize_flag=False), 
    LogisticFunction(binary_flag=False, normalize_flag=False)
]

# Setting the object for function approximation.
approximaion_list = [ContrastiveDivergence(), ContrastiveDivergence()]

dbm = StackedAutoEncoder(
    DBMMultiLayerBuilder(),
    [observed_arr.shape[1], 10, observed_arr.shape[1]],
    activation_list,
    approximaion_list,
    1e-05, # Setting learning rate.
    0.5   # Setting dropout rate.
)

# Execute learning.
dbm.learn(
    observed_arr,
    1, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=100,  # Batch size in mini-batch training.
    r_batch_size=-1,  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
    sgd_flag=True
)

# Extract reconstruction error.
reconstruct_error_arr = dbm.get_reconstruct_error_arr(layer_number=0)
print("Reconstruction error.")
print(reconstruct_error_arr)

print("Feature points:")
print(dbm.feature_points_arr)
