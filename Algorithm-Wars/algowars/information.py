# -*- coding: utf-8 -*-
import numpy as np


def entropy(X):
    '''
    H(X)
    
    Args:
        X:    1-D `np.ndarray`.

    Returns:
        `float`
    '''
    unique, count = np.unique(
        X, 
        return_counts=True, 
        axis=0
    )
    p = count / X.shape[0]
    H = np.sum((-1)*p*np.log2(p))
    return H


def joint_entropy(X, Y):
    '''
    H(Y;X)

    Args:
        X:    1-D `np.ndarray`.
        Y:    1-D `np.ndarray`.

    Returns:
        `float`
    '''
    return entropy(np.c_[X, Y])


def conditional_entropy(X, Y):
    """
    H(Y|X) = H(Y;X) - H(X)

    Args:
        X:    1-D `np.ndarray`.
        Y:    1-D `np.ndarray`.

    Returns:
        `float`
    """
    return joint_entropy(Y, X) - entropy(X)


def thermodynamic_depth(S):
    '''
    References:
        - Bennett, C. (1988). "Logical Depth and Physical Complexity", In Rolf Herken, The universal Turning Machine: A Half-Century Survey, Oxford University Press, 1988, pp.227-257.
        - Lloyd, S., & Pagels, H. (1988). Complexity as thermodynamic depth. Annals of physics, 188(1), pp186-213.
    '''
    if S.shape[0] % 2 != 0:
        S = S[1:]

    if S.shape[0] <= 2:
        return 0.0

    S_h = S[:S.shape[0]//2]
    S_0 = S[S.shape[0]//2:]

    D = conditional_entropy(X=S_0, Y=S_h)
    return D


def thermodynamic_dive(S):
    '''
    References:
        - Bennett, C. (1988). "Logical Depth and Physical Complexity", In Rolf Herken, The universal Turning Machine: A Half-Century Survey, Oxford University Press, 1988, pp.227-257.
        - Crutchfield, J. P., & Shalizi, C. R. (1999). Thermodynamic depth of causal states: Objective complexity via minimal representations. Physical review E, 59(1), 275.
        - Lloyd, S., & Pagels, H. (1988). Complexity as thermodynamic depth. Annals of physics, 188(1), pp186-213.
    '''
    S_before = S[:S.shape[0]//2]
    S_after = S[S.shape[0]//2:]
    D_before = thermodynamic_depth(S_before)
    D_after = thermodynamic_depth(S_after)
    return D_after - D_before
