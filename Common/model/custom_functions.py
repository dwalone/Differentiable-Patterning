import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from einops import rearrange,repeat
from jaxtyping import Array, Float, Int, Scalar
from itertools import combinations_with_replacement

"""
A collection of useful functions that are reused throughout PDE and NCA models
"""

@eqx.filter_jit
def construct_polynomials(X:Float[Array, "C"],max_power: Int[Scalar, ""])->Float[Array, "K"]:
    """
    Takes the vector X, and returns a longer vector including all polynomials of componentes of X, up to power
    """
    if max_power == 1:
        return X
    else:
        n = X.shape[0]
        terms = []

        for power in range(1, max_power + 1):
            for combo in combinations_with_replacement(range(n), power):
                indices = jnp.array(combo)
                term = jnp.prod(X[indices])
                terms.append(term)
        
        return jnp.array(terms)

def set_layer_weights(shape,key,INIT_TYPE,INIT_SCALE):
    if INIT_TYPE=="orthogonal":
        r = jr.orthogonal(key,n=max(shape[0],shape[1]))
        r = r[:shape[0],:shape[1]]
        r = repeat(r,"i j -> i j () ()")
        return INIT_SCALE*r
    if INIT_TYPE=="normal":
        return INIT_SCALE*jr.normal(key,shape)
    if INIT_TYPE=="diagonal":
        a = 0.9
        i = jnp.eye(shape[0],shape[1])
        i = repeat(i,"i j -> i j () ()")
        r = jr.normal(key,shape)
        return INIT_SCALE*(a*i+(1-a)*r)
    if INIT_TYPE=="permuted":
        a = 0.9
        i = jr.permutation(key,jnp.eye(shape[0],shape[1]),axis=1)
        i = repeat(i,"i j -> i j () ()")
        r = jr.normal(key,shape)
        return INIT_SCALE*(a*i+(1-a)*r)

def construct_polynomials_with_labels(X: jnp.ndarray, max_power: int, var_names=None, return_labels=True):
    """
    Returns all polynomial terms up to `max_power` over vector X,
    along with symbolic labels (if `return_labels=True`).

    If var_names is None, variables are named x0, x1, ...
    """
    n = X.shape[0]
    terms = []
    labels = []

    # Default variable names
    if var_names is None:
        var_names = [f"x{i}" for i in range(n)]

    for power in range(1, max_power + 1):
        for combo in combinations_with_replacement(range(n), power):
            indices = jnp.array(combo)
            term = jnp.prod(X[indices])
            terms.append(term)

            if return_labels:
                label_parts = [var_names[i] for i in combo]
                label = "".join(label_parts)
                labels.append(label)

    return (jnp.array(terms), labels) if return_labels else jnp.array(terms)
