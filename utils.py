from jax import numpy as jnp

def human_readable_dict(dictionary):
    """
    Transform all jax arrays of one element into scalars.
    """
    meta_keys = dictionary.keys()
    for meta_key in meta_keys:
        if isinstance(dictionary[meta_key], jnp.ndarray):
            dictionary[meta_key] = dictionary[meta_key].item()
        elif isinstance(dictionary[meta_key], dict):
            for key in dictionary[meta_key].keys():
                if isinstance(dictionary[meta_key][key], jnp.ndarray):
                    dictionary[meta_key][key] = dictionary[meta_key][key].item()
