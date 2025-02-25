import jax
import jax.numpy as jnp

def squash(s, axis=-1, epsilon=1e-7):
    """
    Squash activation function used in Capsule Networks.
    
    This function squashes an input vector such that its length falls between 0 and 1
    while preserving its direction.
    
    Args:
        s: Input tensor, represents the vector(s) to be squashed
        axis: The axis along which to compute the norm (default: -1)
        epsilon: Small constant to avoid division by zero (default: 1e-7)
        
    Returns:
        The squashed vectors with the same shape as the input
    """
    # Compute squared norm of s along the specified axis
    squared_norm = jnp.sum(jnp.square(s), axis=axis, keepdims=True)
    
    # Compute the norm (length) of the vector
    norm = jnp.sqrt(squared_norm + epsilon)
    
    # Apply the squashing formula:
    # v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
    scale = squared_norm / (1.0 + squared_norm)
    unit_vector = s / norm
    
    # Return the squashed vector
    return scale * unit_vector


# Example usage
def test_squash():
    # Create a sample input tensor
    x = jnp.array([
        [1.0, 0.0, 0.0],  # Unit vector along x-axis
        [0.0, 2.0, 0.0],  # Vector with length 2 along y-axis
        [0.0, 0.0, 10.0], # Vector with length 10 along z-axis
        [1.0, 1.0, 1.0]   # Vector along diagonal
    ])
    
    # Apply squash function
    squashed = squash(x)
    
    # Print results
    print("Original vectors:")
    print(x)
    print("\nOriginal vector norms:")
    print(jnp.sqrt(jnp.sum(jnp.square(x), axis=-1)))
    print("\nSquashed vectors:")
    print(squashed)
    print("\nSquashed vector norms:")
    print(jnp.sqrt(jnp.sum(jnp.square(squashed), axis=-1)))
    
    return squashed
