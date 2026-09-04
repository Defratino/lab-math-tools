import numpy as np
from lab_math_tools.derivatives import av_gradient_saap, derivative_saap

def propagate_uncertainty_saap(f, x: float | np.ndarray, dx: float | np.ndarray, method: str = "statistical", h: float = 1e-5) -> float:
    """
    Calculates the propagated uncertainty of a function using numerical 
    derivative approximations (SAAP).

    Supports both scalar (R -> R) and vector (R^n -> R) mappings.

    This function automatically computes the partial derivatives (sensitivities)
    of the function with respect to each input variable, then scales them by 
    their respective uncertainties to find the total combined error.

    Parameters:
    * s_f (callable): The scalar-valued objective function.
    * av_x (np.ndarray): The nominal values of the independent variables.
    * av_dx (np.ndarray): The absolute uncertainties (errors) associated with each variable in av_x.
    * method (str): The combination rule for the errors.
        - "statistical": (Default) Assumes errors are independent and random, combining them via Root-Sum-Square (RSS).
        - "absolute": Assumes a worst-case scenario where all errors stack in the same direction.
    * h (float): The step size for the numerical derivative approximation.

    Returns:
    * float: The total propagated uncertainty (\Delta f).

    Mathematical Formulation:
    Statistical (Root-Sum-Square):
    Absolute (Worst-Case == Sum-Absolute):
    """

    # 1. Handle pure scalar mapping (R -> R)
    if np.isscalar(x) and np.isscalar(dx):
        sensitivity = derivative_saap(f, x, h)
        return float(np.abs(sensitivity * dx))
        
    # 2. Handle vector mapping (R^n -> R)
    assert len(x) == len(dx), "Value and uncertainty vectors must have the same dimension."
    
    av_sensitivities = av_gradient_saap(f, x, h)
    av_terms = av_sensitivities * dx
    
    if method == "statistical":
        return float(np.sqrt(np.sum(av_terms**2)))
    elif method == "absolute":
        return float(np.sum(np.abs(av_terms)))
    else:
        raise ValueError("Method must be 'statistical' or 'absolute'.")