"""
Graphing module

This module provides graphing tools for 2D data and 3D data. 

Conventions followed:
- `v_` : Abstract vector (numpy.ndarray with dynamic dimensions)
- `m_` : Abstract matrix (numpy.ndarray with dynamic dimensions)
- `s_f` : Scalar-valued function
- `v_f`: Vector-valued function
- `f`   : Generic function (can output scalar or vector)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Union

def plot_measurements(m_x: Union[np.ndarray, List[np.ndarray]], 
                      m_y: Union[np.ndarray, List[np.ndarray]], 
                      x_label: str = "X", 
                      y_label: str = "Y", 
                      title: str = "",
                      line_lables: List[str] = [],
                      grid: bool = True,
                      linestyles: Union[str, List[str]] = ['-'],
                      markers: Union[str, List[str]] = ['.'],
                      colors: List[str] = ['blue', 'green'],
                      color_method: str = 'gradient',
                      save_file_name: str = '',
                      ):
    """
    Plot multiple measurements on a 2D graph.
    
    Parameters:
    * m_x (np.ndarray): A matrix of x-values of the measurements. Each row is a different measurement.
        ** If m_x is of dimentions 1Xn (just a vector) all y-values will use the same v_x.
    * m_y (np.ndarray): A matrix of y-values of the measurements. Each row is a different measurement.
        ** if some x-values don't have y-values use np.nan.
    * x_label (str): The label for the x-axis.
    * y_label (str): The label for the y-axis.
    * title (str): The title of the graph.
    * linestyle (str): The style of the lines.
    * marker (str): The style of the markers.
    * colors (list[str]): The colors of the lines.
    * color_method (str): The method used to determine the colors of the lines.
        - "gradient": The colors of the lines will vary smoothly from one color to another.
        - "specific": The colors of the lines will be specific colors that are passed in by the user.
    """

    # Convert lists to matrix
    if isinstance(m_x, list):
        m_x = turn_list_of_vectors_to_matrix(m_x)
        print('Note: m_x was converted from a list of vectors to a matrix.')
    if isinstance(m_y, list):
        m_y = turn_list_of_vectors_to_matrix(m_y)
        print('Note: m_y was converted from a list of vectors to a matrix.')
    
    # Ensure m_x and m_y are numpy arrays
    m_x = np.array(m_x)
    m_y = np.array(m_y)

    # Correct dimensions if they are 1D vectors
    if m_x.ndim == 1:
        m_x = np.array([m_x])
    if m_y.ndim == 1:
        m_y = np.array([m_y])

    # duplicate X values if m_x has 1 row and m_y has more than 1 row
    if m_x.shape[0] == 1 and m_y.shape[0] > 1:
        m_x = np.tile(m_x, (m_y.shape[0], 1))

    # Ensure all y-values have x-values (more x-values than y-values is okay!)
    if m_y.shape[1] > m_x.shape[1]:
        raise ValueError("Not all y-values have x-values.")

    # Trim down any extra x values if the x-values are longer
    if m_x.shape[1] > m_y.shape[1]:
        m_x = m_x[:, :m_y.shape[1]]
        print('Warning: x-values were trimmed to match the y-values.')

    # Normalize linestyles to lists
    if not isinstance(linestyles, list):
        linestyles = [linestyles]
    if len(linestyles) == 1:
        linestyles = [linestyles[0]] * m_y.shape[0]
    
    # Normalize markers to lists
    if not isinstance(markers, list):
        markers = [markers]
    if len(markers) == 1:
        markers = [markers[0]] * m_y.shape[0]

    # Validate color method and bounds
    if color_method == 'gradient':
        if len(colors) != 2:
            raise ValueError("Colors must be a list of 2 colors when using the gradient method.")
    elif color_method == 'specific':
        if len(colors) != m_y.shape[0]:
            raise ValueError("Colors must be a list of colors for each measurement.")
    else:
        raise ValueError("Invalid color method, use 'gradient' or 'specific'.")
    
    if len(linestyles) != m_y.shape[0]:
        raise ValueError("linestyle must be a list of strings with the same length as the number of measurements.")
    
    if len(markers) != m_y.shape[0]:
        raise ValueError("marker must be a list of strings with the same length as the number of measurements.")

    if line_lables:
        if len(line_lables) != m_y.shape[0]:
            raise ValueError("line_lables must be a list of strings with the same length as the number of measurements.")

    if color_method == 'gradient':
        colors = _color_gradient(m_y.shape[0], colors[0], colors[1])
    
    # Plotting loop
    for i in range(m_y.shape[0]):
        if line_lables:
            plt.plot(m_x[i], m_y[i], linestyle=linestyles[i], marker=markers[i], color=colors[i], label=line_lables[i])
        else:
            plt.plot(m_x[i], m_y[i], linestyle=linestyles[i], marker=markers[i], color=colors[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if line_lables:
        plt.legend()
    if grid:
        plt.grid()
    if save_file_name:
        plt.savefig(save_file_name)

    plt.show()
    return

def turn_list_of_vectors_to_matrix(v_vectors : List[np.ndarray]) -> np.ndarray:
    """
    Turn an list of vectors into a matrix of measurements.
    
    Parameters:
    * v_vectors (List[np.ndarray]): A list of vectors (1D ndarrays).
    """
    if not v_vectors:
        return np.array([]).reshape(0, 0)

    # Find max length vector in the array
    max_len = max(len(vec) for vec in v_vectors)

    # Pad shorter vectors with np.nan (convert to float first to avoid type errors)
    lv_padded = []
    for vec in v_vectors:
        float_vec = vec.astype(float)
        padded_vec = np.pad(
            float_vec, 
            (0, max_len - len(float_vec)), 
            'constant', 
            constant_values=np.nan
        )
        lv_padded.append(padded_vec)

    return np.array(lv_padded)

def _color_gradient(n : int, color_1 : str, color_2 : str) -> list[str]:
    """
    Generate a list of colors that vary smoothly from one color to another.
    """
    rgb_1 = np.array(mcolors.to_rgb(color_1))
    rgb_2 = np.array(mcolors.to_rgb(color_2))
    return [mcolors.to_hex(rgb_1 + (rgb_2 - rgb_1) * i / (n-1)) for i in range(n)]
    
    