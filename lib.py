import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import copy
from scipy.optimize import minimize
from math import exp, log, fabs
from math import pi, cos, sin
from scipy.stats import beta

def Revenue(F, A, v, sign = 1.0):
    return sign * (np.dot(np.array([[1 - F(element) for element in v]]), A).dot(v.reshape((v.shape[0], 1)))[0, 0])

def Gradient(F, F_derivative, A, v, sign = 1.0):
    B = A * np.array([[-F_derivative(element)] for element in v.reshape((v.shape[0], ))])
    left = np.dot(B, v.reshape((v.shape[0], 1)))
    right = np.dot(A.transpose(), np.array([[1 - F(element)] for element in v.reshape((v.shape[0], ))]))
    return (sign * (left + right)).flatten()


def lambda_condition(i):
    return (lambda x: np.array(x[i] - x[i - 1]))


def jac_condition(i, dim):
    return (lambda x: np.eye(dim)[i, :] - np.eye(dim)[i - 1, :])


def tetrahedron_conditions(dim):
    conditions_list = [{'type': 'ineq', 'fun': lambda x: x[0],
                        'jac': lambda x, dim=dim: np.eye(dim)[0, :]}]

    conditions_list.extend([{'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[i - 1],
                             'jac': lambda x, i=i, dim=dim: np.eye(dim)[i, :] - np.eye(dim)[i - 1, :]} for i in
                            range(1, dim)])

    conditions_list.append({'type': 'ineq', 'fun': lambda x, dim=dim: 1.0 - x[dim - 1],
                            'jac': lambda x, dim=dim: - np.eye(dim)[dim - 1, :]})

    return tuple(conditions_list)


def TransitionMatrix(T, gamma, infinite):
    if T == 1:
        return np.array([[1.0 / (1.0 - gamma)]]) if infinite else np.array([[1.0]])
    left_up = gamma * TransitionMatrix(T - 1, gamma, infinite)
    right_up = np.zeros((2 ** (T - 1) - 1, 2 ** (T - 1)))
    left_down = np.zeros((2 ** (T - 1), 2 ** (T - 1) - 1))
    right_down = np.bmat([[np.array([[1.0]]), np.zeros((1, 2 ** (T - 1) - 1))],
                          [1 + np.zeros((2 ** (T - 1) - 1, 1)), gamma * TransitionMatrix(T - 1, gamma, infinite)]])
    return np.array(np.bmat([[left_up, right_up], [left_down, right_down]]))


def MatrixStringsWeightsOrder(matrix):
    return np.argsort((np.sum(matrix, axis=1)).reshape((matrix.shape[0],)))


def RearrangeRows(matrix, order):
    return matrix[order, :]


def TwoDiagonalMatrix(T):
    k = 2 ** T - 1
    matrix = np.eye(k)
    for i in np.arange(k - 1, 0, -1):
        matrix[i] = matrix[i] - matrix[i - 1]
    return matrix


def TwoDiagonalMatrixInverse(T):
    k = 2 ** T - 1
    matrix = np.eye(k)
    for i in np.arange(1, k, 1):
        matrix[i] = matrix[i] + matrix[i - 1]
    return matrix


def uncompress(v, zero_indices):
    uncompressed = np.zeros((len(zero_indices),))
    j = 0
    for i in range(0, len(zero_indices)):
        if zero_indices[i]:
            uncompressed[i] = uncompressed[i - 1]
        else:
            uncompressed[i] = v[j]
            j = j + 1
    return uncompressed


def p_dictionary(prefix, p):
    size = p.shape[0]
    half_size = size >> 1
    result = dict()
    result['p_' + prefix] = p[half_size]
    if size > 1:
        left_tree_result = p_dictionary(prefix + '0', p[0:half_size])
        right_tree_result = p_dictionary(prefix + '1', p[half_size + 1: size])
        result.update(left_tree_result)
        result.update(right_tree_result)
    return result


def DictionaryOutput(T, v, p, max_revenue):
    result = dict()
    result['revenue'] = max_revenue
    for i in range(0, v.shape[0]):
        result['v_' + str(i + 1)] = v[i]
    result.update(p_dictionary('', p))
    return result


def MatrixPairGenerator(T, gamma_s, gamma_b, MatrixGenerator):
    eps = 0.000001
    K_b = MatrixGenerator(T, gamma_b)
    K_supp = MatrixGenerator(T, gamma_b + eps)
    first_coord = np.sum(K_b, axis=1)
    second_coord = np.sum(K_supp, axis=1)
    weights_order = np.lexsort((first_coord, second_coord))
    return (RearrangeRows(K_b, weights_order), RearrangeRows(MatrixGenerator(T, gamma_s), weights_order))


def compress_matrix(A, zero_indices):
    B = np.zeros(A.shape)
    for i in range(1, len(zero_indices)):
        B[i - 1] = A[i - 1] + A[i] if zero_indices[i] else A[i - 1]
    B[len(zero_indices) - 1] = A[len(zero_indices) - 1]
    indices_array = np.array([not element for element in zero_indices])
    B = B[indices_array, :]
    return B[:, indices_array]


def GlobalSolver(T, cdf, pdf, gamma_s, gamma_b, infinite):
    transition_matrix = lambda T, gamma, infinite=infinite: TransitionMatrix(T, gamma, infinite)
    (K_b, K_s) = MatrixPairGenerator(T, gamma_s, gamma_b, transition_matrix)

    E = TwoDiagonalMatrix(T)
    E_inverse = TwoDiagonalMatrixInverse(T)

    weights = np.sum(K_b, axis=1)

    diagonal_list = [weights[0]]
    diagonal_list.extend([weights[i] - weights[i - 1] for i in range(1, weights.shape[0])])
    D = np.diag(diagonal_list)

    A = np.linalg.multi_dot([E, K_s, np.linalg.inv(K_b), E_inverse, D])
    zero_indices = [element < 1.0e-15 for element in diagonal_list]
    A = compress_matrix(A, zero_indices)

    revenue = lambda v, A=A, cdf=cdf: Revenue(cdf, A, v, -1.0)
    gradient = lambda v, A=A, cdf=cdf, pdf=pdf: Gradient(cdf, pdf, A, v, -1.0)

    dimensionality = 2 ** T - 1 - np.sum(zero_indices)
    conditions = tetrahedron_conditions(dimensionality)
    v_start = np.random.rand(dimensionality) / dimensionality
    for i in range(1, dimensionality):
        v_start[i] = v_start[i] + v_start[i - 1]
    # v_start = np.array([1.0 * i / (dimensionality + 1) for i in range(1, dimensionality + 1)])

    optimization_result = minimize(revenue, v_start, jac=gradient, constraints=conditions,
                                   method='SLSQP', options={'disp': False})

    v = uncompress(optimization_result['x'], zero_indices)
    p = np.linalg.multi_dot([np.linalg.inv(K_b), E_inverse, D, v.reshape((v.shape[0], 1))]).reshape((2 ** T - 1,))

    return DictionaryOutput(T, v, p, -optimization_result['fun'])


def GlobalSolverMaxN(T, cdf, pdf, gamma_s, gamma_b, infinite, attempts):
    results = [GlobalSolver(T, cdf, pdf, gamma_s, gamma_b, infinite) for i in range(0, attempts)]
    max_revenue = 0.0
    best_result = results[0]
    for result in results:
        if result['revenue'] > max_revenue:
            best_result = result
            max_revenue = result['revenue']
    return best_result


def CountBias(gamma, str_encoding, algorithm):
    discount = 1.0
    bias = 0.0
    prefix, suffix = 'p_', ''
    T = str_encoding.shape[0]
    for index in range(0, T):
        #        coeff = 1.0 if index < T - 1 else 1.0 / (1.0 - gamma)
        bias += discount * algorithm[prefix + suffix] * str_encoding[index]
        suffix += str(str_encoding[index])
        discount *= gamma
    return bias


def CountSlope(gamma, str_encoding):
    discount = 1.0
    slope = 0.0
    T = str_encoding.shape[0]
    for index in range(0, T):
        #        coeff = 1.0 if index < T - 1 else 1.0 / (1.0 - gamma)
        slope += discount * str_encoding[index]
        discount = discount * gamma
    return slope


def NextStrategy(algorithm, gamma, strategy):
    str_encoding = copy.copy(strategy[2])
    first_zero = -1
    T = str_encoding.shape[0]
    while first_zero + 1 < T and str_encoding[first_zero + 1] == 1:
        first_zero += 1
    if first_zero == T - 1:
        return (0.0, 0.0, np.zeros((T,), dtype='int'))
    else:
        first_zero += 1
    for index in range(0, first_zero):
        str_encoding[index] = 0
    str_encoding[first_zero] = 1

    slope = CountSlope(gamma, str_encoding)
    bias = CountBias(gamma, str_encoding, algorithm)

    return (slope, bias, str_encoding)


def AppendLine(surplus_curve, line):
    """ Appends a line with a slope (line[0]) greater than all previous slopes to the surplus curve. Surplus curve is guaranteed to contatin point (0.0, 0.0). Bias of a new line is guaranteed to be non-negative.

    """

    pop_last = True
    k_new, b_new = line[0], line[1]
    last_intersection = 0.0
    while pop_last:
        size = len(surplus_curve)
        k_last, b_last, v_last = surplus_curve[size - 1][1], surplus_curve[size - 1][2], surplus_curve[size - 1][0]
        last_intersection = (b_new - b_last) / (k_new - k_last)
        pop_last = (last_intersection < v_last)
        if (pop_last):
            del surplus_curve[-1]
    surplus_curve.append((last_intersection, k_new, b_new, line[2]))


def SurplusCurve(gamma, algorithm):
    """ This function takes a buyer input and a seller algorithm as an input and outputs the surplus curve.

    INPUT:
    gamma - real number
    algorithm - dictionary, where each node is encoded by a binary sequence, e.g. p_001

    OUTPUT:

    surplus_curve = [(v_0, strategy_0), (v_1, strategy_1), ...], where v_i is a real number, v_i < v_{i + 1}, and
    strategy_i is a slope, a bias and a binary string. slopes are increasing. Surplus curve is
    an envelope of all lines that stands for one strategy surplus.
    """

    T = int(log(len(algorithm) + 1, 2))  # maybe debug
    zero_strategy = (0.0, 0.0, np.zeros((T,), dtype='int'))  # slope, bias, strategy encoding
    strategies = [zero_strategy]
    current_strategy = NextStrategy(algorithm, gamma, zero_strategy)
    while (current_strategy[2] != zero_strategy[2]).any():
        strategies.append(current_strategy)
        current_strategy = NextStrategy(algorithm, gamma, current_strategy)
    # comparison_operator = lambda x, y : x[0] < y[0] if (fabs(x[0] - y[0]) < 1.0e-10) else x[1] > y[1]
    strategies = sorted(strategies, key=lambda x: (x[0], -x[1]))

    surplus_curve = [(0.0, 0.0, 0.0, zero_strategy[2])]  # (v_start, slope, bias)
    for strategy in strategies[1:]:
        AppendLine(surplus_curve, strategy)
    return surplus_curve


def CountExpectedRevenue(gamma_S, gamma_B, algorithm, CDF):
    surplus_curve = SurplusCurve(gamma_B, algorithm)
    ESR = 0.0
    for index in range(len(surplus_curve) - 1):
        v_start = surplus_curve[index][0]
        v_end = surplus_curve[index + 1][0]
        probability = CDF(v_end) - CDF(v_start)
        str_code = surplus_curve[index][3]
        revenue = CountBias(gamma_S, surplus_curve[index][3], algorithm)
        ESR += revenue * probability
    last_index = len(surplus_curve) - 1
    probability = 1 - CDF(surplus_curve[last_index][0])
    revenue = CountBias(gamma_S, surplus_curve[last_index][3], algorithm)
    ESR += revenue * probability
    return ESR


def AnalyticalPlot(x_values, y_columns, func, x_label, title, legend_location, legend_ncol):
    plt.figure(figsize=(16, 9))
    y_results = [func(x_value) for x_value in x_values]
    y_values = np.array([[result[column] for column in y_columns] for result in y_results]).transpose()
    lines = []
    for y_row in y_values:
        line, = plt.plot(x_values, y_row)
        lines.append(line)
    plt.figlegend(lines, y_columns, loc= legend_location, ncol = legend_ncol)
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()


def AnalyticalComparisonPlot(x_values, y_columns, functions, x_label, title, legend_location, legend_ncol, labels):
    plt.figure(figsize=(16, 9))
    y_results = [[function(x_value) for x_value in x_values] for function in functions]

    lines, line_names = [], []
    for (result, label) in zip(y_results, labels):
        y_values = np.array([[dictionary[column] for column in y_columns] for dictionary in result]).transpose()
        for y_row in y_values:
            line, = plt.plot(x_values, y_row)
            lines.append(line)
        line_names.extend([y_column + label for y_column in y_columns])
    plt.figlegend(lines, line_names, loc=legend_location, ncol=legend_ncol)
    plt.title(title)
    plt.xlabel(x_label)
    plt.show()