# -*- coding: utf-8 -*-
"""
Created on Thursday Jan 12 18:45:53 2023

@author: Olivier BEAUDE
"""
import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from dataclasses import dataclass
from typing import Tuple


# a very simple "dataclass", to be understood as an object, which will be used only to store data
# no class method, etc.
@dataclass
class SolverOutput:
    status: str
    objective: float


def set_tuple_of_ranges(tuple_of_ints: Tuple[int]) -> Tuple[range]:
    """
    Set a tuple of ranges based on a tuple of int
    """
    return tuple([range(elt) for elt in tuple_of_ints])


def create_pyo_concrete_model():
    """
    Create a Pyomo concrete model
    """
    return pyo.ConcreteModel()


def write_simple_lp_model() -> pyo.ConcreteModel:
    """
    Create a very simple lp model using Pyomo
    min 2x_1 + 3x_2 s.t. 3x_1 + 4x_2 >= 1 and x_1, x_2 >= 0
    :return: a Pyomo model
    """
    # taken from https://pyomo.readthedocs.io/en/stable/pyomo_overview/simple_examples.html
    # create a Pyomo concrete model (for LP)
    model = create_pyo_concrete_model()

    # add a variable x to this model, x = (x_1, x_2) here, and is a non-negative real
    model.x = pyo.Var([1, 2], domain=pyo.NonNegativeReals)

    # define objective function of this model obj = 2*x_1 + 3*x_2
    # N.B. default optimization "sense" is to minimize;
    # if you want to maximize add sense=pyo.maximize after "expr" definition
    model.obj = pyo.Objective(expr=2 * model.x[1] + 3 * model.x[2])

    # add a constraint 3 * x_1 + 4 * x_2 >= 1
    # N.B. you can name the constraint as you prefer, explicitly is better (related to its practical meaning)!
    model.unique_constraint = pyo.Constraint(expr=3 * model.x[1] + 4 * model.x[2] >= 1)

    return model


def write_parametrized_lp_model(var_dimensions: Tuple[int], constraint_coeffs: np.array, sum_lower_bound: float,
                                obj_coeffs: np.array, var_upper_bound: float = None) -> pyo.ConcreteModel:
    """
    Write a Pyomo model with parametrized aspects, typically the indices of the variable

    :param var_dimensions: dimensions of the optimization variable, e.g. (J, T)
    :param constraint_coeffs: coefficients to be used in the linear constraint
    :param sum_lower_bound: lower bound associated to linear constraint
    :param obj_coeffs: coefficients to be used for the linear objective function
    :param var_upper_bound: upper_bound for the variable
    :return: returns the created Pyomo model
    """
    # create a Pyomo concrete model (for LP)
    model = create_pyo_concrete_model()

    # add a variable x to this model, BASED on a PARAMETRIZED SET OF INDICES
    tuple_of_ranges = set_tuple_of_ranges(tuple_of_ints=var_dimensions)
    # and with an upper bound (the same for all elements of x), if provided
    if var_upper_bound is not None:
        model.x = pyo.Var(*tuple_of_ranges, domain=pyo.NonNegativeReals, bounds=(0, var_upper_bound))
    else:
        model.x = pyo.Var(*tuple_of_ranges, domain=pyo.NonNegativeReals)

    # add a constraint, based on a summation of all x components
    if not constraint_coeffs.shape == var_dimensions:
        print(f"Matrix of coefficients of the sum-constraint must be of the same dimensions as the variable, "
              f"i.e. {var_dimensions} -> this constraint can not be applied")
    else:
        model.unique_constraint = pyo.Constraint(expr=pyo.summation(constraint_coeffs, model.x) >= sum_lower_bound)

    # and an objective, also dependent on a matrix of coefficients
    if not obj_coeffs.shape == var_dimensions:
        print(f"Matrix of coefficients of the (linear) obj. function must be of the same dimensions as the variable, "
              f"i.e. {var_dimensions} -> this obj. can not be applied")
    else:
        model.obj = pyo.Objective(expr=pyo.summation(obj_coeffs, model.x))

    return model


def solve_pyomo_model(model: pyo.ConcreteModel, output_dir: str, solver_name: str = None, exe_path: str = None,
                      write_lp_file: bool = True, lp_file: str = "simple_optim_pb") -> SolverOutput:
    """
    Solve a Pyomo optimization model
    :param model: considered Pyomo model
    :param output_dir: directory in which potential outputs must be saved (e.g., .lp file)
    :param solver_name: name of the solver to be used
    :param exe_path: path to the executable of this solver
    :param write_lp_file: write .lp file?
    :param lp_file: name of the .lp file to be written
    :return: a SolverOutput object
    """

    if solver_name is not None and exe_path is not None:
        opt = pyo.SolverFactory(solver_name, executable=exe_path)
    else:
        print(f"A solver name, and executable path must be provided to solve a Pyomo model")
        return None

    # write problem in a .lp file (standard format to write optimization pbs, that can be
    # then "sent" to any standard solver)
    if write_lp_file:
        model.write(os.path.join(output_dir, f"{lp_file}.lp"), io_options={"symbolic_solver_labels": True})

    results = opt.solve(model)

    return SolverOutput(status=results.solver.status, objective=pyo.value(model.obj))


def get_optimal_variable_value(model: pyo.ConcreteModel) -> np.array:
    """
    Get the value of the variable (here unique) at optimum.
    N.B. if multiple variables; the different values will be obtained by adapting the following code with
    the different names of the variables
    :return: the optimal decisions in a numpy array (of the same size as the optimization variable)
    """

    var_opt_values = {}

    # loop over all optimization variables in the model
    for v in model.component_objects(pyo.Var, active=True):
        var_opt_values[v.name] = {}
        # then over the index of this variable
        for index in v:
            var_opt_values[v.name][index] = pyo.value(v[index])

    return var_opt_values


def solve_and_get_opt_values(model: pyo.ConcreteModel, output_dir: str, solver_name: str = None,
                             solver_exe_path: str = None,
                             lp_file: str = "simple_optim_pb") -> (SolverOutput, dict):
    """
    Solve a Pyomo model and get optimal values
    :param model: considered Pyomo model
    :param output_dir: directory in which potential outputs must be saved (e.g., .lp file)
    :param solver_name: name of the solver to be used
    :param solver_exe_path: path to the executable of this solver
    :param lp_file: name of the .lp file to be written
    """

    # then solve it
    solver_output = solve_pyomo_model(model=model, solver_name=solver_name, exe_path=solver_exe_path,
                                      output_dir=output_dir, lp_file=lp_file)
    # and get the optimal solution
    var_opt_values = get_optimal_variable_value(model=model)
    # print the status and value of the optimization pb solved
    print(f"Optimization status is {solver_output.status} with \n- opt. decision: {var_opt_values} "
          f"and \n- value={solver_output.objective:.2f}")

    return solver_output, var_opt_values


def var_opt_values_to_csv(var_opt_values: dict, output_dir: str, csv_file: str):
    """
    Save variables optimal decisions into a .csv file
    N.B. this function assumes that all variables share the same indices; otherwise it has to be slightly modified
    """

    dict_opt_values = {}
    # get first variable name from dict of optimal values
    first_var_name = list(var_opt_values.keys())[0]

    # index column taken from first dictionary, assuming that other dict. share the same list
    dict_opt_values["index"] = list(var_opt_values[first_var_name].keys())

    # then loop over the different variables to add an elt in the created dictionary
    for elt_var in var_opt_values:
        dict_opt_values[elt_var] = list(var_opt_values[elt_var].values())

    # convert dictionary to pandas dataframe
    df_opt_values = pd.DataFrame(dict_opt_values)

    # set "index" as first column in the written file
    ordered_cols = ["index"]
    other_cols = list(set(dict_opt_values) - {"index"})
    ordered_cols.extend(other_cols)

    # and save it as a .csv file
    df_opt_values.to_csv(os.path.join(output_dir, f"{csv_file}.csv"), sep=";", decimal=".",
                         columns=ordered_cols, index=False)


if __name__ == "__main__":
    current_dir = os.getcwd()

    # solver parameters
    solver_name = "glpk"  # name of the solver to be used
    # path to the solver executable -> I recommend using Glpk, that can be downloaded
    # here https://www.gnu.org/software/glpk/ (in this case path of glpsol must be given);
    # CBC could be used instead -> see
    # https://pyomo.readthedocs.io/en/stable/solving_pyomo_models.html#supported-solvers
    # adapt the following path after having installed glpk
    solver_exe_path = "C://Users//B57876//winglpk-4.65//glpk-4.65//w32//glpsol"
    # you have to manually create an "output_pyomo" folder in current directory
    # for this script to be directly functional
    output_dir = os.path.join(current_dir, "output_pyomo")

    # Ex1: write a first LP model with Pyomo
    simple_pyo_model = write_simple_lp_model()

    # then solve it, and get optimal decision values (and obj.)
    solver_output, var_opt_values = \
        solve_and_get_opt_values(model=simple_pyo_model, output_dir=output_dir, solver_name=solver_name,
                                 solver_exe_path=solver_exe_path)
    # and save optimal decisions to a .csv file
    var_opt_values_to_csv(var_opt_values=var_opt_values, output_dir=output_dir, csv_file="simple_optim_pb_sol")

    # Ex2: now a parametrized model -> 1D (e.g., temporal indices)
    var_dimensions = (10,)
    constraint_coeffs = np.arange(var_dimensions[0])
    # obj. function coefficients are the one of the constraint coeffs... but in reverse order!
    obj_coeffs = np.fliplr(constraint_coeffs.reshape(1, len(constraint_coeffs))).flatten()
    parametrized_pyo_model = \
        write_parametrized_lp_model(var_dimensions=var_dimensions, constraint_coeffs=constraint_coeffs,
                                    sum_lower_bound=10, obj_coeffs=obj_coeffs, var_upper_bound=5)
    solver_output, var_opt_values = \
        solve_and_get_opt_values(model=parametrized_pyo_model, output_dir=output_dir, solver_name=solver_name,
                                 solver_exe_path=solver_exe_path, lp_file="parametrized_optim_pb")
    # and save optimal decisions to a .csv file
    var_opt_values_to_csv(var_opt_values=var_opt_values, output_dir=output_dir, csv_file="parametrized_optim_pb_sol")

    # Ex2bis: now a parametrized model -> 2D (e.g., EV and temporal indices)
    var_dimensions = (3, 10)
    constraint_coeffs = np.concatenate((np.arange(var_dimensions[1]).reshape(1, var_dimensions[1]),
                                        0.5 * np.arange(var_dimensions[1]).reshape(1, var_dimensions[1]),
                                        2 * np.arange(var_dimensions[1]).reshape(1, var_dimensions[1])
                                        ))
    # obj. function coefficients are the one of the constraint coeffs... but in reverse order!
    obj_coeffs = np.random.rand(*constraint_coeffs.shape)
    two_dim_parametrized_pyo_model = \
        write_parametrized_lp_model(var_dimensions=var_dimensions, constraint_coeffs=constraint_coeffs,
                                    sum_lower_bound=10, obj_coeffs=obj_coeffs, var_upper_bound=5)
    solver_output, var_opt_values = \
        solve_and_get_opt_values(model=two_dim_parametrized_pyo_model, output_dir=output_dir, solver_name=solver_name,
                                 solver_exe_path=solver_exe_path, lp_file="two_dim_parametrized_optim_pb")
    # and save optimal decisions to a .csv file
    var_opt_values_to_csv(var_opt_values=var_opt_values, output_dir=output_dir,
                          csv_file="two_dim_parametrized_optim_pb_sol")

