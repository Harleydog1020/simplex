"""
This is my LP Simplex Solver.  It uses matrix algebra
to solve simple max and min problems. ***
"""

import pandas as pd
import numpy as np
from pandas import DataFrame


class SimplexData:
    def __init__(self, input_df):
        self.var_labels = np.array(list(input_df.columns))
        self.con_ar = np.array(input_df.iloc[0:, :1])
        temp1a = np.array(input_df.iloc[1:, 1:])
        temp1b = np.array(input_df.iloc[0:1, 1:])
        temp1 = np.concatenate((temp1a, temp1b), axis=0)

        self.p_type = [[x.upper() for x in y] for y in np.array(input_df.iloc[0:1, 0:1])]
        if self.p_type == [['MIN']]:
            temp2 = temp1.T
        else:
            temp2 = temp1

        self.n_rows, self.n_cols = temp2.shape
        self.rhs_ar = np.array(temp2[0:, self.n_cols - 1:])

        self.obj_ar = np.array(temp2[self.n_rows-1:self.n_rows, 0:self.n_cols - 1]) * - 1
        a1 = np.identity(self.n_rows - 1)
        a3 =[[0 for element in range(self.n_rows - 1)]]
        a2 = np.concatenate((a1, a3), axis=0)
        self.eqn_ar = np.concatenate((np.array(temp2[0:self.n_rows-1, 0:self.n_cols-1]),
                                               np.array(self.obj_ar)),
                                     axis=0)
        self.table = np.concatenate((self.eqn_ar, a2), axis=1)


def read_problem(in_file: str):
    input_df: DataFrame = pd.read_csv(in_file, header=0).fillna(0)
    problem_ar = SimplexData(input_df)

    return problem_ar


def solver(solve_ar: SimplexData):
    new_equations = solve_ar.table
    new_rhs = solve_ar.rhs_ar
    n_rows = solve_ar.n_rows
    num_neg = np.count_nonzero(new_equations[n_rows - 1:n_rows, 0:] < 0)
    i_stop = 0
    while num_neg > 0 and i_stop < 3:
        i_stop = i_stop + 1
        # Find most negative column in obj function = pivot_column
        obj_min = np.min(new_equations[n_rows - 1:n_rows, 0:])

        found = False
        i_col = 0
        while not found:
            if new_equations[n_rows - 1:n_rows, i_col: i_col + 1] == obj_min:
                found = True
            else:
                i_col = i_col + 1
        # Find row in pivot_column where rhs / row_value is min of operation = pivot_row
        pvt_row = np.NAN
        i_min = np.NAN
        for i_row in range(0, n_rows - 1):

            if np.isnan(i_min) or new_rhs[i_row:i_row + 1] / new_equations[i_row:i_row + 1, i_col: i_col + 1] < i_min:
                i_min = new_rhs[i_row:i_row + 1] / new_equations[i_row:i_row + 1, i_col: i_col + 1]
                pvt_row = i_row

        a2 = np.identity(n_rows)

        # column = pivot_row: 1/row_value for pivot_row
        # column <> pivot_row: 1
        new_col = -1 * new_equations[0:, i_col: i_col + 1] / new_equations[pvt_row:pvt_row + 1, i_col: i_col + 1]
        new_col[pvt_row:pvt_row + 1] = 1 / new_equations[pvt_row:pvt_row + 1, i_col: i_col + 1]

        # column = pivot_row: 1
        # column <> pivot_row: (- pivot_column) value for that row
        a2[0:, pvt_row:pvt_row + 1] = new_col
        new_equations = np.dot(a2, new_equations)
        new_rhs = np.dot(a2, new_rhs)
        num_neg = np.count_nonzero(new_equations[n_rows - 1:n_rows, 0:] < 0)

    if solve_ar.p_type == [['MIN']]:
        temp1 = np.concatenate((new_equations, new_rhs), axis=1)
        temp2 = temp1.T
        new_equations = temp2[0:, 0:solve_ar.n_rows-1]
        new_rhs = temp2[0:, solve_ar.n_rows-1:solve_ar.n_rows]
    return new_equations, new_rhs


# #########################  MAIN  #########################################
if __name__ == '__main__':
    problem_data = read_problem('~/simplex/data/sample5.csv')
    solution_equations, solution_rhs = solver(problem_data)

    print("----------------- Problem ----------------------")
    print(problem_data.var_labels)
    print(problem_data.table)
    print(problem_data.rhs_ar)

    print("----------------- Solution ---------------------")
    print(np.round(solution_equations,2))
    print(np.round(solution_rhs,2))
