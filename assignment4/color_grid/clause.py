"""
Class used to represent a clause in CNF for the color grid problem.
Variable X_i_j_k is true iff cell at position (i,j) has color k.

For example, to create a clause:

X_0_1_1 or ~X_1_2 or X_3_3_2

you can do:

clause = Clause(4)
clause.add_positive(0, 1, 1)
clause.add_negative(1, 2, 2)
clause.add_positive(3, 3, 2)

"""


class Clause:

    def __init__(self, size, varname='C'):
        self.varname = varname
        self.n_rows = self.n_columns = self.n_colors = size
        self.value = []

    def index(self, row_ind, column_ind, color_ind):
        if row_ind >= 0 and row_ind < self.n_rows and column_ind >= 0 and column_ind < self.n_columns and color_ind >= 0 and color_ind < self.n_colors:
            return (row_ind * self.n_columns + column_ind) * self.n_colors + color_ind
        else:
            raise ValueError("Indices : row_ind =", row_ind, "column_ind =", column_ind, "color_ind =", color_ind, "is incorrect")

    def str_from_index(self, index):
        if index >= 0:
            index -= 1
        else:
            index += 1
        color_ind = index % self.n_colors
        # if color_ind == 0:
        #     color_ind = self.n_colors - 1
        # else:
        #     color_ind -= 1
        row_ind = index // self.n_colors // self.n_columns
        column_ind = index // self.n_colors % self.n_columns
        if index < 0:
            return '~{0}_{1}_{2}_{3}'.format(self.varname, row_ind, column_ind, color_ind)
        return '{0}_{1}_{2}_{3}'.format(self.varname, row_ind, column_ind, color_ind)

    def add_positive(self, row_ind, column_ind, color_ind):
        self.value.append(self.index(row_ind, column_ind, color_ind)+1)

    def add_negative(self, row_ind, column_ind, color_ind):
        self.value.append(-self.index(row_ind, column_ind, color_ind)-1)

    def minisat_str(self):
        return ' '.join([str(x) for x in self.value])

    def __str__(self):
        return ' or '.join([self.str_from_index(x) for x in self.value])


if __name__ == '__main__':
    clause = Clause(3)
    clause.add_positive(1, 1, 1)
    clause.add_negative(1, 2, 2)
    clause.add_positive(2, 2, 0)
    print(clause)
    print(clause.minisat_str())
