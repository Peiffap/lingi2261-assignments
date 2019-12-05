from clause import *

"""
For the color grid problem, the only code you have to do is in this file.

You should replace

# your code here

by a code generating a list of clauses modeling the grid color problem
for the input file.

You should build clauses using the Clause class defined in clause.py

Read the comment on top of clause.py to see how this works.
"""


def get_expression(size, points=None):
    expression = []
    for i in range(size):
        for j in range(size):
            # Color existence constraint.
            clause8 = Clause(size)
            clause9 = Clause(size)
            for k in range(size):
                clause8.add_positive(i, j, k)
                clause9.add_positive(j, i, k)
                expression.append(clause8)
                expression.append(clause9)
                for alpha in range(size):
                    if alpha != j:
                        # Row constraint.
                        clause1 = Clause(size)
                        clause1.add_negative(i, j, k)
                        clause1.add_negative(i, alpha, k)
                        expression.append(clause1)
                    if alpha != i:
                        # Column constraint.
                        clause2 = Clause(size)
                        clause2.add_negative(i, j, k)
                        clause2.add_negative(alpha, j, k)
                        expression.append(clause2)
                    if alpha != 0 and 0 <= i+alpha < size and 0 <= j+alpha < size:
                        # Constant-difference diagonal constraint 1.
                        clause3 = Clause(size)
                        clause3.add_negative(i, j, k)
                        clause3.add_negative(i+alpha, j+alpha, k)
                        expression.append(clause3)
                    if alpha != 0 and 0 <= i-alpha < size and 0 <= j-alpha < size:
                        # Constant-difference diagonal constraint 2.
                        clause4 = Clause(size)
                        clause4.add_negative(i, j, k)
                        clause4.add_negative(i-alpha, j-alpha, k)
                        expression.append(clause4)
                    if alpha != 0 and 0 <= i+alpha < size and 0 <= j-alpha < size:
                        # Constant-sum diagonal constraint 1.
                        clause5 = Clause(size)
                        clause5.add_negative(i, j, k)
                        clause5.add_negative(i+alpha, j-alpha, k)
                        expression.append(clause5)
                    if alpha != 0 and 0 <= i-alpha < size and 0 <= j+alpha < size:
                        # Constant-sum diagonal constraint 2.
                        clause6 = Clause(size)
                        clause6.add_negative(i, j, k)
                        clause6.add_negative(i-alpha, j+alpha, k)
                        expression.append(clause6)
                    if alpha != k:
                        # Color unicity constraint.
                        clause7 = Clause(size)
                        clause7.add_negative(i, j, k)
                        clause7.add_negative(i, j, alpha)
                        expression.append(clause7)
                        
    if points is not None:
        # Inputs.
        for point in points:
            clause = Clause(size)
            clause.add_positive(point[0], point[1], point[2])
            expression.append(clause)
    return expression


if __name__ == '__main__':
    expression = get_expression(3)
    for clause in expression:
        print(clause)
