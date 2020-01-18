import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

DAYS = 100
LB = 125
UB = 300
BRANGE = UB - LB + 1

PREF_COST = 62868
ACC_COST = 6020.043432

solver = pywraplp.Solver('cbc_program', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

fpath = 'family_data.csv'
families = pd.read_csv(fpath, index_col='family_id')

totPeople = 0
for i in range(families.shape[0]):
    totPeople += int(families.iloc[i]["n_people"])


dayList = []
costList = []
varList = []
for d0 in range(BRANGE):
    for d1 in range(BRANGE):
        nday0 = d0 + LB
        nday1 = d1 + LB

        ndaydiff = abs(nday0 - nday1)

        penalty = (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

        dayList.append(nday0)
        costList.append(penalty)
        varList.append(solver.IntVar(0, DAYS, "dvar_%d" % len(varList)))


cons = solver.Constraint(totPeople, totPeople)
for i in range(len(dayList)):
    cons.SetCoefficient(varList[i], dayList[i])

cons = solver.Constraint(DAYS, DAYS)
for i in range(len(dayList)):
    cons.SetCoefficient(varList[i], 1)

TOL = 0.001
cons = solver.Constraint(ACC_COST - TOL, ACC_COST + TOL)
for i in range(len(dayList)):
    cons.SetCoefficient(varList[i], costList[i])

solver.SetNumThreads(4)
#solver.EnableOutput()
#solver.SetTimeLimit(120000)

status = solver.Solve()

if status == solver.OPTIMAL or status == solver.FEASIBLE:
    print(status)
    for i in range(len(dayList)):
        nVar = int(varList[i].solution_value())
        if nVar > 0:
            print("%d,%d,%d" % (nVar, dayList[i], i))
