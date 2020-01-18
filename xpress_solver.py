from __future__ import print_function
import xpress as xp
import pandas as pd

problem = xp.problem()
problem.read('milp_santa.lp')

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

sol_fpath = 'submission68910.csv'
sol = pd.read_csv(sol_fpath, index_col='family_id')

var_names = []
var_values = []
for i in range(data.shape[0]):
    days = data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"]]
    assigned_day = int(sol.iloc[i]["assigned_day"])

    for j, day in enumerate(days):
        var_names.append("fam_w%d_%d" % (i, j))
        var_values.append(1 if day == assigned_day else 0)

problem.addmipsol(var_values, var_names)

problem.setControl('miprelstop', 1e-9)

problem.solve()

#print("solution:", problem.getSolution())

import pdb; pdb.set_trace()

var_found_values = problem.getSolution(var_names)

with open("submission_xpress.csv", "w") as fd:
    print("family_id,assigned_day", file=fd)
    for i in range(data.shape[0]):
        lst = [int(var) for var in var_found_values[i//5:(i+5)//5]]
        print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

import pdb; pdb.set_trace()
