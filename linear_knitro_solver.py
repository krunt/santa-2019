
from knitro import *
import math

import numpy as np
import pandas as pd

DAYS = 100
LB = 125
UB = 300

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

try:
    kc = KN_new()
except:
    print ("Failed to find a valid license.")
    quit ()

KN_set_int_param(kc, "mip_method", KN_MIP_METHOD_BB)
KN_set_int_param(kc, "algorithm", KN_ALG_BAR_DIRECT)
KN_set_int_param(kc, "outmode", KN_OUTMODE_FILE)
KN_set_int_param(kc, KN_PARAM_OUTLEV, KN_OUTLEV_SUMMARY)
KN_set_int_param(kc, KN_PARAM_MIP_OUTINTERVAL, 1)
KN_set_int_param(kc, KN_PARAM_MIP_MAXNODES, 10000)
KN_set_int_param(kc, KN_PARAM_MAXIT, 3000)


fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = 'submission71951.csv'
initial_solution = pd.read_csv(fpath, index_col='family_id')

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []

# constraint on sum(wij) = 1
for i in range(data.shape[0]):
    days = list(data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5"]])

    n_people = int(data.iloc[i]["n_people"])

    vs = [] 
    for j, day in enumerate(days):
        var = KN_add_var(kc)
        KN_set_var_types(kc, [var], [KTR_VARTYPE_INTEGER])
        KN_set_var_lobnds(kc, [var], [0])
        KN_set_var_upbnds(kc, [var], [1])
        vs.append(var)
        groupedByDay[day - 1].append((var, n_people))

    familyVars.append(vs)

    cons = KN_add_con(kc)
    KN_set_con_eqbnds(kc, [cons], [1])

    KN_add_con_linear_struct(kc, [cons] * len(vs), vs, [1] * len(vs))

initialVars = []
initialVals = []

# set initial solution
for i in range(data.shape[0]):
    assDay = initial_solution.iloc[i]["assigned_day"]
    days = list(data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4", "choice_5"]])
    assert(len(days) == len(familyVars[i]))
    valS = [0] * len(familyVars[i])
    valS[days.index(assDay)] = 1
    initialVars += familyVars[i]
    initialVals += valS

KN_set_var_primal_init_values(kc, initialVars, initialVals)

# constraint on 125 <= sum(n_people_i * wij) <= 300
for i in range(len(groupedByDay)):
    vs = groupedByDay[i]

    cons = KN_add_con(kc)
    KN_set_con_lobnds(kc, [cons], [LB])
    KN_set_con_upbnds(kc, [cons], [UB])

    varArr = [var for (var, n_people) in vs]
    npeopleArr = [n_people for (var, n_people) in vs]

    KN_add_con_linear_struct(kc, [cons] * len(vs), varArr, npeopleArr)


#def callbackEvalFC (kc, cb, evalRequest, evalResult, userParams):
#    if evalRequest.type != KN_RC_EVALFC:
#        print ("*** callbackEvalFC incorrectly called with eval type %d" % evalRequest.type)
#        return -1
#    x = evalRequest.x
#
#    # Evaluate nonlinear objective structure
#    dTmp1 = x[0] - x[1] + 1.0
#    dTmp2 = x[1] + 1.0
#    evalResult.obj = -18.0*math.log(dTmp2) - 19.2*math.log(dTmp1)
#
#    return 0

#cb = KN_add_eval_callback(kc, evalObj = True, funcCallback = callbackEvalFC)

objVars = []
objCoef = []
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])
    for j, var in enumerate(familyVars[i]):
        objVars.append(var)
        objCoef.append(PENALTY1[j] + PENALTY2[j] * n_people)

KN_add_obj_linear_struct(kc, objVars, objCoef)

# objective function
KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)


nStatus = KN_solve(kc)

print("status="+str(nStatus))

nSTatus, objSol, x_, lambda_ = KN_get_solution (kc)
print ("Optimal objective value  = %e", objSol)

with open("submission_knitro.csv", "w") as fd:
    print("family_id,assigned_day", file=fd)
    for i in range(data.shape[0]):
        lst = [int(x_[int(familyVars[i][j])]) for j in range(len(familyVars[i]))]
        print("%d,%d" % (i, data.iloc[i][lst.index(1)]), file=fd)

## An example of obtaining solution information.
#nSTatus, objSol, x, lambda_ = KN_get_solution (kc)
#print ("Optimal objective value  = %e", objSol)
#print ("Optimal x")
#for i in range (n):
#    print ("  x[%d] = %e" % (i, x[i]))

# Delete the Knitro solver instance.
KN_free(kc)
