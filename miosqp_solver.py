
import numpy as np
import pandas as pd
import miosqp

import scipy as sp
import scipy.sparse as spa
import numpy as np
import pandas as pd

from tqdm import tqdm

DAYS = 100
LB = 125
UB = 300
BRANGE = (UB - LB + 1)
CHOICE_NUM = 5

PENALTY1 = [ 0, 50, 50, 100, 200, 200, 300, 300, 400, 500, 500 ]
PENALTY2 = [ 0, 0, 9, 9, 9, 18, 18, 36, 36, 36 + 199, 36 + 398 ]

fpath = 'family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

NFAMILY_VARS = data.shape[0] * CHOICE_NUM
NDAYOCCUP_VARS = BRANGE * (DAYS + 1)
TOT_VARS = NFAMILY_VARS + NDAYOCCUP_VARS

q = np.zeros(TOT_VARS)
i_idx = np.arange(TOT_VARS)
i_l = np.zeros(TOT_VARS)
i_u = np.ones(TOT_VARS)

groupedByDay = []
for i in range(DAYS):
    groupedByDay.append([])

familyVars = []
for i in range(data.shape[0]):
    days = data.iloc[i][["choice_0", "choice_1", "choice_2", "choice_3", "choice_4"]]
    assert(len(days) == CHOICE_NUM)

    n_people = int(data.iloc[i]["n_people"])

    vs = [] 
    for j, day in enumerate(days):
        vs.append(i * CHOICE_NUM + j)
        groupedByDay[day - 1].append((vs[-1], n_people))

    familyVars.append(vs)

# preference cost
for i in range(data.shape[0]):
    n_people = int(data.iloc[i]["n_people"])

    for j, var in enumerate(familyVars[i]):
        q[var] = PENALTY1[j] + PENALTY2[j] * n_people

# accounting cost
def get_acc_penalty(day0, day1, nday0, nday1):
    ndaydiff = 0 if day1 == DAYS else abs(nday0 - nday1)
    return (nday0 - 125.0) / 400.0 * (nday0 ** (0.5 + ndaydiff / 50.0))

P_row = []
P_col = []
P_data = []
for day in range(1, DAYS + 1):
    day0 = day - 1
    day1 = day

    offs = NFAMILY_VARS 
    for i in range(BRANGE):
        for j in range(BRANGE):
            nday0 = LB + i
            nday1 = LB + j
            penalty = get_acc_penalty(day0, day1, nday0, nday1)
            P_row.append(NFAMILY_VARS + day0 * BRANGE + i)
            P_col.append(NFAMILY_VARS + day1 * BRANGE + j)
            P_data.append(penalty)

P = spa.csc_matrix((P_data, (P_row, P_col)), shape=(TOT_VARS, TOT_VARS))

lcon = []
ucon = []
Acon = []

def add_con(coef, lhs, rhs):
    Acon.append(coef)
    lcon.append(lhs)
    ucon.append(rhs)

for i in range(data.shape[0]):
    vals = []
    for j in range(CHOICE_NUM):
        vals.append((i * CHOICE_NUM + j, 1))
    add_con(vals, 1, 1)

for i in range(DAYS):
    vals = []
    for (var, n_people) in groupedByDay[i]:
        vals.append((var, n_people))
    add_con(vals, LB, UB)

for i in range(DAYS + 1):
    vals = []
    for j in range(BRANGE):
        var = NFAMILY_VARS + i * BRANGE + j
        vals.append((var, 1))
    add_con(vals, 1, 1)

lcon = np.array(lcon)
ucon = np.array(ucon)

A_row = []
A_col = []
A_data = []
for i in range(len(Acon)):
    vs = Acon[i]
    for (var, coef) in vs:
        A_row.append(i)
        A_col.append(var)
        A_data.append(coef)

A = spa.csc_matrix((A_data, (A_row, A_col)), shape=(lcon.shape[0], TOT_VARS))

# solver part
model = miosqp.MIOSQP()
miosqp_settings = {
                   # integer feasibility tolerance
                   'eps_int_feas': 1e-03,
                   # maximum number of iterations
                   'max_iter_bb': 1000,
                   # tree exploration rule
                   #   [0] depth first
                   #   [1] two-phase: depth first until first incumbent and then  best bound
                   'tree_explor_rule': 1,
                   # branching rule
                   #   [0] max fractional part
                   'branching_rule': 0,
                   'verbose': False,
                   'print_interval': 1}

osqp_settings = {'eps_abs': 1e-03,
                 'eps_rel': 1e-03,
                 'eps_prim_inf': 1e-04,
                 'verbose': False}
model.setup(P, q, A, lcon, ucon, i_idx, i_l, i_u,
            miosqp_settings,
            osqp_settings)
res_miosqp = model.solve()
print(res_miosqp["status"])
