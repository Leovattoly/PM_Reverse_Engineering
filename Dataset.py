# %% Gets morphology over time
import sys

# Based on code of Shagayegh Hamzehlou

################################################################################
###########COMMENTS IN CAPITAL LETTTERS ARE WRITTEN BY LEO VATTOLY #############
################################################################################
# %% Import statements
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from numpy import random
import pandas as pd
import random

# %% Input parameters - initial setup
M1_max = 37.4957
M2_max = 87.4904
Max_intervals = 55
Min_intervals = 9
Nintervals = 20  # NEW PARAMETER

# FEED RATE AND TEMPERATURE PARAMETERS ARE NOT INITIALIZED

# gas constant
rg = 8.3144621
# maximum rise in temperature of reactor (K/s)
maxrise = 0.01
# volume of the reactor
v_reactor = 50000
# volume of matrix(pol2)L
Vpol2 = 5000
sc = 0.5
# volume of the water in the seed
vw_0 = 5000

# Average number of radicals per particle
nbarguess = 2
# Avogadro number
NA = 6.023e23
# Initial moles of monomer in the reaction, mol
M1_0 = 0.001
M2_0 = 0.001
M0 = M1_0 + M2_0
# density of monomer kg/L
densmon1 = 0.9
densmon2 = 0.9
# molar mass of monomer kg/mol
wm1 = 0.1
wm2 = 0.1
# density of polymer kg/L
denspol = 1.1
# diameter of particles
dp = 150
# Number of particles
Np = Vpol2 / (4 * (math.pi / 3) * (dp / 2 * 1e-08) ** 3)
# moles of pol2
wm = 0.1
Pol2 = (Vpol2 * denspol) / wm
# Saturation volume fraction for diffusion
phic = 0.0005
# volume fraction for phase separation(nucleation)
phinuc = 0.01
# diffusion coefficient mol/dm2.s
kd1_0 = 7.3e-09
# aggregation coefficient L/s
ka_0 = 6e-18
# ka_0=6e-20; # for case 7
# ka_0=6e-20; # for case 8 Case 2.B
# ka_0=6e-19; # for case 9 Case 2.A
# migration rate 1/s
kmov_0 = 0.05
# kmov_0=0.5e-2; # for case 8 Case 2.B
# kmov_0=0.5e-2; # for case 9 Case 2.A

kmov2_0 = 2e-22 * 1e-05
# nucleation rate coefficient mol/s
kn = 800.0
#
r1 = 0.18
r2 = 0.95
# kinetic chain length in the matrix
xn = 5000
# the number of monomeric unites in the nucleated clusters
xmin = 40000

# Surface tensions
sigma12 = 1.63
sigma13 = 5.66
sigma23 = 4.46
# Tg of Monomer, seed and feed
Tg_M = -110
Tg_Seed = 70
# entry rate
kads = 1E10
# exit rate
kdes = 1e-05

# initial initiator concentration 0.4 mol#
I_0 = 100 / vw_0

# initial volume of particles and fraction of polymer
Vp = Vpol2 + (M1_0 * wm1) / densmon1 + (M2_0 * wm2) / densmon2
phip = Vpol2 / Vp
## define the equilibrium morphology
equilibrium_morphology = 0

T_sigma = abs(sigma13 - sigma12) / sigma23
U_sigma = sigma12 / sigma13
if (T_sigma > 1 and U_sigma > 1):
    equilibrium_morphology = 1
    print('The equilibrium morphology is individual particles')

if (T_sigma > 1 and U_sigma < 1):
    equilibrium_morphology = 2
    print('The equilibrium morphology is inverted core-shell')

if (T_sigma < (abs(1 - U_sigma) / (1 + U_sigma))):
    equilibrium_morphology = 3
    print('The equilibrium morphology is core-shell')

if (equilibrium_morphology == 0):
    equilibrium_morphology = 4
    print('The equilibrium morphology is hemispherical')

# hh=3 + Npivot
# weight fraction of M1/M2
w_f = 30 / 70
# M1 to be fed in kg
w_M1 = 0.3 * 5000
# M2 to be fed in kg
w_M2 = 0.7 * 5000

# moles of M1 to be fed
m_M1 = w_M1 / wm1
# moles of M2 to be fed
m_M2 = w_M2 / wm2

# Fm1_w=Fm1*wm1 # feed of water with monomer 1 (equal mass to monomer)
# Fm2_w=Fm2*wm2 # feed of water with monomer 2 (equal mass to monomer)

T_set = 273.15 + 71  # TEMPERATURE INITIALIZATION (71ºC)

# FeedM1=np.random.normal(1, 0.2,Nintervals)*Fm1
# FeedM1[-1]=0
# FeedM2=np.random.normal(1, 0.2,Nintervals)*Fm2
# FeedM2[-1]=0
# Settemp=np.random.normal(1, 0.05,Nintervals)*(T_set)

## number of pivots(number of ODE's to solve will be = 2*Npivot + 7)

Npivot = 30  # VALUE CHANGES (OLD VALUE WAS 50)

# max number of monomer units in clusters in a particle at the end of reaction
Mmax = M1_0 + M2_0 + m_M1 + m_M2
xavmax = (Mmax * NA) / Np
xmax = (xavmax * 2)
x_star = (0.8 * xavmax)
# width of each pivot
x = np.zeros(Npivot + 1)  # SHAPE OF X WILL BE 31
x[0] = xmin
incx = (xmax - xmin) / (Npivot - 1)
for i in range(1, Npivot + 1):
    x[i] = x[i - 1] + incx  # AP IS INITIALIZING WITH d =  (xmax- xmin) / 30

# y(0)  dimensionless monomer1 (M1/M0)
# y(1)  dimensionless monomer2 (M2/M0)
# y(2)  initiator concenteration
# y(3)  water
# y(4)  initiator concenteration
# y(5)  water
# y(6)  dimensionless polymer 1 (pol1/M0)polymer 1 is a copolymer of
# monomer 1 and 2
# y(7)  dimensionless polymer 1 in matrix(pol1_m/M0)
# y(8) polymer in the clusters
# y(9) normalized number of clusters m (per particle)
# y(10) normalized number of clusters n (per particle)
# y(11) temperature
# y(12)... y(12+Npivot) normalized number non-equilibrium clusters(m/Np)
# y(12+Npivot)...y(12+2Npivot)normalized number equilibrium clusters(n/Np)
M1dim_0 = M1_0 / M0
M2dim_0 = M2_0 / M0
M1fed_0 = M1_0
M2fed_0 = M2_0
P1dim_0 = 0
P1mat_0 = 0
P_cl_0 = 0
nCl_m_0 = 0
nCl_n_0 = 0
temp_0 = T_set  # 71 ºC
non_eq_cl_0 = np.zeros(Npivot)  # 31 VALUES CAN BE ACCOMADATED
eq_cl_0 = np.zeros(Npivot)
if (equilibrium_morphology == 2):
    eq_cl_0[0] = 1
    P1dim_0 = x[0] * (eq_cl_0[0] + eq_cl_0[1]) * Np / (NA * M0)
    P_cl_0 = P1dim_0
    nCl_n_0 = 1
y0 = M1dim_0, M2dim_0, M1fed_0, M2fed_0, I_0, vw_0, P1dim_0, P1mat_0, P_cl_0, nCl_m_0, nCl_n_0, temp_0
y0 = np.append(y0, non_eq_cl_0)
y0 = np.append(y0, eq_cl_0)


# %% Set of ODEs

def deriv(t, y):
    # Concentrations
    M1 = y[0]
    M2 = y[1]
    M1fed = y[2]
    M2fed = y[3]
    I_w = y[4]
    vw = y[5]
    P1 = y[6]
    P1mat = y[7]
    P_cl = y[8]
    nCl_m = y[9]
    nCl_n = y[10]
    temp = y[11]
    non_eq_cl = y[12:12 + Npivot]
    eq_cl = y[12 + Npivot:12 + (2 * Npivot)]

    # Establish zero deriviatives
    dM1dt = 0
    dM2dt = 0
    dM1feddt = 0
    dM2feddt = 0
    dI_wdt = 0
    dvwdt = 0
    dP1dt = 0
    dP1matdt = 0
    dP_cldt = 0
    dnCl_mdt = 0
    dnCl_ndt = 0
    dtempdt = 0
    dnon_eq_cldt = np.zeros(Npivot)
    deq_cldt = np.zeros(Npivot)

    if inhibition_error == 1:
        inh_fac = 1E7
    else:
        inh_fac = 1

    # kinetic constants
    # Polymerization coefficient L/mol.s
    kp1_s = 2.21e7 * np.exp(- 17900 / (rg * temp))
    kp1_t = 1.58e6 * np.exp(- 28900 / (rg * temp))
    kp1 = rate_adjustment_factor * (0.2 * kp1_s + 0.8 * kp1_t)
    kp2 = rate_adjustment_factor * (4.27E7 * np.exp(- 32500 / (rg * temp)))

    # termination rate in particles
    kt_0 = inh_fac * 3.18E7 * np.exp(- 7990 / (rg * temp))
    # termination rate in water
    kt_w = kt_0
    # initiator decomposition
    ki = 8E15 * np.exp(- 135000 / (rg * temp))
    eff = 0.65

    # copolymerization helping variables
    kp21 = kp2 / r2
    kp12 = kp1 / r1
    FP1 = (kp21 * M1) / (kp21 * M1 + kp12 * M2)  # ok
    FP2 = 1 - FP1
    f1 = M1 / (M1 + M2)  # ok
    f2 = 1 - f1
    kp_bar = ((r1 * f1 ** 2) + (2 * f1 * f2) + (r2 * f2 ** 2)) / ((r1 * f1 / kp1) + (r2 * f2 / kp2))
    aux0 = ((wm1 * f1) + (wm2 * f2)) / denspol

    aux00 = ((wm1 * f1) / densmon1) + ((wm2 * f2) / densmon2)

    aux01 = Np / NA
    Vpol1 = P1 * M0 * aux0  # ok

    Vpol = Vpol1 + Vpol2

    Vp = Vpol + (M1 + M2) * M0 * aux00  # ok

    Vpol1m = P1mat * M0 * aux0  # ok

    Vmonm = (M1 + M2) * M0 * aux00 * (Vpol1m + Vpol2) / Vpol  # ok

    Vmatrix = Vpol1m + Vpol2 + Vmonm

    phim = Vmatrix / Vp
    phim2 = ((P1mat * M0) + Pol2) / ((P1 * M0) + Pol2)  # ok
    phip = Vpol / Vp
    phi = max(((Vpol1m / Vmatrix) - phic), 0)
    phin = max(((Vpol1m / Vmatrix) - phinuc), 0)
    phimon = Vmonm / Vmatrix
    # calculating the Tg of the mixture during polymerization
    k_fact = 1.0
    Tg_mix = (Tg_Seed + ((k_fact * Tg_M) - Tg_Seed) * phimon) / (1 + (k_fact - 1) * phimon) + 273.15
    auxtg = (1.36e-05 * np.exp(3.2 / (temp / Tg_mix - 0.866)))
    # viscosity dependent constants
    kd1 = kd1_0 / (auxtg * phip ** 5)
    ka = ka_0 / (auxtg * phip ** 5)
    kmov = kmov_0 / (auxtg * phip ** 5)
    kmov2 = kmov2_0 / (auxtg * phip ** 5)

    # %% nbar calculation

    nbar = nbarguess
    nbar1 = 1
    itera = 0
    kt = kt_0 * np.exp(- 2.23 * phip)
    while abs((nbar1 - nbar) / nbar) > 1e-10:
        c = (kt * Np) / (2 * Vp * NA)
        b = (- kads * Np) / (NA * vw)
        a = - kt_w
        cc = (kdes * nbar * Np) / (NA * vw) + (2 * eff * ki * I_w)
        r_w = (- b - (b ** 2 - (4 * a * cc)) ** 0.5) / (2 * a)
        f = 2 * (2 * kads * r_w + kdes) / (2 * kads * r_w + kdes + c)
        nbar1 = nbar
        nbar = 2 * kads * r_w / (kdes + (kdes ** 2 + 4 * kads * r_w * c * f) ** 0.5)
        itera = itera + 1
        # nbarguess=nbar
        if (itera > 1000):
            nbar = nbar1
            print('error in iteration nbar')
    # print(nbar)

    # helping variables to calculate polymerization
    aux4 = nbar * Np * tf / (Vp * NA);
    if (t <= (t_feed / tf)) and (M1fed + M2fed < Mmax):
        # Conditions to stop feeding
        dM1dt += (-kp1 * aux4 * M1 * FP1 - kp21 * aux4 * M1 * FP2 + Fm1 * tf / M0)
        dM2dt += (-kp12 * aux4 * M2 * FP1 - kp2 * aux4 * M2 * FP2 + Fm2 * tf / M0)
    else:
        dM1dt += (-kp1 * aux4 * M1 * FP1 - kp21 * aux4 * M1 * FP2)
        dM2dt += (-kp12 * aux4 * M2 * FP1 - kp2 * aux4 * M2 * FP2)
    # ok
    dP1dt += aux4 * M1 * (kp1 * FP1 + kp21 * FP2) + aux4 * M2 * (kp12 * FP1 + kp2 * FP2)  # ok

    # looks good to here
    # total surface of clusters
    atotclusters = 0
    if (Vp - Vmatrix > 1e-10):
        phip_m_n = (P1 * M0 - P1mat * M0) * aux0 / (Vp - Vmatrix)  # ok
        for i in range(Npivot):
            atotclusters = atotclusters + ((6 * math.sqrt(math.pi)) ** (2 / 3)) * (
                    (x[i] * aux0 / (phip_m_n * NA)) ** (2 / 3)) * (non_eq_cl[i] + eq_cl[i])  # ok
    else:
        atotclusters = 0

    # helping variables to calculate rates - all ok
    auxp1 = (M1 * (kp1 * FP1 + kp21 * FP2) + M2 * (kp12 * FP1 + kp2 * FP2)) * nbar * Np * tf / (Vp * NA)  # ok
    phix = 1 / (NA * M0 * (P1 + Pol2 / M0))  # ok
    auxp = phix * auxp1
    auxd = kd1 * phi * NA * tf / xn
    auxc = ka * Np * tf / (Vp)
    auxn = kn * phin * NA * tf / (x[0] * Np)  # ok
    aux_mov = kmov2 * Np * tf / (Vp)

    dP1matdt += ((aux4 * M1 * (kp1 * FP1 + kp21 * FP2) + aux4 * M2 * (
            kp12 * FP1 + kp2 * FP2)) * phim - kn * phin * tf / M0 - kd1 * phi * tf * Np * atotclusters / M0)  # ok
    xn_bar = 0.0
    xm_bar = 0.0
    for j in range(Npivot):
        xn_bar = xn_bar + eq_cl[j] * x[j]  # ok
        xm_bar = xm_bar + non_eq_cl[j] * x[j]  # ok

    a = np.zeros(Npivot)
    ad = np.zeros(Npivot)
    bd = np.zeros(Npivot)
    ap = np.zeros(Npivot)
    bp = np.zeros(Npivot)
    p = np.zeros(Npivot)
    d = np.zeros(Npivot)
    sumt1 = np.zeros(Npivot)
    sumt2 = np.zeros(Npivot)
    coag1 = np.zeros(Npivot)
    coag2 = np.zeros(Npivot)
    coag3 = np.zeros(Npivot)

    for i in range(Npivot):  # ok
        # partition coefficients of KR method
        a[i] = x[i + 1] - x[i]
        ad[i] = (x[i + 1] - (x[i] + xn)) / a[i]
        bd[i] = 1 - ad[i]
        ap[i] = ((x[i + 1] - (x[i] + 1)) / a[i])
        bp[i] = 1 - ap[i]
        p[i] = auxp * x[i] * NA * M0;
        d[i] = auxd * 4.836 * (x[i] * aux0 / (phip * NA)) ** 0.6667
        # disapearing term of coagulation
        alfa1 = 0.8
        sumt1[i] = 0
        sumt2[i] = 0
        for h in range(Npivot - i - 1):  # ok
            alfa_m = 1
            alfa_n = 1
            aux1 = x[i + h + 1]  # ok
            if (aux1 > xmax):
                alfa_m = 0
                alfa_n = 0
            else:
                if (aux1 > alfa1 * (xavmax - xn_bar)) and (aux1 <= xmax):
                    alfa_m = np.exp(-40 * (aux1 - alfa1 * (xavmax - xn_bar)) / (alfa1 * xavmax))
                if (aux1 > alfa1 * (xavmax - xm_bar)) and (aux1 <= xmax):
                    alfa_n = np.exp(-40 * (aux1 - alfa1 * (xavmax - xm_bar)) / (alfa1 * xavmax))
            # zigma alfa_i*m(i)
            sumt1[i] = sumt1[i] + alfa_m * non_eq_cl[h]  # ok to here
            sumt2[i] = sumt2[i] + alfa_n * eq_cl[h]  # ok to here

        # alfa is to reduce the probability of coagulation of big clusters
        # apearing term of coagulation
        if (x[i] <= alfa1 * (xavmax - xn_bar)):
            alfa_m = 1
        if (x[i] <= alfa1 * (xavmax - xm_bar)):
            alfa_n = 1
        if (x[i] > alfa1 * (xavmax - xn_bar)) and (x[i] <= xmax):
            alfa_m = np.exp(-40 * (x[i] - alfa1 * (xavmax - xn_bar)) / (alfa1 * xavmax))
        if (x[i] > alfa1 * (xavmax - xm_bar)) and (x[i] <= xmax):
            alfa_n = np.exp(-40 * (x[i] - alfa1 * (xavmax - xm_bar)) / (alfa1 * xavmax))
        if (x[i] > xmax):
            alfa_n = 0
            alfa_m = 0
        coag1[i] = 0
        coag2[i] = 0
        coag3[i] = 0
        # for j in range(i - 1):
        for j in range(i):
            aux7 = x[j] + x[i - j - 1]  # ok
            bc = (aux7 - x[i - 1]) / (x[i] - x[i - 1])  # ok
            ba = 1 - bc
            coag1[i] += alfa_m * bc * non_eq_cl[j] * non_eq_cl[i - j - 1]
            coag1[i - 1] += alfa_m * (1 - bc) * non_eq_cl[j] * non_eq_cl[i - j - 1]
            coag2[i] += alfa_n * bc * eq_cl[j] * eq_cl[i - j - 1]
            coag2[i - 1] += alfa_n * (1 - bc) * eq_cl[j] * eq_cl[i - j - 1]
            coag3[i] += alfa_m * bc * eq_cl[i - j - 1] * non_eq_cl[j]
            coag3[i - 1] += alfa_m * (1 - bc) * eq_cl[i - j - 1] * non_eq_cl[j]

    sum_n = 0
    for i in range(Npivot):
        sum_n = sum_n + x[i] * (eq_cl[i])  # ok
    # total number of clusters
    m_t = 0
    n_t = 0
    for i in range(Npivot):
        m_t = m_t + non_eq_cl[i]  # ok
        n_t = n_t + eq_cl[i]  # ok

    fac_m = 0
    fac_n = 0
    if (m_t > 1):
        fac_m = 1 - (1 / m_t)

    if (n_t > 1):
        fac_n = 1 - (1 / n_t)

    if (equilibrium_morphology == 4):
        # Population balances
        for i in range(Npivot):
            aox_p = 1
            if (i == Npivot - 1):
                aox_p = 0
            # aox=i - 1
            # if (aox <= 0):
            if i == 0:
                dnon_eq_cldt[i] += -bp[i] * p[i] * non_eq_cl[i] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                    i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - kmov * tf * non_eq_cl[i] + auxn  # ok
                deq_cldt[i] += -bp[i] * p[i] * eq_cl[i] - bd[i] * d[i] * eq_cl[i] + auxc * coag2[i] * fac_n - 2 * auxc * \
                               eq_cl[i] * sumt2[i] * fac_n + kmov * tf * non_eq_cl[i]  # ok
            else:
                dnon_eq_cldt[i] += bp[i - 1] * p[i - 1] * non_eq_cl[i - 1] - aox_p * bp[i] * p[i] * non_eq_cl[i] + bd[
                    i - 1] * d[i - 1] * non_eq_cl[i - 1] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                                       i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - kmov * tf * non_eq_cl[
                                       i]
                deq_cldt[i] += bp[i - 1] * p[i - 1] * eq_cl[i - 1] - aox_p * bp[i] * p[i] * eq_cl[i] + bd[i - 1] * d[
                    i - 1] * eq_cl[i - 1] - bd[i] * d[i] * eq_cl[i] + auxc * coag2[i] * fac_n - 2 * auxc * eq_cl[
                                   i] * fac_n * sumt2[i] + kmov * tf * non_eq_cl[i]

    # the morphology is inverted core-shell
    if (equilibrium_morphology == 2):
        # Population balances
        for i in range(Npivot):
            aox_p = 1
            if (i == Npivot - 1):
                aox_p = 0
            # aox=i - 1
            # if (aox <= 0):
            if i == 0:
                dnon_eq_cldt[i] += -bp[i] * p[i] * non_eq_cl[i] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                    i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - aux_mov * non_eq_cl[i] * sumt2[i] + auxn
                deq_cldt[i] += -bp[i] * p[i] * eq_cl[i] - bd[i] * d[i] * eq_cl[i] + aux_mov * coag3[i] - aux_mov * \
                               eq_cl[i] * sumt1[i]
            else:
                dnon_eq_cldt[i] += bp[i - 1] * p[i - 1] * non_eq_cl[i - 1] - aox_p * bp[i] * p[i] * non_eq_cl[i] + bd[
                    i - 1] * d[i - 1] * non_eq_cl[i - 1] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                                       i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - aux_mov * non_eq_cl[
                                       i] * sumt2[i]
                deq_cldt[i] += bp[i - 1] * p[i - 1] * eq_cl[i - 1] - aox_p * bp[i] * p[i] * eq_cl[i] + bd[i - 1] * d[
                    i - 1] * eq_cl[i - 1] - bd[i] * d[i] * eq_cl[i] + aux_mov * coag3[i] - aux_mov * eq_cl[i] * sumt1[i]

    if (equilibrium_morphology == 3):
        sum_t = 0
        for i in range(Npivot):
            sum_t = sum_t + x[i] * (eq_cl[i])
        if (sum_t < x_star):
            # Population balances
            for i in range(Npivot):
                aox_p = 1
                if (i == Npivot - 1):
                    aox_p = 0
                # aox=i - 1
                # if (aox <= 0):
                if i == 0:
                    dnon_eq_cldt[i] += -bp[i] * p[i] * non_eq_cl[i] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                        i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - kmov * tf * non_eq_cl[i] + auxn
                    deq_cldt[i] += -bp[i] * p[i] * eq_cl[i] - bd[i] * d[i] * eq_cl[i] + auxc * coag2[
                        i] * fac_n - 2 * auxc * eq_cl[i] * sumt2[i] * fac_n + kmov * tf * non_eq_cl[i]
                else:
                    dnon_eq_cldt[i] += bp[i - 1] * p[i - 1] * non_eq_cl[i - 1] - aox_p * bp[i] * p[i] * non_eq_cl[i] + \
                                       bd[i - 1] * d[i - 1] * non_eq_cl[i - 1] - bd[i] * d[i] * non_eq_cl[i] + auxc * \
                                       coag1[i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - kmov * tf * \
                                       non_eq_cl[i]
                    deq_cldt[i] += bp[i - 1] * p[i - 1] * eq_cl[i - 1] - aox_p * bp[i] * p[i] * eq_cl[i] + bd[i - 1] * \
                                   d[i - 1] * eq_cl[i - 1] - bd[i] * d[i] * eq_cl[i] + auxc * coag2[
                                       i] * fac_n - 2 * auxc * eq_cl[i] * fac_n * sumt2[i] + kmov * tf * non_eq_cl[i]
        if (sum_t >= x_star):
            for i in range(Npivot):
                aox_p = 1
                if (i == Npivot - 1):
                    aox_p = 0
                # aox=i - 1
                # if (aox <= 0):
                if i == 0:
                    dnon_eq_cldt[i] += -bp[i] * p[i] * non_eq_cl[i] - bd[i] * d[i] * non_eq_cl[i] + auxc * coag1[
                        i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - aux_mov * non_eq_cl[i] * sumt2[
                                           i] + auxn
                    deq_cldt[i] += -bp[i] * p[i] * eq_cl[i] - bd[i] * d[i] * eq_cl[i] + aux_mov * coag3[i] - aux_mov * \
                                   eq_cl[i] * sumt1[i];
                else:
                    dnon_eq_cldt[i] += bp[i - 1] * p[i - 1] * non_eq_cl[i - 1] - aox_p * bp[i] * p[i] * non_eq_cl[i] + \
                                       bd[i - 1] * d[i - 1] * non_eq_cl[i - 1] - bd[i] * d[i] * non_eq_cl[i] + auxc * \
                                       coag1[i] * fac_m - 2 * auxc * non_eq_cl[i] * sumt1[i] * fac_m - aux_mov * \
                                       non_eq_cl[i] * sumt2[i]
                    deq_cldt[i] += bp[i - 1] * p[i - 1] * eq_cl[i - 1] - aox_p * bp[i] * p[i] * eq_cl[i] + bd[i - 1] * \
                                   d[i - 1] * eq_cl[i - 1] - bd[i] * d[i] * eq_cl[i] + aux_mov * coag3[i] - aux_mov * \
                                   eq_cl[i] * sumt1[i];

    Fm1_w = Fm1 * wm1  # feed of water with monomer 1 (equal mass to monomer)
    Fm2_w = Fm2 * wm2  # feed of water with monomer 2 (equal mass to monomer)

    if M1fed + M2fed < Mmax:
        dM1feddt += Fm1 * tf
        dM2feddt += Fm2 * tf
        dvwdt += Fm1_w * tf + Fm2_w * tf
    if T_contol_error == 1:
        delta_h_1 = 78.2
        delta_h_2 = 73
        Heat_release = (kp1 * aux4 * M1 * FP1 + kp21 * aux4 * M1 * FP2) * delta_h_1 + (
                kp12 * aux4 * M2 * FP1 + kp2 * aux4 * M2 * FP2) * delta_h_2
        # print(Heat_release/(2*vw*4.18))
        # print(maxrise*tf)
        dtempdt += Heat_release / (2 * vw * 4.18)
        if temp > 373:  # if the temperature exceeds 100 degrees an upper limit is reached due to water boiling
            dtempdt = 0
    else:
        if temp < T_set:
            dtempdt += maxrise * tf
        if temp > T_set:
            dtempdt -= maxrise * tf
        if temp == T_set:
            dtempdt = 0
    sum_t = 0  # ok from here
    for i in range(Npivot):
        sum_t += x[i] * (non_eq_cl[i] + eq_cl[i])
    # z(4+2*Npivot+1)= kp_bar*(y(1)+y(2))*nbar*Np*tf/(Vp*NA);
    auxdiff = (M1 * (kp1 * FP1 + kp21 * FP2) + M2 * (kp12 * FP1 + kp2 * FP2)) * nbar * Np * tf / (Vp * NA)
    dP_cldt = auxdiff * phix * Np * sum_t + kn * phin * tf / M0 + kd1 * phi * tf * Np * atotclusters / M0
    dnCl_mdt += np.sum(dnon_eq_cldt)
    dnCl_ndt += np.sum(deq_cldt)
    dI_wdt += -ki * I_w * tf + (Fi / vw) * tf

    dydt = dM1dt, dM2dt, dM1feddt, dM2feddt, dI_wdt, dvwdt, dP1dt, dP1matdt, dP_cldt, dnCl_mdt, dnCl_ndt, dtempdt
    dydt = np.append(dydt, dnon_eq_cldt)
    dydt = np.append(dydt, deq_cldt)
    return dydt


# %% Solution and printing

t0int, tfint = 0, 1

appended_data = []

# SEQUENCE OF REACTION

# Generate set of reactions different feed profiles
nReactions = int(sys.argv[2])  # number of reactions to run to store data
timeseries_interval = 400  # interval (in seconds) to store data for time series   # TIME SERIES DATA INTERVAL
tfmax = 6 * 60 * 60  # maximum reaction time   # 6 HRS
tfmin = 1 * 60 * 60  # minimum reaction time   # 1 HR

# ADDING NOISE

# Three potential disturbances are included contolled by
# 1) 'p_inhibition_error'- error that leads to no polymerization (i.e. due to impurities or oxygen in reaction)
# 2) 'p_T_contol_error' - error in temperature control that leads to adiabatic conditions (no heat removal)
# 3) 'rate_adjustment_factor' - a factor that leads to some stochastic differences in the rate of reaction due to random fluctuations in chemicals and reactor conditions

p_inhibition_error = 0  # 0.01 # probability that during a given interval there is some inhibition error leading to no conversion
p_T_contol_error = 0  # 0.01 # probability that during a given interval there is some temperature control error leading to adiabatic conditions until boiling point of water

# the next section of code generates a feed profile for the reaction and the solves the set of odes in stages, loggin the result at each one

for i in range(nReactions):  # 10 TIMES ITERATION

    rate_adjustment_factor = 1  # abs(np.random.normal(1, 0.25)) # disturbance that gives different rates (i.e. due to retardation effects). # MEAN =1 and std.DEV = 0.25
    tf = np.random.randint(tfmin, tfmax)  # get reaction times between 1 and 6 hr

    Nintervals = int(
        np.floor(tf / timeseries_interval))  # get number of intervals in the reaction.  #  [9-54]

    t_feed = tf - 20 * 60  # feeding time is final time minus 20 minutes   # FEED TIME IN SECONDS AND RANDOM FOR EACH REACTIONS.     [2400 - 20400]

    Fi = 100 / t_feed  # amount of initiator is constant and fed over the course of the reaction

    Fm1av = m_M1 / t_feed  # gets an average feed rate based on the selected feed time
    Fm2av = m_M2 / t_feed

    FeedM1 = abs(np.random.normal(2, 1.0,
                                  Nintervals)) * Fm1av  # gets a bunch of different feeds centered around the average    #Nintervals = [9-54]  VALUE WILL BE BETWEEN (1,0.3)* Fm1av WITH ARRAY SIZE OF Nintervals
    FeedM2 = abs(np.random.normal(2, 1.0,
                                  Nintervals)) * Fm2av  # gets a bunch of different feeds centered around the average
    FeedM1[int(-(
            20 * 60 / timeseries_interval)):] = 0  # no feeding in last twenty minutes.       #WHY !!?     FeedM1[-3:] = 0 LAST 3 ELEMENTS ARE REPLACING AS ZERO
    FeedM2[
    int(-(20 * 60 / timeseries_interval)):] = 0  # COOKING TIME ALL MONOMERS NEEDS TO BE COOKED DURING THE THIS TIME



    # need to correct the feed rates to get the same final amounts of monomer in every reaction
    FeedM1error = np.sum(FeedM1 * timeseries_interval) - m_M1
    while abs(FeedM1error / m_M1) > 0.001:
        if FeedM1error < 0:
            FeedM1 = FeedM1 * 1.001
        if FeedM1error > 0:
            FeedM1 = FeedM1 * 0.999
        FeedM1error = np.sum(FeedM1 * timeseries_interval) - m_M1

    FeedM2error = np.sum(FeedM2 * timeseries_interval) - m_M2
    while abs(FeedM2error / m_M2) > 0.001:
        if FeedM2error < 0:
            FeedM2 = FeedM2 * 1.001
        if FeedM2error > 0:
            FeedM2 = FeedM2 * 0.999
        FeedM2error = np.sum(FeedM2 * timeseries_interval) - m_M2

    # get temperature profile randomly varied around the average using the normal distribution

    T_set = np.random.randint( 60,80)  # TEMPERATURE [60-80] INTEGER VALUES
    Settemp = [273.15 + (
        T_set)] * Nintervals  # gets a bunch of different feeds centered around a certain value for the reaction # REMOVED THE RANDOMNESS INT TEMP
    y0[11] = Settemp[0]

    inttime = 0
    yy = y0
    tt = 0
    ystart = y0

    for j in range(Nintervals):

        Fm1 = FeedM1[j]
        Fm2 = FeedM2[j]
        T_set = Settemp[j]

        if np.random.rand() < p_inhibition_error:
            inhibition_error = 1
        else:
            inhibition_error = 0

        if np.random.rand() < p_T_contol_error:
            T_contol_error = 1
            print('Terror')
        else:
            T_contol_error = 0

        soln = solve_ivp(deriv, (inttime, inttime + timeseries_interval / tf), ystart,
                         method='RK45')  # ,atol=1E-15)     #solve_ivp(func,(to,tf),initial_values,method = Runge-Kutta method )
        inttime += timeseries_interval / tf
        ystart = soln.y[:, -1]  # STORED ONLY LAST (3RD) COLUMN DATA FROM THE SOLUTION OF TIME AT soln.t
        # #print("Stored Data :",ystart)
        # #print("Tolatal Solution",soln.y)
        # 	    yall=soln.y
        # 	    yy = np.c_[yy, yall]
        # 	    time=soln.t
        # 	    tt= np.append(tt, time)
        yy = np.c_[yy, soln.y[:, -1]]  # [[YY1,1],[YY2,2],[YY3,3]....[YYN,N]]
        tt = np.append(tt, soln.t[-1] * tf)

        M1fed = soln.y[2, -1]
        M2fed = soln.y[3, -1]
        Mi = M1_0 + M2_0 + M1fed + M2fed;
        X_inst = ((Mi - (soln.y[0, -1] + soln.y[1, -1]) * M0) + Pol2) / (Mi + Pol2)

        delta_h_1 = 78.2  # kJ/mol for BA
        delta_h_2 = 73
        Heat_reaction = (M1_0 + M1fed - (soln.y[0, -1] * M0)) * delta_h_1 + (
                M2_0 + M2fed - (soln.y[1, -1] * M0)) * delta_h_2
        f1 = soln.y[1, -1] / soln.y[0, -1]
        C_inst = (r1 + f1) / (r1 + 2 * f1 + r2 * f1 ** 2)

        if j == (Nintervals - 1):
            diff = Max_intervals - len(FeedM1)

            add_values = np.zeros(diff) # Zero added for making all rows in equal length
            FeedM1 = np.append(FeedM1, add_values)
            FeedM2 = np.append(FeedM2, add_values)

            new_row = {'Temperature': T_set - 273.15, 'Intervales': Nintervals}

            for kk in range(len(FeedM1)):
                new_row['M1fed'f"{kk}"] = FeedM1[kk]

            for kk in range(len(FeedM2)):
                new_row['M2fed'f"{kk}"] = FeedM2[kk]

            non_eq_cl = np.zeros(Npivot + 1)
            eq_cl = np.zeros(Npivot + 1)

            non_eq_cl[0:Npivot] = soln.y[12:12 + Npivot, -1]
            eq_cl[0:Npivot] = soln.y[12 + Npivot:12 + (2 * Npivot), -1]
            for jj in range(0, Npivot - 1):
                new_row['non_eq_cl'f"{jj}"] = non_eq_cl[jj] * x[jj]
            for jj in range(0, Npivot - 1):
                new_row['eq_cl'f"{jj}"] = eq_cl[jj] * x[jj]

            new_df = pd.DataFrame([new_row])
            # print(new_row)
            appended_data.append(new_df)

            dfseries = pd.concat(appended_data)

dfseries.to_csv("Data/" + sys.argv[1] + "_data.csv", index=False)  # SAVING DATA in COLAB

# python Dataset.py <#train/test/opt <#Reactoin number>
