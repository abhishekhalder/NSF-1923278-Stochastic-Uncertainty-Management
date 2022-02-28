import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from itertools import cycle
from pylab import *
import glob
from pylab import rcParams
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import interp1d
from fractions import Fraction as frac

#====================================================
# Make plots beautiful
#====================================================
myalphavalue=0.3
pts_per_inch = 72.27
# write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
text_width_in_pts = 300.0
# inside a figure environment in latex, the result will be on the
# dvi/pdf next to the figure. See url above.
text_width_in_inches = text_width_in_pts / pts_per_inch
# figure.png or figure.eps will be intentionally larger, because it is prettier
inverse_latex_scale = 2
fig_proportion = (3.0 / 3.0)
csize = inverse_latex_scale * fig_proportion * text_width_in_inches
# always 1.0 on the first argument
fig_size = (1.0 * csize, 0.85 * csize)
# find out the fontsize of your latex text, and put it here
text_size = inverse_latex_scale * 9
label_size = inverse_latex_scale * 10
tick_size = inverse_latex_scale * 8
# learn how to configure:
# http://matplotlib.sourceforge.net/users/customizing.html
params = {'backend': 'ps',
          'axes.labelsize': 16,
          'legend.fontsize': tick_size,
          'legend.handlelength': 2.5,
          'legend.borderaxespad': 0,
          'axes.labelsize': label_size,
          'xtick.labelsize': tick_size,
          'ytick.labelsize': tick_size,
          'font.family': 'serif',
          'font.size': text_size,
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          'figure.figsize': fig_size,
          # include here any neede package for latex
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          }
plt.rcParams.update(params)
fig = plt.figure(1, figsize=fig_size)  # figsize accepts only inches.
# fig.subplots_adjust(left=0.04, right=0.98, top=0.93, bottom=0.15,
#                     hspace=0.05, wspace=0.02)

# =============================================================================
# Computational time plot [IEEE 14 bus 5 gen, Case I and Case II]
# =============================================================================
Time_Synthetic = np.loadtxt("TimeSyntheticIEEE14bus.txt")
# CompTime_Synthetic_Case1 = np.loadtxt("case0_norminal_ComptimeSytheticIEEE14bus.txt")
# CompTime_Synthetic_Case2 = np.loadtxt("case1_line_13_failure_ComptimeSytheticIEEE14bus.txt")

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# ax.semilogy(Time_Synthetic, CompTime_Synthetic_Case1,'-', color='b', lw=1.5, alpha=0.7,label=r"Case I")
# ax.semilogy(Time_Synthetic, CompTime_Synthetic_Case2,':', color='r', lw=1.5, alpha=0.7,label=r"Case II")

# # ax.set_ylim(10**-3, 6*10**-2)

# ax.tick_params(direction='in',which='both')
# #  # axx.xaxis.tick_top()
# ax.grid(True,which="both",ls="-", color='0.75')
# # # axx2.grid(True,which="both",ls="-", color='0.75')
# ax.tick_params(axis='both', labelsize=18)


# ax.set_ylabel(r"Computational time [s]")
# ax.set_xlabel(r"Physical time $t=kh$ [s]")
# # # # axx.yaxis.set_label_coords(-0.125,-0.05)

# ax.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
# fig.set_size_inches(10.2, 6.2)

# #plt.savefig('ComputationalTimeSynthetic50Gen.png', dpi=400)
# plt.savefig('ComputationalTimeSyntheticIEEE14busCase1and2combined.png', dpi=400)

# =============================================================================
# Computational time plot [50 gen synthetic]
# =============================================================================
# Time_Synthetic = np.loadtxt("TimeSynthetic50Gen.txt")
# CompTime_Synthetic = np.loadtxt("ComptimeSythetic50Gen.txt")

# Time_Synthetic = np.loadtxt("TimeSyntheticIEEE14bus.txt")
# CompTime_Synthetic = np.loadtxt("ComptimeSytheticIEEE14bus.txt")

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# ax.semilogy(Time_Synthetic, CompTime_Synthetic,'-', color='b', lw=1.5)

# ax.set_ylim(10**-3, 6*10**-2)

# ax.tick_params(direction='in',which='both')
#  # axx.xaxis.tick_top()

# ax.grid(True,which="both",ls="-", color='0.75')
# # axx2.grid(True,which="both",ls="-", color='0.75')
# ax.tick_params(axis='both', labelsize=18)


# ax.set_ylabel(r"Computational time [s]")
# ax.set_xlabel(r"Physical time $t=kh$ [s]")
# # # axx.yaxis.set_label_coords(-0.125,-0.05)

# # axx.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
# fig.set_size_inches(10.2, 6.2)

# # plt.savefig('ComputationalTimeSynthetic50Gen.png', dpi=400)
# plt.savefig('ComputationalTimeSyntheticIEEE14bus.png', dpi=400)

# =============================================================================
# Relative error in mean vector: MC versus Prox plot
# =============================================================================
# RelErrMeanVectorMCvsProx_Synthetic = np.loadtxt("RelErrMeanVectorMCvsProxSythetic50Gen.txt")

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

# ax.semilogy(Time_Synthetic[3:-1], RelErrMeanVectorMCvsProx_Synthetic[4:-1],'-', color='k', lw=1.5)

# # ax.set_ylim(1*10**-3, 1.2*10**-2)

# ax.tick_params(direction='in',which='both')
#  # axx.xaxis.tick_top()

# ax.grid(True,which="both",ls="-", color='0.75')
# # axx2.grid(True,which="both",ls="-", color='0.75')
# ax.tick_params(axis='both', labelsize=18)


# ax.set_ylabel(r"Realtive error $\frac{\|\boldsymbol{\mu}_{k}^{\rm{MC}}-\boldsymbol{\mu}_{k}^{\rm{Prox}}\|_{2}}{\|\boldsymbol{\mu}_{k}^{\rm{MC}}\|_{2}}$")
# ax.set_xlabel(r"Physical time $t=kh$ [s]")
# # # axx.yaxis.set_label_coords(-0.125,-0.05)

# # axx.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
# fig.set_size_inches(10.2, 6.2)

# plt.savefig('RelativeErrorMeanMCVersusProxSynthetic50Gen.png', dpi=400)

# =============================================================================
# Univariate omega marginal plot: Case 0 
# =============================================================================

# case0omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1omega1Dt0.2.txt")
# case0omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1omega1Dt0.4.txt")
# case0omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1omega1Dt0.6.txt")
# case0omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1omega1Dt0.8.txt")
# case0omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1omega1Dt1.txt")

# case0omega2t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2omega1Dt0.2.txt")
# case0omega2t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2omega1Dt0.4.txt")
# case0omega2t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2omega1Dt0.6.txt")
# case0omega2t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2omega1Dt0.8.txt")
# case0omega2t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2omega1Dt1.txt")

# case0omega3t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3omega1Dt0.2.txt")
# case0omega3t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3omega1Dt0.4.txt")
# case0omega3t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3omega1Dt0.6.txt")
# case0omega3t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3omega1Dt0.8.txt")
# case0omega3t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3omega1Dt1.txt")

# case0omega4t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4omega1Dt0.2.txt")
# case0omega4t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4omega1Dt0.4.txt")
# case0omega4t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4omega1Dt0.6.txt")
# case0omega4t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4omega1Dt0.8.txt")
# case0omega4t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4omega1Dt1.txt")

# case0omega5t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5omega1Dt0.2.txt")
# case0omega5t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5omega1Dt0.4.txt")
# case0omega5t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5omega1Dt0.6.txt")
# case0omega5t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5omega1Dt0.8.txt")
# case0omega5t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5omega1Dt1.txt")

# case0marg1D_omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1marg1Dt0.2.txt")
# case0marg1D_omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1marg1Dt0.4.txt")
# case0marg1D_omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1marg1Dt0.6.txt")
# case0marg1D_omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1marg1Dt0.8.txt")
# case0marg1D_omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx1marg1Dt1.txt")

# case0marg1D_omega2t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2marg1Dt0.2.txt")
# case0marg1D_omega2t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2marg1Dt0.4.txt")
# case0marg1D_omega2t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2marg1Dt0.6.txt")
# case0marg1D_omega2t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2marg1Dt0.8.txt")
# case0marg1D_omega2t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx2marg1Dt1.txt")

# case0marg1D_omega3t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3marg1Dt0.2.txt")
# case0marg1D_omega3t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3marg1Dt0.4.txt")
# case0marg1D_omega3t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3marg1Dt0.6.txt")
# case0marg1D_omega3t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3marg1Dt0.8.txt")
# case0marg1D_omega3t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx3marg1Dt1.txt")

# case0marg1D_omega4t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4marg1Dt0.2.txt")
# case0marg1D_omega4t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4marg1Dt0.4.txt")
# case0marg1D_omega4t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4marg1Dt0.6.txt")
# case0marg1D_omega4t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4marg1Dt0.8.txt")
# case0marg1D_omega4t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx4marg1Dt1.txt")


# case0marg1D_omega5t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5marg1Dt0.2.txt")
# case0marg1D_omega5t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5marg1Dt0.4.txt")
# case0marg1D_omega5t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5marg1Dt0.6.txt")
# case0marg1D_omega5t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5marg1Dt0.8.txt")
# case0marg1D_omega5t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx5marg1Dt1.txt")


# case0rho1D_omega1t1 = interp1d(case0omega1t1, case0marg1D_omega1t1, kind='cubic')
# case0new_omega1t1 = np.linspace(case0omega1t1.min(), case0omega1t1.max(), num=41, endpoint=True)
# case0rho1D_omega1t2 = interp1d(case0omega1t2, case0marg1D_omega1t2, kind='cubic')
# case0new_omega1t2 = np.linspace(case0omega1t2.min(), case0omega1t2.max(), num=41, endpoint=True)
# case0rho1D_omega1t3 = interp1d(case0omega1t3, case0marg1D_omega1t3, kind='cubic')
# case0new_omega1t3 = np.linspace(case0omega1t3.min(), case0omega1t3.max(), num=41, endpoint=True)
# case0rho1D_omega1t4 = interp1d(case0omega1t4, case0marg1D_omega1t4, kind='cubic')
# case0new_omega1t4 = np.linspace(case0omega1t4.min(), case0omega1t4.max(), num=41, endpoint=True)
# case0rho1D_omega1t5 = interp1d(case0omega1t5, case0marg1D_omega1t5, kind='cubic')
# case0new_omega1t5 = np.linspace(case0omega1t5.min(), case0omega1t5.max(), num=41, endpoint=True)

# case0rho1D_omega2t1 = interp1d(case0omega2t1, case0marg1D_omega2t1, kind='cubic')
# case0new_omega2t1 = np.linspace(case0omega2t1.min(), case0omega2t1.max(), num=41, endpoint=True)
# case0rho1D_omega2t2 = interp1d(case0omega2t2, case0marg1D_omega2t2, kind='cubic')
# case0new_omega2t2 = np.linspace(case0omega2t2.min(), case0omega2t2.max(), num=41, endpoint=True)
# case0rho1D_omega2t3 = interp1d(case0omega2t3, case0marg1D_omega2t3, kind='cubic')
# case0new_omega2t3 = np.linspace(case0omega2t3.min(), case0omega2t3.max(), num=41, endpoint=True)
# case0rho1D_omega2t4 = interp1d(case0omega2t4, case0marg1D_omega2t4, kind='cubic')
# case0new_omega2t4 = np.linspace(case0omega2t4.min(), case0omega2t4.max(), num=41, endpoint=True)
# case0rho1D_omega2t5 = interp1d(case0omega2t5, case0marg1D_omega2t5, kind='cubic')
# case0new_omega2t5 = np.linspace(case0omega2t5.min(), case0omega2t5.max(), num=41, endpoint=True)

# case0rho1D_omega3t1 = interp1d(case0omega3t1, case0marg1D_omega3t1, kind='cubic')
# case0new_omega3t1 = np.linspace(case0omega3t1.min(), case0omega3t1.max(), num=41, endpoint=True)
# case0rho1D_omega3t2 = interp1d(case0omega3t2, case0marg1D_omega3t2, kind='cubic')
# case0new_omega3t2 = np.linspace(case0omega3t2.min(), case0omega3t2.max(), num=41, endpoint=True)
# case0rho1D_omega3t3 = interp1d(case0omega3t3, case0marg1D_omega3t3, kind='cubic')
# case0new_omega3t3 = np.linspace(case0omega3t3.min(), case0omega3t3.max(), num=41, endpoint=True)
# case0rho1D_omega3t4 = interp1d(case0omega3t4, case0marg1D_omega3t4, kind='cubic')
# case0new_omega3t4 = np.linspace(case0omega3t4.min(), case0omega3t4.max(), num=41, endpoint=True)
# case0rho1D_omega3t5 = interp1d(case0omega3t5, case0marg1D_omega3t5, kind='cubic')
# case0new_omega3t5 = np.linspace(case0omega3t5.min(), case0omega3t5.max(), num=41, endpoint=True)

# case0rho1D_omega4t1 = interp1d(case0omega4t1, case0marg1D_omega4t1, kind='cubic')
# case0new_omega4t1 = np.linspace(case0omega4t1.min(), case0omega4t1.max(), num=41, endpoint=True)
# case0rho1D_omega4t2 = interp1d(case0omega4t2, case0marg1D_omega4t2, kind='cubic')
# case0new_omega4t2 = np.linspace(case0omega4t2.min(), case0omega4t2.max(), num=41, endpoint=True)
# case0rho1D_omega4t3 = interp1d(case0omega4t3, case0marg1D_omega4t3, kind='cubic')
# case0new_omega4t3 = np.linspace(case0omega4t3.min(), case0omega4t3.max(), num=41, endpoint=True)
# case0rho1D_omega4t4 = interp1d(case0omega4t4, case0marg1D_omega4t4, kind='cubic')
# case0new_omega4t4 = np.linspace(case0omega4t4.min(), case0omega4t4.max(), num=41, endpoint=True)
# case0rho1D_omega4t5 = interp1d(case0omega4t5, case0marg1D_omega4t5, kind='cubic')
# case0new_omega4t5 = np.linspace(case0omega4t5.min(), case0omega4t5.max(), num=41, endpoint=True)

# case0rho1D_omega5t1 = interp1d(case0omega5t1, case0marg1D_omega5t1, kind='cubic')
# case0new_omega5t1 = np.linspace(case0omega5t1.min(), case0omega5t1.max(), num=41, endpoint=True)
# case0rho1D_omega5t2 = interp1d(case0omega5t2, case0marg1D_omega5t2, kind='cubic')
# case0new_omega5t2 = np.linspace(case0omega5t2.min(), case0omega5t2.max(), num=41, endpoint=True)
# case0rho1D_omega5t3 = interp1d(case0omega5t3, case0marg1D_omega5t3, kind='cubic')
# case0new_omega5t3 = np.linspace(case0omega5t3.min(), case0omega5t3.max(), num=41, endpoint=True)
# case0rho1D_omega5t4 = interp1d(case0omega5t4, case0marg1D_omega5t4, kind='cubic')
# case0new_omega5t4 = np.linspace(case0omega5t4.min(), case0omega5t4.max(), num=41, endpoint=True)
# case0rho1D_omega5t5 = interp1d(case0omega5t5, case0marg1D_omega5t5, kind='cubic')
# case0new_omega5t5 = np.linspace(case0omega5t5.min(), case0omega5t5.max(), num=41, endpoint=True)

# plt.figure()
# ax = plt.subplot(projection='3d')

# colors = ['crimson', 'y', 'g', 'b', 'k']

# # GENERATOR Idx 1
# ax.plot(case0new_omega1t1, 0.2*np.ones(case0new_omega1t1.size), case0rho1D_omega1t1(case0new_omega1t1), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega1t1, np.zeros(case0new_omega1t1.size), case0rho1D_omega1t1(case0new_omega1t1), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega1t2, 0.4*np.ones(case0new_omega1t2.size), case0rho1D_omega1t2(case0new_omega1t2), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega1t2, np.zeros(case0new_omega1t2.size), case0rho1D_omega1t2(case0new_omega1t2), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega1t3, 0.6*np.ones(case0new_omega1t3.size), case0rho1D_omega1t3(case0new_omega1t3), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega1t3, np.zeros(case0new_omega1t3.size), case0rho1D_omega1t3(case0new_omega1t3), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega1t4, 0.8*np.ones(case0new_omega1t4.size), case0rho1D_omega1t4(case0new_omega1t4), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega1t4, np.zeros(case0new_omega1t4.size), case0rho1D_omega1t4(case0new_omega1t4), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega1t5, 1.0*np.ones(case0new_omega1t5.size), case0rho1D_omega1t5(case0new_omega1t5), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega1t5, np.zeros(case0new_omega1t5.size), case0rho1D_omega1t5(case0new_omega1t5), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 2
# ax.plot(case0new_omega2t1, 0.2*np.ones(case0new_omega2t1.size), case0rho1D_omega2t1(case0new_omega2t1), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega2t1, np.zeros(case0new_omega2t1.size), case0rho1D_omega2t1(case0new_omega2t1), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega2t2, 0.4*np.ones(case0new_omega2t2.size), case0rho1D_omega2t2(case0new_omega2t2), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega2t2, np.zeros(case0new_omega2t2.size), case0rho1D_omega2t2(case0new_omega2t2), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega2t3, 0.6*np.ones(case0new_omega2t3.size), case0rho1D_omega2t3(case0new_omega2t3), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega2t3, np.zeros(case0new_omega2t3.size), case0rho1D_omega2t3(case0new_omega2t3), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega2t4, 0.8*np.ones(case0new_omega2t4.size), case0rho1D_omega2t4(case0new_omega2t4), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega2t4, np.zeros(case0new_omega2t4.size), case0rho1D_omega2t4(case0new_omega2t4), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega2t5, 1.0*np.ones(case0new_omega2t5.size), case0rho1D_omega2t5(case0new_omega2t5), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega2t5, np.zeros(case0new_omega2t5.size), case0rho1D_omega2t5(case0new_omega2t5), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 3
# ax.plot(case0new_omega3t1, 0.2*np.ones(case0new_omega3t1.size), case0rho1D_omega3t1(case0new_omega3t1), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega3t1, np.zeros(case0new_omega3t1.size), case0rho1D_omega3t1(case0new_omega3t1), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega3t2, 0.4*np.ones(case0new_omega3t2.size), case0rho1D_omega3t2(case0new_omega3t2), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega3t2, np.zeros(case0new_omega3t2.size), case0rho1D_omega3t2(case0new_omega3t2), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega3t3, 0.6*np.ones(case0new_omega3t3.size), case0rho1D_omega3t3(case0new_omega3t3), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega3t3, np.zeros(case0new_omega3t3.size), case0rho1D_omega3t3(case0new_omega3t3), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega3t4, 0.8*np.ones(case0new_omega3t4.size), case0rho1D_omega3t4(case0new_omega3t4), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega3t4, np.zeros(case0new_omega3t4.size), case0rho1D_omega3t4(case0new_omega3t4), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega3t5, 1.0*np.ones(case0new_omega3t5.size), case0rho1D_omega3t5(case0new_omega3t5), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega3t5, np.zeros(case0new_omega3t5.size), case0rho1D_omega3t5(case0new_omega3t5), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 4
# ax.plot(case0new_omega4t1, 0.2*np.ones(case0new_omega4t1.size), case0rho1D_omega4t1(case0new_omega4t1), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t1, np.zeros(case0new_omega4t1.size), case0rho1D_omega4t1(case0new_omega4t1), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega4t2, 0.4*np.ones(case0new_omega4t2.size), case0rho1D_omega4t2(case0new_omega4t2), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t2, np.zeros(case0new_omega4t2.size), case0rho1D_omega4t2(case0new_omega4t2), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega4t3, 0.6*np.ones(case0new_omega4t3.size), case0rho1D_omega4t3(case0new_omega4t3), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t3, np.zeros(case0new_omega4t3.size), case0rho1D_omega4t3(case0new_omega4t3), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega4t4, 0.8*np.ones(case0new_omega4t4.size), case0rho1D_omega4t4(case0new_omega4t4), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t4, np.zeros(case0new_omega4t4.size), case0rho1D_omega4t4(case0new_omega4t4), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega4t5, 1.0*np.ones(case0new_omega4t5.size), case0rho1D_omega4t5(case0new_omega4t5), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t5, np.zeros(case0new_omega4t5.size), case0rho1D_omega4t5(case0new_omega4t5), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 5
# ax.plot(case0new_omega5t1, 0.2*np.ones(case0new_omega5t1.size), case0rho1D_omega5t1(case0new_omega5t1), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega5t1, np.zeros(case0new_omega5t1.size), case0rho1D_omega5t1(case0new_omega5t1), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega5t2, 0.4*np.ones(case0new_omega5t2.size), case0rho1D_omega5t2(case0new_omega5t2), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega5t2, np.zeros(case0new_omega5t2.size), case0rho1D_omega5t2(case0new_omega5t2), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega5t3, 0.6*np.ones(case0new_omega5t3.size), case0rho1D_omega5t3(case0new_omega5t3), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega5t3, np.zeros(case0new_omega5t3.size), case0rho1D_omega5t3(case0new_omega5t3), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega5t4, 0.8*np.ones(case0new_omega5t4.size), case0rho1D_omega5t4(case0new_omega5t4), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega5t4, np.zeros(case0new_omega5t4.size), case0rho1D_omega5t4(case0new_omega5t4), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega5t5, 1.0*np.ones(case0new_omega5t5.size), case0rho1D_omega5t5(case0new_omega5t5), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega5t5, np.zeros(case0new_omega5t5.size), case0rho1D_omega5t5(case0new_omega5t5), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# plt.title('$\omega$ marginals for the IEEE 14 bus simulation, Case I', color='gray')
# plt.xlabel("$\omega$ [rad/s]",labelpad=10)
# plt.ylabel("time [s]",labelpad=12)
# # # ax.set_zticks([])
# # # ax.set_xticks([-10, -5, 0, 5, 10])
# ax.grid(False)
# ax.xaxis.pane.set_edgecolor('black')
# ax.yaxis.pane.set_edgecolor('black')
# # ax.zaxis.pane.set_edgecolor('blank')
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')
# ax.w_zaxis.line.set_lw(0.)

# names=["Generator 1",
#  "Generator 2",
#  "Generator 3", 
#  "Generator 4",
#  "Generator 5"
#  ]

# patches=[]
# for i in range(5):
#     patches.append(mpatches.Patch(color=colors[i], label=names[i], alpha=myalphavalue))

# lgnd=ax.legend(handles=patches[0:7], bbox_to_anchor=(1.23,0.95), loc='upper right', frameon=False, ncol=5,columnspacing=1.0,labelspacing=0.2, handletextpad=0.2, handlelength=1,fancybox=False, shadow=False)
# ax.view_init(elev=33.5, azim=140.)

# fig.set_size_inches(10.2, 5)

# plt.savefig('Case0_OmegaMarginals.png', dpi=300)

# # =============================================================================
# # Univariate omega marginal plot: Case 1 
# # =============================================================================

# t_vec = [0.2, 0.4, 0.6, 0.8, 1.0]

# case1omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.2.txt")
# case1omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.4.txt")
# case1omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.6.txt")
# case1omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.8.txt")
# case1omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt1.txt")

# case1omega2t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2omega1Dt0.2.txt")
# case1omega2t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2omega1Dt0.4.txt")
# case1omega2t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2omega1Dt0.6.txt")
# case1omega2t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2omega1Dt0.8.txt")
# case1omega2t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2omega1Dt1.txt")

# case1omega3t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3omega1Dt0.2.txt")
# case1omega3t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3omega1Dt0.4.txt")
# case1omega3t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3omega1Dt0.6.txt")
# case1omega3t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3omega1Dt0.8.txt")
# case1omega3t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3omega1Dt1.txt")

# case1omega4t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4omega1Dt0.2.txt")
# case1omega4t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4omega1Dt0.4.txt")
# case1omega4t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4omega1Dt0.6.txt")
# case1omega4t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4omega1Dt0.8.txt")
# case1omega4t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4omega1Dt1.txt")

# case1omega5t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5omega1Dt0.2.txt")
# case1omega5t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5omega1Dt0.4.txt")
# case1omega5t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5omega1Dt0.6.txt")
# case1omega5t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5omega1Dt0.8.txt")
# case1omega5t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5omega1Dt1.txt")

# case1marg1D_omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.2.txt")
# case1marg1D_omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.4.txt")
# case1marg1D_omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.6.txt")
# case1marg1D_omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.8.txt")
# case1marg1D_omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt1.txt")

# case1marg1D_omega2t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2marg1Dt0.2.txt")
# case1marg1D_omega2t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2marg1Dt0.4.txt")
# case1marg1D_omega2t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2marg1Dt0.6.txt")
# case1marg1D_omega2t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2marg1Dt0.8.txt")
# case1marg1D_omega2t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx2marg1Dt1.txt")

# case1marg1D_omega3t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3marg1Dt0.2.txt")
# case1marg1D_omega3t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3marg1Dt0.4.txt")
# case1marg1D_omega3t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3marg1Dt0.6.txt")
# case1marg1D_omega3t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3marg1Dt0.8.txt")
# case1marg1D_omega3t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx3marg1Dt1.txt")

# case1marg1D_omega4t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4marg1Dt0.2.txt")
# case1marg1D_omega4t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4marg1Dt0.4.txt")
# case1marg1D_omega4t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4marg1Dt0.6.txt")
# case1marg1D_omega4t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4marg1Dt0.8.txt")
# case1marg1D_omega4t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx4marg1Dt1.txt")


# case1marg1D_omega5t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5marg1Dt0.2.txt")
# case1marg1D_omega5t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5marg1Dt0.4.txt")
# case1marg1D_omega5t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5marg1Dt0.6.txt")
# case1marg1D_omega5t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5marg1Dt0.8.txt")
# case1marg1D_omega5t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx5marg1Dt1.txt")


# case1rho1D_omega1t1 = interp1d(case1omega1t1, case1marg1D_omega1t1, kind='cubic')
# case1new_omega1t1 = np.linspace(case1omega1t1.min(), case1omega1t1.max(), num=41, endpoint=True)
# case1rho1D_omega1t2 = interp1d(case1omega1t2, case1marg1D_omega1t2, kind='cubic')
# case1new_omega1t2 = np.linspace(case1omega1t2.min(), case1omega1t2.max(), num=41, endpoint=True)
# case1rho1D_omega1t3 = interp1d(case1omega1t3, case1marg1D_omega1t3, kind='cubic')
# case1new_omega1t3 = np.linspace(case1omega1t3.min(), case1omega1t3.max(), num=41, endpoint=True)
# case1rho1D_omega1t4 = interp1d(case1omega1t4, case1marg1D_omega1t4, kind='cubic')
# case1new_omega1t4 = np.linspace(case1omega1t4.min(), case1omega1t4.max(), num=41, endpoint=True)
# case1rho1D_omega1t5 = interp1d(case1omega1t5, case1marg1D_omega1t5, kind='cubic')
# case1new_omega1t5 = np.linspace(case1omega1t5.min(), case1omega1t5.max(), num=41, endpoint=True)

# case1rho1D_omega2t1 = interp1d(case1omega2t1, case1marg1D_omega2t1, kind='cubic')
# case1new_omega2t1 = np.linspace(case1omega2t1.min(), case1omega2t1.max(), num=41, endpoint=True)
# case1rho1D_omega2t2 = interp1d(case1omega2t2, case1marg1D_omega2t2, kind='cubic')
# case1new_omega2t2 = np.linspace(case1omega2t2.min(), case1omega2t2.max(), num=41, endpoint=True)
# case1rho1D_omega2t3 = interp1d(case1omega2t3, case1marg1D_omega2t3, kind='cubic')
# case1new_omega2t3 = np.linspace(case1omega2t3.min(), case1omega2t3.max(), num=41, endpoint=True)
# case1rho1D_omega2t4 = interp1d(case1omega2t4, case1marg1D_omega2t4, kind='cubic')
# case1new_omega2t4 = np.linspace(case1omega2t4.min(), case1omega2t4.max(), num=41, endpoint=True)
# case1rho1D_omega2t5 = interp1d(case1omega2t5, case1marg1D_omega2t5, kind='cubic')
# case1new_omega2t5 = np.linspace(case1omega2t5.min(), case1omega2t5.max(), num=41, endpoint=True)

# case1rho1D_omega3t1 = interp1d(case1omega3t1, case1marg1D_omega3t1, kind='cubic')
# case1new_omega3t1 = np.linspace(case1omega3t1.min(), case1omega3t1.max(), num=41, endpoint=True)
# case1rho1D_omega3t2 = interp1d(case1omega3t2, case1marg1D_omega3t2, kind='cubic')
# case1new_omega3t2 = np.linspace(case1omega3t2.min(), case1omega3t2.max(), num=41, endpoint=True)
# case1rho1D_omega3t3 = interp1d(case1omega3t3, case1marg1D_omega3t3, kind='cubic')
# case1new_omega3t3 = np.linspace(case1omega3t3.min(), case1omega3t3.max(), num=41, endpoint=True)
# case1rho1D_omega3t4 = interp1d(case1omega3t4, case1marg1D_omega3t4, kind='cubic')
# case1new_omega3t4 = np.linspace(case1omega3t4.min(), case1omega3t4.max(), num=41, endpoint=True)
# case1rho1D_omega3t5 = interp1d(case1omega3t5, case1marg1D_omega3t5, kind='cubic')
# case1new_omega3t5 = np.linspace(case1omega3t5.min(), case1omega3t5.max(), num=41, endpoint=True)

# case1rho1D_omega4t1 = interp1d(case1omega4t1, case1marg1D_omega4t1, kind='cubic')
# case1new_omega4t1 = np.linspace(case1omega4t1.min(), case1omega4t1.max(), num=41, endpoint=True)
# case1rho1D_omega4t2 = interp1d(case1omega4t2, case1marg1D_omega4t2, kind='cubic')
# case1new_omega4t2 = np.linspace(case1omega4t2.min(), case1omega4t2.max(), num=41, endpoint=True)
# case1rho1D_omega4t3 = interp1d(case1omega4t3, case1marg1D_omega4t3, kind='cubic')
# case1new_omega4t3 = np.linspace(case1omega4t3.min(), case1omega4t3.max(), num=41, endpoint=True)
# case1rho1D_omega4t4 = interp1d(case1omega4t4, case1marg1D_omega4t4, kind='cubic')
# case1new_omega4t4 = np.linspace(case1omega4t4.min(), case1omega4t4.max(), num=41, endpoint=True)
# case1rho1D_omega4t5 = interp1d(case1omega4t5, case1marg1D_omega4t5, kind='cubic')
# case1new_omega4t5 = np.linspace(case1omega4t5.min(), case1omega4t5.max(), num=41, endpoint=True)

# case1rho1D_omega5t1 = interp1d(case1omega5t1, case1marg1D_omega5t1, kind='cubic')
# case1new_omega5t1 = np.linspace(case1omega5t1.min(), case1omega5t1.max(), num=41, endpoint=True)
# case1rho1D_omega5t2 = interp1d(case1omega5t2, case1marg1D_omega5t2, kind='cubic')
# case1new_omega5t2 = np.linspace(case1omega5t2.min(), case1omega5t2.max(), num=41, endpoint=True)
# case1rho1D_omega5t3 = interp1d(case1omega5t3, case1marg1D_omega5t3, kind='cubic')
# case1new_omega5t3 = np.linspace(case1omega5t3.min(), case1omega5t3.max(), num=41, endpoint=True)
# case1rho1D_omega5t4 = interp1d(case1omega5t4, case1marg1D_omega5t4, kind='cubic')
# case1new_omega5t4 = np.linspace(case1omega5t4.min(), case1omega5t4.max(), num=41, endpoint=True)
# case1rho1D_omega5t5 = interp1d(case1omega5t5, case1marg1D_omega5t5, kind='cubic')
# case1new_omega5t5 = np.linspace(case1omega5t5.min(), case1omega5t5.max(), num=41, endpoint=True)

# plt.figure()
# ax = plt.subplot(projection='3d')

# colors = ['crimson', 'y', 'g', 'b', 'k']

# # GENERATOR Idx 1
# ax.plot(case1new_omega1t1, 0.2*np.ones(case1new_omega1t1.size), case1rho1D_omega1t1(case1new_omega1t1), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega1t1, np.zeros(case1new_omega1t1.size), case1rho1D_omega1t1(case1new_omega1t1), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega1t2, 0.4*np.ones(case1new_omega1t2.size), case1rho1D_omega1t2(case1new_omega1t2), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega1t2, np.zeros(case1new_omega1t2.size), case1rho1D_omega1t2(case1new_omega1t2), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega1t3, 0.6*np.ones(case1new_omega1t3.size), case1rho1D_omega1t3(case1new_omega1t3), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega1t3, np.zeros(case1new_omega1t3.size), case1rho1D_omega1t3(case1new_omega1t3), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega1t4, 0.8*np.ones(case1new_omega1t4.size), case1rho1D_omega1t4(case1new_omega1t4), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega1t4, np.zeros(case1new_omega1t4.size), case1rho1D_omega1t4(case1new_omega1t4), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega1t5, 1.0*np.ones(case1new_omega1t5.size), case1rho1D_omega1t5(case1new_omega1t5), color='crimson',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega1t5, np.zeros(case1new_omega1t5.size), case1rho1D_omega1t5(case1new_omega1t5), color='crimson', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 2
# ax.plot(case1new_omega2t1, 0.2*np.ones(case1new_omega2t1.size), case1rho1D_omega2t1(case1new_omega2t1), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega2t1, np.zeros(case1new_omega2t1.size), case1rho1D_omega2t1(case1new_omega2t1), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega2t2, 0.4*np.ones(case1new_omega2t2.size), case1rho1D_omega2t2(case1new_omega2t2), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega2t2, np.zeros(case1new_omega2t2.size), case1rho1D_omega2t2(case1new_omega2t2), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega2t3, 0.6*np.ones(case1new_omega2t3.size), case1rho1D_omega2t3(case1new_omega2t3), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega2t3, np.zeros(case1new_omega2t3.size), case1rho1D_omega2t3(case1new_omega2t3), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega2t4, 0.8*np.ones(case1new_omega2t4.size), case1rho1D_omega2t4(case1new_omega2t4), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega2t4, np.zeros(case1new_omega2t4.size), case1rho1D_omega2t4(case1new_omega2t4), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega2t5, 1.0*np.ones(case1new_omega2t5.size), case1rho1D_omega2t5(case1new_omega2t5), color='y',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega2t5, np.zeros(case1new_omega2t5.size), case1rho1D_omega2t5(case1new_omega2t5), color='y', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 3
# ax.plot(case1new_omega3t1, 0.2*np.ones(case1new_omega3t1.size), case1rho1D_omega3t1(case1new_omega3t1), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega3t1, np.zeros(case1new_omega3t1.size), case1rho1D_omega3t1(case1new_omega3t1), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega3t2, 0.4*np.ones(case1new_omega3t2.size), case1rho1D_omega3t2(case1new_omega3t2), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega3t2, np.zeros(case1new_omega3t2.size), case1rho1D_omega3t2(case1new_omega3t2), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega3t3, 0.6*np.ones(case1new_omega3t3.size), case1rho1D_omega3t3(case1new_omega3t3), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega3t3, np.zeros(case1new_omega3t3.size), case1rho1D_omega3t3(case1new_omega3t3), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega3t4, 0.8*np.ones(case1new_omega3t4.size), case1rho1D_omega3t4(case1new_omega3t4), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega3t4, np.zeros(case1new_omega3t4.size), case1rho1D_omega3t4(case1new_omega3t4), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega3t5, 1.0*np.ones(case1new_omega3t5.size), case1rho1D_omega3t5(case1new_omega3t5), color='g',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega3t5, np.zeros(case1new_omega3t5.size), case1rho1D_omega3t5(case1new_omega3t5), color='g', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 4
# ax.plot(case1new_omega4t1, 0.2*np.ones(case1new_omega4t1.size), case1rho1D_omega4t1(case1new_omega4t1), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t1, np.zeros(case1new_omega4t1.size), case1rho1D_omega4t1(case1new_omega4t1), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega4t2, 0.4*np.ones(case1new_omega4t2.size), case1rho1D_omega4t2(case1new_omega4t2), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t2, np.zeros(case1new_omega4t2.size), case1rho1D_omega4t2(case1new_omega4t2), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega4t3, 0.6*np.ones(case1new_omega4t3.size), case1rho1D_omega4t3(case1new_omega4t3), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t3, np.zeros(case1new_omega4t3.size), case1rho1D_omega4t3(case1new_omega4t3), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega4t4, 0.8*np.ones(case1new_omega4t4.size), case1rho1D_omega4t4(case1new_omega4t4), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t4, np.zeros(case1new_omega4t4.size), case1rho1D_omega4t4(case1new_omega4t4), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega4t5, 1.0*np.ones(case1new_omega4t5.size), case1rho1D_omega4t5(case1new_omega4t5), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t5, np.zeros(case1new_omega4t5.size), case1rho1D_omega4t5(case1new_omega4t5), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # GENERATOR Idx 5
# ax.plot(case1new_omega5t1, 0.2*np.ones(case1new_omega5t1.size), case1rho1D_omega5t1(case1new_omega5t1), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega5t1, np.zeros(case1new_omega5t1.size), case1rho1D_omega5t1(case1new_omega5t1), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega5t2, 0.4*np.ones(case1new_omega5t2.size), case1rho1D_omega5t2(case1new_omega5t2), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega5t2, np.zeros(case1new_omega5t2.size), case1rho1D_omega5t2(case1new_omega5t2), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega5t3, 0.6*np.ones(case1new_omega5t3.size), case1rho1D_omega5t3(case1new_omega5t3), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega5t3, np.zeros(case1new_omega5t3.size), case1rho1D_omega5t3(case1new_omega5t3), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega5t4, 0.8*np.ones(case1new_omega5t4.size), case1rho1D_omega5t4(case1new_omega5t4), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega5t4, np.zeros(case1new_omega5t4.size), case1rho1D_omega5t4(case1new_omega5t4), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega5t5, 1.0*np.ones(case1new_omega5t5.size), case1rho1D_omega5t5(case1new_omega5t5), color='k',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega5t5, np.zeros(case1new_omega5t5.size), case1rho1D_omega5t5(case1new_omega5t5), color='k', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# plt.title('$\omega$ marginals for the IEEE 14 bus system, Case II', color='gray')
# plt.xlabel("$\omega$ [rad/s]",labelpad=10)
# plt.ylabel("time [s]",labelpad=12)
# # # ax.set_zticks([])
# # # ax.set_xticks([-10, -5, 0, 5, 10])
# ax.grid(False)
# ax.xaxis.pane.set_edgecolor('black')
# ax.yaxis.pane.set_edgecolor('black')
# # ax.zaxis.pane.set_edgecolor('blank')
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')
# ax.w_zaxis.line.set_lw(0.)

# names=["Generator 1",
#  "Generator 2",
#  "Generator 3", 
#  "Generator 4",
#  "Generator 5"
#  ]

# patches=[]
# for i in range(5):
#     patches.append(mpatches.Patch(color=colors[i], label=names[i], alpha=myalphavalue))

# lgnd=ax.legend(handles=patches[0:7], bbox_to_anchor=(1.23,0.95), loc='upper right', frameon=False, ncol=5,columnspacing=1.0,labelspacing=0.2, handletextpad=0.2, handlelength=1,fancybox=False, shadow=False)
# ax.view_init(elev=33.5, azim=140.)

# fig.set_size_inches(10.2, 5)

# plt.savefig('Case1_OmegaMarginals.png', dpi=300)


# # ===================================================================================
# # Univariate omega marginal for Generator 4 (IEEE 14 system Bus #6): Case 0 and 1 
# # ===================================================================================
# plt.figure()
# ax = plt.subplot(projection='3d')

# colors = ['b', 'c']

# # Case 0 (nominal): GENERATOR Idx 4
# ax.plot(case0new_omega4t1, 0.2*np.ones(case0new_omega4t1.size), case0rho1D_omega4t1(case0new_omega4t1), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t1, np.zeros(case0new_omega4t1.size), case0rho1D_omega4t1(case0new_omega4t1), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case0new_omega4t2, 0.4*np.ones(case0new_omega4t2.size), case0rho1D_omega4t2(case0new_omega4t2), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t2, np.zeros(case0new_omega4t2.size), case0rho1D_omega4t2(case0new_omega4t2), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case0new_omega4t3, 0.6*np.ones(case0new_omega4t3.size), case0rho1D_omega4t3(case0new_omega4t3), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t3, np.zeros(case0new_omega4t3.size), case0rho1D_omega4t3(case0new_omega4t3), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case0new_omega4t4, 0.8*np.ones(case0new_omega4t4.size), case0rho1D_omega4t4(case0new_omega4t4), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t4, np.zeros(case0new_omega4t4.size), case0rho1D_omega4t4(case0new_omega4t4), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case0new_omega4t5, 1.0*np.ones(case0new_omega4t5.size), case0rho1D_omega4t5(case0new_omega4t5), color='b',alpha=myalphavalue)
# fill = plt.fill_between(case0new_omega4t5, np.zeros(case0new_omega4t5.size), case0rho1D_omega4t5(case0new_omega4t5), color='b', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# # Case 1 (line 13 fails): GENERATOR Idx 4
# ax.plot(case1new_omega4t1, 0.2*np.ones(case1new_omega4t1.size), case1rho1D_omega4t1(case1new_omega4t1), color='c',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t1, np.zeros(case1new_omega4t1.size), case1rho1D_omega4t1(case1new_omega4t1), color='c', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.2, zdir='y')
# ax.plot(case1new_omega4t2, 0.4*np.ones(case1new_omega4t2.size), case1rho1D_omega4t2(case1new_omega4t2), color='c',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t2, np.zeros(case1new_omega4t2.size), case1rho1D_omega4t2(case1new_omega4t2), color='c', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.4, zdir='y')
# ax.plot(case1new_omega4t3, 0.6*np.ones(case1new_omega4t3.size), case1rho1D_omega4t3(case1new_omega4t3), color='c',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t3, np.zeros(case1new_omega4t3.size), case1rho1D_omega4t3(case1new_omega4t3), color='c', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.6, zdir='y')
# ax.plot(case1new_omega4t4, 0.8*np.ones(case1new_omega4t4.size), case1rho1D_omega4t4(case1new_omega4t4), color='c',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t4, np.zeros(case1new_omega4t4.size), case1rho1D_omega4t4(case1new_omega4t4), color='c', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=0.8, zdir='y')
# ax.plot(case1new_omega4t5, 1.0*np.ones(case1new_omega4t5.size), case1rho1D_omega4t5(case1new_omega4t5), color='c',alpha=myalphavalue)
# fill = plt.fill_between(case1new_omega4t5, np.zeros(case1new_omega4t5.size), case1rho1D_omega4t5(case1new_omega4t5), color='c', alpha=0.5*myalphavalue)
# fill_collection = ax.add_collection3d(fill, zs=1.0, zdir='y')

# plt.title('$\omega$ marginals for the bus 6 (generator 4) in IEEE 14 bus system', color='gray')
# plt.xlabel("$\omega$ [rad/s]",labelpad=10)
# plt.ylabel("time [s]",labelpad=12)
# # # ax.set_zticks([])
# # # ax.set_xticks([-10, -5, 0, 5, 10])
# ax.grid(False)
# ax.xaxis.pane.set_edgecolor('black')
# ax.yaxis.pane.set_edgecolor('black')
# # ax.zaxis.pane.set_edgecolor('blank')
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor('w')
# ax.yaxis.pane.set_edgecolor('w')
# ax.zaxis.pane.set_edgecolor('w')
# ax.w_zaxis.line.set_lw(0.)

# names=["Case I",
#  "Case II"
#  ]

# patches=[]
# for i in range(2):
#     patches.append(mpatches.Patch(color=colors[i], label=names[i], alpha=myalphavalue))

# lgnd=ax.legend(handles=patches[0:7], bbox_to_anchor=(0.7,0.95), loc='upper right', frameon=False, ncol=2,columnspacing=3.0,labelspacing=0.2, handletextpad=0.2, handlelength=1,fancybox=False, shadow=False)
# ax.view_init(elev=33.5, azim=140.)

# fig.set_size_inches(10.2, 5)

# plt.savefig('Case0Case1Combined_Gen4_OmegaMarginals.png', dpi=300)

# =============================================================================
# Rotor angle uncertainty bound plots (IEEE 14 case study)
# =============================================================================

# Case 0
# fCount = len(glob.glob('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx[1-5]theta.txt'))
# theta_files = glob.iglob('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case0_norminal_IEEE14BusGenIdx[1-5]theta.txt')

fCount = len(glob.glob('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx[1-5]theta.txt'))
theta_files = glob.iglob('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx[1-5]theta.txt')
# Mean_theta_files = glob.iglob('/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14bus_Mean[1-5]theta.txt')


fig, axs = plt.subplots(fCount,1,sharex=True, sharey=True)
axs = axs.ravel()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)

colors = ['crimson', 'y', 'g', 'b', 'k']

t_vec = np.insert(Time_Synthetic,0,0.0,axis=0)
ticklen = np.pi

names=["Generator 1",
 "Generator 2",
 "Generator 3", 
 "Generator 4",
 "Generator 5"
 ]

patches=[]
for i in range(5):
    patches.append(mpatches.Patch(color=colors[i], label=names[i], alpha=myalphavalue))

lgnd=axs[0].legend(handles=patches[0:7], bbox_to_anchor=(0.5,1.3), loc='center', frameon=False, ncol=5,columnspacing=1.0,labelspacing=0.2, handletextpad=0.2, handlelength=1,fancybox=False, shadow=False)

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """
    format label properly
    for example: 0.6666 pi --> 2π/3
               : 0      pi --> 0
               : 0.50   pi --> π/2  
    """
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val/np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator
    
    fmt2 = "%s" % d 
    if n == 0:
        fmt1 = "0"
    elif n == 1:
        fmt1 = pi
    else:
        fmt1 = r"%s%s" % (n,pi)
        
    fmtstring = "$" + minus + (fmt1 if d == 1 else r"{%s}/{%s}" % (fmt1, fmt2)) + "$"
    
    return fmtstring

numel_t = size(arange(0,1001,30))

for fID in range(fCount):
    theta_now = np.loadtxt(next(theta_files))
    # Mean_theta_now = np.loadtxt(next(Mean_theta_files))
    boxplot_dict = axs[fID].boxplot(theta_now[:,0:1001:30], showfliers=False, patch_artist=True,
            boxprops=dict(facecolor=colors[fID], color=colors[fID]),
            capprops=dict(color=colors[fID]),
            whiskerprops=dict(color=colors[fID]),
            flierprops=dict(color=colors[fID], markeredgecolor=colors[fID]),
            medianprops=dict(linestyle='none'))
    for b in boxplot_dict['boxes']:
        b.set_alpha(myalphavalue)
    axs[fID].plot(1+(numel_t-1)*t_vec,theta_now.mean(axis=0),'-', color=colors[fID], lw=1.2) 
    # axs[fID].plot(1+(numel_t-1)*t_vec,Mean_theta_now,':', color=colors[fID], lw=1.2)   
    axs[fID].set_ylim(-0.2, 6.5)
    axs[fID].tick_params(direction='in',which='both')
    axs[fID].yaxis.set_major_formatter(FuncFormatter(pi_axis_formatter))
    axs[fID].yaxis.set_major_locator(MultipleLocator(base=ticklen))
    axs[fID].tick_params(axis='both', labelsize=18)
    axs[fID].set_ylabel(r"$\theta$" f"$_{fID+1}$ [rad]")
    axs[fID].set_xticks([0, numel_t])
    axs[fID].set_xticklabels([0, 1])
    axs[fID].axis('tight')

axs[fID].set_xlabel(r"Time $t$ [s]")

# # ax.grid(True,which="both",ls="-", color='0.75')
# # # axx2.grid(True,which="both",ls="-", color='0.75')




# # # axx.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
fig.set_size_inches(10.2, 6.2)

#plt.savefig('IEEE14_Case0_RotorAngles.png', dpi=300)
plt.savefig('IEEE14_Case1_RotorAngles.png', dpi=300)











