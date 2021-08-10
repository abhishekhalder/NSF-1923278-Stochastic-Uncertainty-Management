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

#====================================================
# Make plots beautiful
#====================================================
myalphavalue=0.6
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
# Computational time plot 
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
# Univariate omega marginal plot: Case 1 
# =============================================================================
# Time_Synthetic = np.loadtxt("case1_line_13_failure_TimeSyntheticIEEE14bus.txt")

t_vec = [0.2, 0.4, 0.6, 0.8, 1.0]

omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.2.txt")
omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.4.txt")
omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.6.txt")
omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt0.8.txt")
omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1omega1Dt1.txt")

marg1D_omega1t1 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.2.txt")
marg1D_omega1t2 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.4.txt")
marg1D_omega1t3 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.6.txt")
marg1D_omega1t4 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt0.8.txt")
marg1D_omega1t5 = np.loadtxt("/Users/abhishekhaldermac/NSF-1923278-Stochastic-Uncertainty-Management/code/case1_line_13_failure_IEEE14BusGenIdx1marg1Dt1.txt")

plt.figure()
ax = plt.subplot(projection='3d')

# # colors = ['crimson', 'g', 'm', 'k', 'b']

# GENERATOR Idx 1
markerline, stemlines, baseline = ax.stem(omega1t1,0.2*np.ones(marg1D_omega1t1.size),marg1D_omega1t1, linefmt='crimson')
markerline.set_markerfacecolor('crimson')
markerline.set_markeredgecolor('none')
baseline.set_alpha(0.7)
stemlines.set_alpha(0.7)
markerline.set_alpha(0.7)
markerline, stemlines, baseline = ax.stem(omega1t2,0.4*np.ones(marg1D_omega1t2.size),marg1D_omega1t2, linefmt='crimson')
markerline.set_markerfacecolor('crimson')
markerline.set_markeredgecolor('none')
baseline.set_alpha(0.7)
stemlines.set_alpha(0.7)
markerline.set_alpha(0.7)
markerline, stemlines, baseline = ax.stem(omega1t3,0.6*np.ones(marg1D_omega1t3.size),marg1D_omega1t3, linefmt='crimson')
markerline.set_markerfacecolor('crimson')
markerline.set_markeredgecolor('none')
baseline.set_alpha(0.7)
stemlines.set_alpha(0.7)
markerline.set_alpha(0.7)

plt.title('$\omega$ marginals', color='gray')
plt.xlabel("$\omega$ [rad/s]",labelpad=10)
plt.ylabel("time [s]",labelpad=12)
# # ax.set_zticks([])
# # ax.set_xticks([-10, -5, 0, 5, 10])
ax.grid(False)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
# ax.zaxis.pane.set_edgecolor('blank')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.w_zaxis.line.set_lw(0.)

# # lgnd=ax.legend(handles=patches[0:7], bbox_to_anchor=(1.05,0.9), loc='upper right', frameon=False, ncol=7,columnspacing=2.0,labelspacing=0.2, handletextpad=0.2, handlelength=1,fancybox=False, shadow=False)
# ax.view_init(elev=33.5, azim=140.)

fig.set_size_inches(10.2, 5)

plt.savefig('Case1_OmegaMarginals.png', dpi=300)


























