import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid
from itertools import cycle
from pylab import *
import glob
from pylab import rcParams

#====================================================
# Make plots beautiful
#====================================================

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
Time_Synthetic = np.loadtxt("TimeSynthetic.txt")
CompTime_Synthetic = np.loadtxt("ComptimeSythetic.txt")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.semilogy(Time_Synthetic, CompTime_Synthetic,'-', color='b', lw=1.5)

ax.set_ylim(1*10**-3, 1.2*10**-2)

ax.tick_params(direction='in',which='both')
 # axx.xaxis.tick_top()

ax.grid(True,which="both",ls="-", color='0.75')
# axx2.grid(True,which="both",ls="-", color='0.75')
ax.tick_params(axis='both', labelsize=18)


ax.set_ylabel(r"Computational time [s]")
ax.set_xlabel(r"Physical time $t=kh$ [s]")
# # axx.yaxis.set_label_coords(-0.125,-0.05)

# axx.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
fig.set_size_inches(10.2, 6.2)

plt.savefig('ComputationalTimeSynthetic.png', dpi=400)

# =============================================================================
# Relative error in mean vector: MC versus Prox plot
# =============================================================================
RelErrMeanVectorMCvsProx_Synthetic = np.loadtxt("RelErrMeanVectorMCvsProxSythetic.txt")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.semilogy(Time_Synthetic[3:-1], RelErrMeanVectorMCvsProx_Synthetic[4:-1],'-', color='k', lw=1.5)

# ax.set_ylim(1*10**-3, 1.2*10**-2)

ax.tick_params(direction='in',which='both')
 # axx.xaxis.tick_top()

ax.grid(True,which="both",ls="-", color='0.75')
# axx2.grid(True,which="both",ls="-", color='0.75')
ax.tick_params(axis='both', labelsize=18)


ax.set_ylabel(r"Realtive error $\frac{\|\boldsymbol{\mu}_{\rm{MC}}-\boldsymbol{\mu}_{\rm{Prox}}\|_{2}}{\|\boldsymbol{\mu}_{\rm{MC}}\|_{2}}$")
ax.set_xlabel(r"Physical time $t=kh$ [s]")
# # axx.yaxis.set_label_coords(-0.125,-0.05)

# axx.legend(markerscale=1.5, numpoints=1,  ncol=1, bbox_to_anchor=(1.005, 1), frameon=False, prop={'size': 10.5})
fig.set_size_inches(10.2, 6.2)

plt.savefig('RelativeErrorMeanMCVersusProxSynthetic.png', dpi=400)



