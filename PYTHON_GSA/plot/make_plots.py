import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

nvar = 10
lim_l =0
labs = ['cbG','cbL','dsG','dsL','lsG',
        'lsL','qnG','qnL','xcG','xcL']

f = open('input.out', "r")
lines_list = []
for line in f:
    lines_list.append(line.strip())

SU_list = []
SU_index_list = []

HEATMAP_val_list = []
HEATMAP_indices_list = []
Si_list = []
Si_T_list = []

for line in lines_list:
    split_line = line.split()
    if split_line[0] == 'SU':
        if len(split_line) > 3:
            HEATMAP_indices_list.append([split_line[1],split_line[2]])
            HEATMAP_val_list.append(float(split_line[-1]))
        else:
            SU_index_list.append(float(split_line[1]))
            SU_list.append(float(split_line[-1]))
    elif split_line[0].isdigit():
        Si_list.append(float(split_line[1]))
        Si_T_list.append(float(split_line[2]))
  

#Plot Sobol HeatMap
       
sobol_ij = np.zeros([nvar, nvar])
for i,indx in enumerate(HEATMAP_indices_list):
    sobol_ij[int(indx[0].strip(',')),int(indx[1])] = HEATMAP_val_list[i]
    sobol_ij[int(indx[1]),int(indx[0].strip(','))] = HEATMAP_val_list[i]

for i in range(nvar):
    sobol_ij[i, i] = 0


config = {
    "font.family": 'serif',
    "font.size": 12,
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'stix',
    'axes.unicode_minus': False
}
rcParams.update(config)
# viz second order of SA indices
fig = plt.figure(figsize=(3.5, 2.5),dpi=300)#3.35
ax = fig.gca()
im = ax.imshow(sobol_ij, cmap='GnBu',vmin = lim_l)
x = np.arange(nvar)
for i in range(nvar):
    ax.plot([-0.5, nvar - 0.5], [i + 0.5, i + 0.5], 'k:', lw=0.5)
    ax.plot([i + 0.5, i + 0.5], [-0.5, nvar - 0.5], 'k:', lw=0.5)
ax.plot([-0.5, nvar - 0.5], [-0.5, nvar - 0.5], 'k-', lw=0.8)
ax.set_xlim([-0.5, nvar - 0.5])
ax.set_ylim([nvar - 0.5, -0.5])
ax.tick_params(which='both', direction='in')
ax.set_xticks(x)
ax.set_xticklabels(labs,fontsize=10,fontproperties='Times New Roman',rotation=45)#
ax.set_yticks(x)
ax.set_yticklabels(labs,fontsize=10,fontproperties='Times New Roman')
cbar=fig.colorbar(im,fraction=0.04,pad=0.05)
cbar.ax.tick_params(labelsize=10)
plt.subplots_adjust(bottom=0.13, top=0.95, left=0.05, right=0.82)
plt.show()

#Plot Sobol Indices
x = np.arange(nvar)
width = 0.4

sobol_i = Si_list
soboltol_i = Si_T_list

fig = plt.figure(figsize=(3.5, 2), dpi=300)#3.35
ax = fig.gca()
ax.bar(x - width / 2., sobol_i, width, label='S$_\mathrm{1}$',color='lightskyblue',edgecolor = "black")#color='bisque',r'S$_1$'
ax.bar(x + width / 2., soboltol_i, width, label='S$_\mathrm{T}$',color='cornflowerblue',edgecolor = "black")#color='gold',r'S$_T$'
for i in range(nvar + 1):
    ax.plot([i - 0.5, i - 0.5], [0, 1.1 * max(soboltol_i)], 'k--', lw=0.4)
    ax.plot([i - 0.5, i - 0.5], [0, 1.1 * max(soboltol_i)], 'k--', lw=0.4)
ax.set_ylim([0, 1.1 * max(soboltol_i)])
ax.tick_params(which='both', direction='in',labelsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labs,fontsize=12)
plt.yticks(fontsize=12,fontproperties='Times New Roman')
plt.xticks(fontsize=12,fontproperties='Times New Roman',rotation=45)
plt.subplots_adjust(bottom=0.18, top=0.95, left=0.15, right=0.98)
font1={'family':'Times New Roman','size':12}
plt.legend(frameon=False, loc='upper left',prop=font1)#
ax.set_ylabel('Sobol Index',fontproperties='Times New Roman',fontsize=12)
plt.show()