import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.lines as lines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap,hsv_to_rgb
from hsluv import hpluv_to_rgb

def hpluv_to_rgb_vec(H,S,V):
    RGB = np.zeros(H.shape + (3,))
    it = np.nditer(H,flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        RGB[idx] = hpluv_to_rgb((360*H[idx],100*S[idx],100*V[idx]))
    return RGB

hue_cmap = ListedColormap(hpluv_to_rgb_vec(np.linspace(0,1,100),np.ones(100),0.7*np.ones(100)))
lit_cmap = ListedColormap(hpluv_to_rgb_vec(np.zeros(100),np.zeros(100),np.linspace(0,1,100)))

def ytitle(ax,text,xloc=-0.25,**kwargs):
    ax.text(xloc,0.5,text,horizontalalignment='right',verticalalignment='center',
        multialignment='center',rotation='vertical',transform=ax.transAxes,**kwargs)
    
def add_cbar(fig,ax,plot,orientation='vertical',**kwargs):
    cax = add_subax(ax,orientation)
    return fig.colorbar(plot, cax=cax, orientation=orientation, **kwargs)

def add_subax(ax,orientation,size='5%',pad=0.05):
    divider = make_axes_locatable(ax)
    if orientation=='vertical':
        subax = divider.append_axes('right', size=size, pad=pad)
    else:
        subax = divider.append_axes('bottom', size=size, pad=pad)
    return subax
    
def imshowticks(ax,xvals,yvals,xskip=1,yskip=1,xfmt=None,yfmt=None):
    if xfmt is None:
        xfmt = '{:.1f}'
    if yfmt is None:
        yfmt = '{:.1f}'
    ax.set_xticks(np.arange(len(xvals))[::xskip],[xfmt.format(xval) for xval in xvals[::xskip]])
    ax.set_yticks(np.arange(len(yvals))[::yskip],[yfmt.format(yval) for yval in yvals[::yskip]])

def imshow(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,cmap=cmap,**kwargs)

def imshowbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    plot = ax.imshow(A,cmap=cmap,**kwargs)
    return add_cbar(fig,ax,plot)

def doubimsh(fig,ax,A1,A2,hide_ticks=True,cmap_name='RdBu_r',vmin=0,vmax=1,**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    data1 = (A1-vmin)/(vmax-vmin)
    data2 = (A2-vmin)/(vmax-vmin)
    data1 = np.clip(data1,0,1)
    data2 = np.clip(data2,0,1)
    data1 = 0.9*data1 + 0.05
    data2 = 0.9*data2 + 0.05
    cmap = mpl.colormaps[cmap_name](data1)
    cmap += mpl.colormaps[cmap_name](data2)
    cmap = 0.5*cmap
    cmap[...,-1] = 1
    plot = ax.imshow(cmap,**kwargs)

def doubimshbar(fig,ax,A1,A2,hide_ticks=True,cmap_name='RdBu_r',vmin=0,vmax=1,**kwargs):
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    data1 = (A1-vmin)/(vmax-vmin)
    data2 = (A2-vmin)/(vmax-vmin)
    data1 = np.clip(data1,0,1)
    data2 = np.clip(data2,0,1)
    data1 = 0.9*data1 + 0.05
    data2 = 0.9*data2 + 0.05
    cmap = mpl.colormaps[cmap_name](data1)
    cmap += mpl.colormaps[cmap_name](data2)
    cmap = 0.5*cmap
    cmap[...,-1] = 1
    plot = ax.imshow(cmap,**kwargs)
    return add_cbar(fig,ax,plot,cmap=cmap_name,
                    norm=mpl.colors.Normalize(vmin=vmin-0.05*(vmax-vmin),vmax=vmax+0.05*(vmax-vmin)))
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(mpl.cm.ScalarMappable(
    #              norm=mpl.colors.Normalize(vmin=vmin-0.05*(vmax-vmin),vmax=vmax+0.05*(vmax-vmin)),cmap=cmap_name),
    #              cax=cax,orientation='vertical')
    
def contourbar(fig,ax,A,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A.shape[1]),np.arange(A.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    # ax.invert_yaxis()
    plot = ax.contour(x,y,A,cmap=cmap,**kwargs)
    return add_cbar(fig,ax,plot)
    
def doubcont(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A1.shape[1]),np.arange(A1.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    # ax.invert_yaxis()
    plot1 = ax.contour(x,y,A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(x,y,A2,cmap=cmap,**kwargs)
    
def doubcontbar(fig,ax,A1,A2,hide_ticks=True,cmap='RdBu_r',**kwargs):
    x,y = np.meshgrid(np.arange(A1.shape[1]),np.arange(A1.shape[0]))
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.set_aspect('equal')
    # ax.invert_yaxis()
    plot1 = ax.contour(x,y,A1,cmap=cmap,**kwargs)
    plot2 = ax.contour(x,y,A2,cmap=cmap,**kwargs)
    return add_cbar(fig,ax,plot1)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(plot1, cax=cax, orientation='vertical')

def domcol(fig,ax,A,hide_ticks=True,rlim=None,alim=None,**kwargs):
    H = np.angle(A)/(2*np.pi) + 0.5
    # r = np.log2(1. + np.abs(A))
    # S = 0.5 * (1. + np.abs(np.sin(2. * np.pi * r)))
    # V = 0.5 * (1. + np.abs(np.cos(2. * np.pi * r)))
    r = np.abs(A)
    if rlim is None:
        rlim = [None,None]
        rlim[0] = np.min(r)
        rlim[1] = np.max(r)
    elif rlim[0] is None:
        rlim[0] = np.min(r)
    elif rlim[1] is None:
        rlim[1] = np.max(r)
    if alim is None:
        alim = [None,None]
        alim[0] = -np.pi
        alim[1] = np.pi
    r = (r - rlim[0]) / (rlim[1] - rlim[0])
    S = 0.05 + 0.9 * np.cos(0.5 * np.pi * r)
    V = 0.05 + 0.9 * np.sin(0.5 * np.pi * r)
    # rgb = hsv_to_rgb(np.dstack((H,S,V)))
    rgb = hpluv_to_rgb_vec(H,np.ones_like(H),r)#S,V)
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax.imshow(rgb,**kwargs)

def domcolbar(fig,ax,A,hide_ticks=True,rlim=None,alim=None,**kwargs):
    H = np.angle(A)/(2*np.pi) + 0.5
    # r = np.log2(1. + np.abs(A))
    # S = 0.5 * (1. + np.abs(np.sin(2. * np.pi * r)))
    # V = 0.5 * (1. + np.abs(np.cos(2. * np.pi * r)))
    r = np.abs(A)
    if rlim is None:
        rlim = [None,None]
        rlim[0] = np.min(r)
        rlim[1] = 1.1*np.max(r)
    elif rlim[0] is None:
        rlim[0] = np.min(r)
    elif rlim[1] is None:
        rlim[1] = np.max(r)
    if alim is None:
        alim = [None,None]
        alim[0] = -np.pi
        alim[1] = np.pi
    r = (r - rlim[0]) / (rlim[1] - rlim[0])
    S = 0.05 + 0.9 * np.cos(0.5 * np.pi * r)
    V = 0.05 + 0.9 * np.sin(0.5 * np.pi * r)
    # rgb = hsv_to_rgb(np.dstack((H,S,V)))
    rgb = hpluv_to_rgb_vec(H,np.ones_like(H),r)#S,V)
    if hide_ticks:
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    cbars = [None,None]
    plot = ax.imshow(np.angle(A),cmap=hue_cmap,vmin=alim[0],vmax=alim[1],**kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbars[0] = fig.colorbar(plot, cax=cax, orientation='vertical')
    plot = ax.imshow(np.abs(A),cmap=lit_cmap,vmin=rlim[0],vmax=rlim[1],**kwargs)
    # divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.55)
    cbars[1] = fig.colorbar(plot, cax=cax, orientation='vertical')
    ax.imshow(rgb,**kwargs)
    return cbars

def violin(ax,data,colors=None,linestyles=None,fills=None,show_points=False):
    if colors is None:
        colors = [f'C{idx%len(data)}' for idx in range(len(data))]
    if linestyles is None:
        linestyles = ['-' for idx in range(len(data))]
    if fills is None:
        fills = [True for idx in range(len(data))]
    if np.isscalar(show_points):
        show_points = [show_points]*len(data)
        
    rng = np.random.default_rng(0)
        
    parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)
    
    quartile1 = np.zeros(len(data))
    quartile3 = np.zeros(len(data))
    medians = np.zeros(len(data))
    for idx,pc in enumerate(parts['bodies']):
        pc.set_alpha(0.2)
        pc.set_linestyle(linestyles[idx])
        if fills[idx]:
            pc.set_facecolor(colors[idx])
            pc.set_edgecolor(colors[idx])
        else:
            pc.set_facecolor('none')
            pc.set_edgecolor(colors[idx])
        quartile1[idx] = np.percentile(data[idx], 25)
        medians[idx] = np.percentile(data[idx], 50)
        quartile3[idx] = np.percentile(data[idx], 75)
        if show_points[idx]:
            ax.scatter(rng.uniform(idx+1-0.15,idx+1+0.15,len(data[idx])),
                    data[idx], s=1, color=colors[idx], alpha=0.2)

    inds = np.arange(1, len(medians) + 1)
    ax.hlines(medians, inds-0.1, inds+0.1, color=colors, linestyle=linestyles, lw=1)
    ax.vlines(inds, quartile1, quartile3, color=colors, linestyle=linestyles, lw=1)

def get_nstar(pval):
    if pval <= 0.001:
        nstar = 3
    elif pval <= 0.01:
        nstar = 2
    elif pval <= 0.05:
        nstar = 1
    else:
        nstar = 0
    return nstar

def add_p_star(fig,ax,x1,x2,y,nstar):
    if nstar==0:
        text = 'ns'
        width = 0.125
    elif nstar==1:
        text = '∗'
        width = 0.075
    elif nstar==2:
        text = '∗∗'
        width = 0.175
    elif nstar==3:
        text = '∗∗∗'
        width = 0.275
        
    if x2 is not None:
        ax.add_line(lines.Line2D([x1,x2],[y,y],lw=1,color='k',clip_on=False))
        t = ax.text((x1+x2)/2,y,text,ha='center',va='center',clip_on=False)
        width = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width
        ax.add_line(lines.Line2D([(x1+x2)/2-width/2,(x1+x2)/2+width/2],[y,y],lw=2,color='w',clip_on=False))
    else:
        ax.text(x1,y,text,ha='center',va='center',clip_on=False)