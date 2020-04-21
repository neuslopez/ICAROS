import warnings
import logging

import matplotlib.pyplot as plt
import numpy             as np

from typing          import Tuple
from typing          import Optional
from pandas          import DataFrame

#from   invisible_cities.evm .ic_containers     import Measurement
import   invisible_cities.core.system_of_units as units

from ..core. stat_functions       import mean_and_std
from ..core. core_functions       import divide_np_arrays
from ..core. histo_functions      import profile1d
from ..core. kr_types             import HistoPar2
from ..core. kr_types             import ProfilePar
from ..core. kr_types             import FitPar
from ..core. kr_types             import S1D
from ..core. kr_types             import S2D
from ..core. kr_types             import Array






from ..core. kr_types        import FitParTS
from ..core. kr_types        import FitCollection

log = logging.getLogger(__name__)
def plot_fit_lifetime(fc : FitCollection):
    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err

        if fc.hp:
            plt.hist2d(fc.hp.var,
                        fc.hp.var2,
                        bins = (fc.hp.nbins,fc.hp.nbins2),
                        range= (fc.hp.range,fc.hp.range2))
        x = fc.fp.x
        y = fc.fp.y
        xu = fc.fp.xu
        yu = fc.fp.yu
        f = fc.fp.f

        plt.errorbar(x, y, yu, xu[0], fmt="kp", ms=7, lw=3)
        plt.plot(x, f(x), "r-", lw=4)
        plt.xlabel('Z')
        plt.ylabel('E')
        plt.title(f'Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)



def plot_time_fcs(fps        : Optional[FitParTS],
                  range_chi2 : Tuple[float, float] = (0,10),
                  range_e0   : Tuple[float, float] = (8000,13500),
                  range_lt   : Tuple[float, float] = (2000, 4000),
                  figsize    : Tuple[int, int]     = (12,6)):
    if fps == None:
        print('Trying to plot a null fit. Refit and try again')
        return None


    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 3, 1)
    (_) = plt.errorbar(fps.ts, fps.e0, fps.e0u, fmt="p")
    plt.ylim(range_e0)
    plt.xlabel('time (s)')
    plt.ylabel('e0 (pes)')
    ax  = fig.add_subplot(1, 3, 2)
    (_) = plt.errorbar(fps.ts, fps.lt, fps.ltu, fmt="p")
    plt.ylim(range_lt)
    plt.xlabel('time (s)')
    plt.ylabel('lt (mus)')
    ax  = fig.add_subplot(1, 3, 3)
    (_) = plt.errorbar(fps.ts, fps.c2, np.sqrt(fps.c2), fmt="p")
    plt.ylim(range_chi2)
    plt.xlabel('time (s)')
    plt.ylabel('chi2')
    plt.tight_layout()

'''
##not used and use deprecated Measurement
def s1d_from_dst(dst       : DataFrame,
                 range_s1e : Tuple[float, float] = (0,40),
                 range_s1w : Tuple[float, float] = (0,500),
                 range_s1h : Tuple[float, float] = (0,10),
                 range_s1t : Tuple[float, float] = (0,600))->S1D:

    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)
    return S1D(E = Measurement(*mean_and_std(dst.S1e.values,range_s1e)),
               W = Measurement(*mean_and_std(dst.S1w.values,range_s1w)),
               H = Measurement(*mean_and_std(dst.S1h.values,range_s1h)),
               R = Measurement(*mean_and_std(hr,(0,1))),
               T = Measurement(*mean_and_std(dst.S1t.values,range_s1t)))


def s2d_from_dst(dst : DataFrame)->S2D:
    return S2D(E = Measurement(*mean_and_std(dst.S2e.values,(0,20000))),
               W = Measurement(*mean_and_std(dst.S2w.values,(0,30))),
               Q = Measurement(*mean_and_std(dst.S2q.values,(0,1000))),
               N = Measurement(*mean_and_std(dst.Nsipm.values,(0,40))),
               X = Measurement(*mean_and_std(dst.X.values,(-200,200))),
               Y = Measurement(*mean_and_std(dst.Y.values,(-200,200))))

'''


def plot_s1histos(dst, s1d, bins=20,
                  range_s1e = (0,40),
                  range_s1w = (0,500),
                  range_s1h = (0,10),
                  range_s1t = (0,800),
                  figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S1 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)#ylabel
    ax.hist(dst.S1e,
            range=range_s1e,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.E.value, s1d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1w,
            range=range_s1w,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.W.value, s1d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1h,
            range=range_s1h,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.H.value, s1d.H.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)
    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)

    ax.set_xlabel(r'height / energy ',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(hr,
            range=(0,1),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1t / units.mus,
            range=range_s1t,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    plt.hist2d(dst.S1t/units.mus, dst.S1e, bins=10, range=(range_s1t,range_s1e))
    plt.colorbar()
    ax.set_xlabel(r'S1 time ($\mu$s) ',fontsize = 11) #xlabel
    ax.set_ylabel('S1 height (pes)', fontsize = 11)
    plt.grid(True)

    plt.tight_layout()


def plot_s2histos(df, s2d, bins=20, emin=3000, emax=15000, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# events', fontsize = 11)#ylabel
    ax.hist(df.S2e,
            range=(emin, emax),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.E.value, s2d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)

    ax.set_xlabel(r'S2 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2w,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.W.value, s2d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)

    ax.set_xlabel(r'Q (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2q,
            range=(0,1000),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Q.value, s2d.Q.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)

    ax.set_xlabel(r'number SiPM',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Nsipm,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.N.value, s2d.N.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)

    ax.set_xlabel(r' X (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.X,
            range=(-220,220),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.X.value, s2d.X.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    ax.set_xlabel(r' Y (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Y,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Y.value, s2d.Y.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()



def plot_histo(x_lab, y_lab, title, ax, legend= True,
               legendsize=10, legendloc='best', labelsize=11):

    if legend:
        ax.legend(fontsize= legendsize, loc=legendloc)
    ax.set_xlabel(x_lab,fontsize = labelsize)
    ax.set_ylabel(y_lab, fontsize = labelsize)
    if title:
        plt.title(title)

def h1(x      : np.array,
       bins    : int,
       range   : Tuple[float],
       weights : Array = None,
       log     : bool  = False,
       normed  : bool  = False,
       color   : str   = 'black',
       width   : float = 1.5,
       style   : str   ='solid',
       stats   : bool  = True,
       lbl     : Optional[str]  = None):
    """
    histogram 1d with continuous steps and display of statsself.
    number of bins (bins) and range are compulsory.
    """

    mu, std = mean_and_std(x, range)

    if stats:
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(mu)
        sigma    =  r'$\sigma$ = {:7.2f}'.format(std)
        stat     =  f'{entries}\n{mean}\n{sigma}'
    else:
        stat     = ''

    if lbl == None:
        lab = ' '
    else:
        lab = lbl

    lab = stat + lab

    if color == None:
        n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
                       histtype  = 'step',
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    else:

        n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
                       histtype  = 'step',
                       edgecolor = color,
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    return n, b, mu, std



def h2(x         : np.array,
       y         : np.array,
       nbins_x   : int,
       nbins_y   : int,
       range_x   : Tuple[float],
       range_y   : Tuple[float],
       profile   : bool   = True):

    xbins  = np.linspace(*range_x, nbins_x + 1)
    ybins  = np.linspace(*range_y, nbins_y + 1)

    nevt, *_  = plt.hist2d(x, y, (xbins, ybins))
    plt.colorbar().set_label("Number of events")

    if profile:
        x, y, yu     = profile1d(x, y, nbins_x, range_x)
        plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    return nevt



def plot_selection_in_band(fpl    : FitPar,
                           fph    : FitPar,
                           hp     : HistoPar2,
                           pp     : ProfilePar,
                           nsigma : float   = 3.5,
                           figsize=(10,6)):
    z       = hp.var
    e       = hp.var2
    range_z = hp.range
    range_e = hp.range2
    nbins_z = hp.nbins
    nbins_e = hp.nbins2

    zc     = pp.x
    emean  = pp.y
    zerror = pp.xu
    esigma = pp.yu

    zbins  = np.linspace(*range_z, nbins_z + 1)
    ebins  = np.linspace(*range_e, nbins_e + 1)


    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Energy-like')
    ax.set_ylabel('Events')
    plt.suptitle('true')
    nevt, *_  = plt.hist2d (z, e, (zbins, ebins))

    plt.errorbar(zc, emean, esigma, zerror,
                 "kp", label="Kr peak energy $\pm 1 \sigma$")

    plt.plot    (zbins,  fpl.f(zbins),  "m", lw=2, label="$\pm "+str(nsigma)+" \sigma$ region")
    plt.plot    (zbins,  fph.f(zbins),  "m", lw=2)
    plt.legend()
    ax.set_xlabel('Energy-like')
    ax.set_ylabel('Events')
    plt.suptitle('true')
