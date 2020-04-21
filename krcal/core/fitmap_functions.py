"""Module fitmap_functions.
This module performs lifetime fits to DataFrame maps

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https

Author: JJGC
Last revised: Feb, 2019

"""
import numpy as np
import warnings
import matplotlib.pyplot as plt

from   pandas               import DataFrame

from typing                 import List
from typing                 import Tuple
from typing                 import Dict

from . core_functions       import value_from_measurement
from . core_functions       import uncertainty_from_measurement
from . fit_lt_functions     import fit_lifetime
from . fit_lt_functions     import pars_from_fcs
from . fit_lt_functions     import lt_params_from_fcs
from . selection_functions  import get_time_series_df
from . fit_functions   import   expo_seed
from invisible_cities.core.fit_functions  import profileX
from invisible_cities.core.fit_functions  import expo
from invisible_cities.core.fit_functions  import fit as ft
from . kr_types             import FitType, FitParTS



import logging
log = logging.getLogger(__name__)


def time_fcs_df(ts      : np.array,
                masks   : List[np.array],
                dst     : DataFrame,
                nbins_z : int,
                nbins_e : int,
                range_z : Tuple[float, float],
                range_e : Tuple[float, float],
                energy  : str                 = 'S2e',
                z       : str                 = 'Z',
                fit     : FitType             = FitType.unbined)->FitParTS:
    """
    Fit lifetime of a time series.

    Parameters
    ----------
        ts
            A vector of floats with the (central) values of the time series.
        masks
            A list of boolean vectors specifying the selection masks that define the time series.
        dst
            A dst DataFrame
        range_z
            Range in Z for fit.
        nbins_z
            Number of bins in Z for the fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        energy:
            Takes by default S2e (uses S2e field in dst) but can take any value specified by str.
        z:
            Takes by default Z (uses Z field in dst) but can take any value specified by str.
        fit
            Selects fit type.


    Returns
    -------
        A FitParTs:

    @dataclass
    class FitParTS:             # Fit parameters Time Series
        ts   : np.array         # contains the time series (integers expressing time differences)
        e0   : np.array         # e0 fitted in time series
        lt   : np.array         # lt fitted in time series
        c2   : np.array         # c2 fitted in time series
        e0u  : np.array         # e0 error fitted in time series
        ltu  : np.array

    """

    #print(f'inside time_fcs_df func.')
    dsts = [dst[sel_mask] for sel_mask in masks]

    logging.debug(f' time_fcs_df: len(dsts) = {len(dsts)}')
    fcs =[fit_lifetime(dst[z].values, dst[energy].values,
                       nbins_z, nbins_e, range_z, range_e, fit) for dst in dsts]

#     print('dst len: ',len(dst),'\n')
#     print('dsts len: ',len(dsts),'\n')

    #print(fcs[0].fr.err[0])
    e0s, lts, c2s = pars_from_fcs(fcs)

    """
    ## Plotting
    x =  [0 for x in range(len(dsts))]
    y =  [0 for x in range(len(dsts))]
    yu = [0 for x in range(len(dsts))]
    median = [0 for x in range(len(dsts))]
    mean   = [0 for x in range(len(dsts))]

    #print(f'len(dsts) = {len(dsts[:])}, type(dsts): {type(dsts)}')

    for i in range(0, len(dsts), 1):
        #print(f'inside loop of plotting for time_fcs_df : i = {i}\n')

        median[i] = np.median(dsts[i].S2e)
        mean[i]   = np.mean(dsts[i].S2e)

        fig = plt.figure(figsize=(16,6))

        ax      = fig.add_subplot(1, 2, 1)
        x[i], y[i], yu[i] = profileX(dsts[i].Z, dsts[i].S2e, nbins_z, range_z)
        plt.errorbar(x[i], y[i], yu[i], fmt="kp")
        plt.plot(x[i], fcs[i].fr.par[0]*np.exp(-x[i]/fcs[i].fr.par[1]), 'yellow', (0,500))
        plt.hist2d(dsts[i].Z.values, dsts[i].S2e.values, (30, 30), range=((0,500), range_e), cmap='coolwarm')
        plt.xlabel('Drift time ($\mu$s)')
        plt.ylabel('S2e (pes)')
        plt.colorbar().set_label("Number of events")


        textstr = '\n'.join((
        '$e0={:.2f} \pm {:.2f}$'         .format(fcs[i].fr.par[0], fcs[i].fr.err[0]),
        '$lt={:.2f} \pm {:.2f}$'         .format(fcs[i].fr.par[1], fcs[i].fr.err[1]),
        '$chi2={:.2f} $'                 .format(fcs[i].fr.chi2)
        ))
        textstr2 = '\n'.join((
            '$u_r(e0)$ =  ${:.4f} $'         .format(fcs[i].fr.err[0] / fcs[i].fr.par[0]),
            '$u_r(lt)$  = ${:.4f} $'         .format(fcs[i].fr.err[1] / fcs[i].fr.par[1]),
            '$lt / u(lt) = {:.2f}$'          .format(fcs[i].fr.par[1] / fcs[i].fr.err[1])
            ))

    #    textstr2 = '\n'.join((
    #    '$e0={:.2f} \pm {:.2f}$'         .format(fcs[i].fr.par[0], fcs[i].fr.err[0]),
#        '$lt={:.2f} \pm {:.2f}$'         .format(fcs[i].fr.par[1], fcs[i].fr.err[1]),
#        '$chi2={:.2f} $'                 .format(fcs[i].fr.chi2)
#        ))

        props = dict(boxstyle='square', facecolor='None', alpha=1)
        plt.gca().text(0.02, 1.13, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
        plt.gca().text(0.7, 1.13, textstr2, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)

        ax      = fig.add_subplot(1, 2, 2)
        plt.hist(dst.S2e, bins = 30, range = range_e, histtype='stepfilled', color='crimson', label=f' Entries =  {len(dsts[i].S2e)}')
        plt.xlabel('S2e (pes)')
        plt.ylabel('Entries')
        plt.axvline(x=median[i] , color='blue', alpha=1, linewidth=4, label = f'median = {median[i]:.1f}')
        plt.axvline(x=mean[i]   , color='green', alpha=1, linewidth=4, label = f'mean = {mean[i]:.1f}')
        plt.legend()

        #plt.show()
        plt.savefig(f'/Users/neus/current-work/maps/maps-plots/7878798081/timebins/Fit_time_evolution_{i}.pdf')
        plt.close(fig)
        #print(i)

        """

    # old plotting
    #for dst in dsts:
    #    x, y, yu = profileX(dst.Z, dst.S2e, 20,range_z,range_e)
#         seed      = expo_seed(x, y)
#         seed
#         f      = ft(expo, x, y, seed, sigma=yu)
#         plt.plot(x[:],y[:])
#         plt.errorbar(x, y, yu, fmt="kp")
#         plt.hist2d(dst.Z, dst.S2e, nbins_e, [(0, 400),(5000, 15000)])
#         plt.plot(x, expo(x, *f.values[ :3]),0, alpha=0.9, color='r')
#         print('Checking if doing alright: \n','Seed = ',seed)
#         plt.show()

    return FitParTS(ts  = np.array(ts),
                    e0  = value_from_measurement(e0s),
                    lt  = value_from_measurement(lts),
                    c2  = c2s,
                    e0u = uncertainty_from_measurement(e0s),
                    ltu = uncertainty_from_measurement(lts))


def fit_map_xy_df(selection_map : Dict[int, List[DataFrame]],
                  event_map     : DataFrame,
                  n_time_bins   : int,
                  time_diffs    : np.array,
                  nbins_z       : int,
                  nbins_e       : int,
                  range_z       : Tuple[float, float],
                  range_e       : Tuple[float, float],
                  energy        : str                 = 'S2e',
                  z             : str                 = 'Z',
                  fit           : FitType             = FitType.profile,
                  n_min         : int                 = 100)->Dict[int, List[FitParTS]]:
    """
    Produce a XY map of fits (in time series).

    Parameters
    ----------
        selection_map
            A DataFrameMap of selections, defining a selection of events.
        event_map
            A DataFrame, containing the events in each XY bin.
        n_time_bins
            Number of time bins for the time series.
        time_diffs
            Vector of time differences for the time series.
        nbins_z
            Number of bins in Z for the fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        energy:
            Takes by default S2e (uses S2e field in dst) but can take any value specified by str.
        z:
            Takes by default Z (uses Z field in dst) but can take any value specified by str.
        fit
            Selects fit type.
        n_min
            Minimum number of events for fit.


    Returns
    -------
        A Dict[int, List[FitParTS]]
        @dataclass
        class FitParTS:             # Fit parameters Time Series
            ts   : np.array          # contains the time series (integers expressing time differences)
            e0   : np.array          # e0 fitted in time series
            lt   : np.array
            c2   : np.array
            e0u  : np.array          # e0 error fitted in time series
            ltu  : np.array

    """

    #print(f'inside fit_map_xy_df func.')
    #print(f'dst length = {len(selection_map)}')

    def fit_fcs_in_xy_bin (xybin         : Tuple[int, int],
                           selection_map : Dict[int, List[DataFrame]],
                           event_map     : DataFrame,
                           n_time_bins   : int,
                           time_diffs    : np.array,
                           nbins_z       : int,
                           nbins_e       : int,
                           range_z       : Tuple[float, float],
                           range_e       : Tuple[float, float],
                           energy        : str                 = 'S2e',
                           z             : str                 = 'Z',
                           fit           : FitType             = FitType.profile,
                           n_min         : int                 = 100)->FitParTS:
        """Returns fits in the bin specified by xybin"""

        #print(f'inside fit_fcs_in_xy_bin func.')

        i = xybin[0]
        j = xybin[1]
        nevt = event_map[i][j]
        tlast = time_diffs.max()
        tfrst = time_diffs.min()
        ts, masks =  get_time_series_df(n_time_bins, (tfrst, tlast), selection_map[i][j])

        logging.debug(f' ****fit_fcs_in_xy_bin: bins ={i,j}')

        if nevt > n_min:
            logging.debug(f' events in fit ={nevt}, time series = {ts}')
            return time_fcs_df(ts, masks, selection_map[i][j],
                               nbins_z, nbins_e, range_z, range_e, energy, z, fit)
        else:
            warnings.warn(f'Cannot fit: events in bin[{i}][{j}] ={event_map[i][j]} < {n_min}',
                         UserWarning)

            dum = np.zeros(len(ts), dtype=float)
            dum.fill(np.nan)
            return FitParTS(ts, dum, dum, dum, dum, dum)

    logging.debug(f' fit_map_xy_df')
    fMAP = {}
    r, c = event_map.shape
    #print(f'event_map.shape : r={r}, c={c}')
    for i in range(r):
    #    print(f'inside loop that calls c x r times, i= {i}')
        fMAP[i] = [fit_fcs_in_xy_bin((i,j), selection_map, event_map, n_time_bins, time_diffs,
                                     nbins_z, nbins_e, range_z,range_e, energy, z, fit, n_min)
                                     for j in range(c) ]

    # for plotting
    x =      [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    y =      [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    yu =     [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    median = [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]
    mean   = [[0 for x in range(len(selection_map))] for y in range(len(selection_map))]

    success = 0
    fail = 0
    for k in range(0, len(selection_map), 1):
        for l in range(0, len(selection_map), 1):
            if event_map[k][l] > 1:
                success += 1
                x[k][l], y[k][l], yu[k][l] = profileX(selection_map[k][l].Z, selection_map[k][l].S2e, nbins_z, range_z)
                median[k][l] = np.median(selection_map[k][l].S2e)
                mean[k][l]   = np.mean(selection_map[k][l].S2e)

                textstr = '\n'.join((
                    '$e0={:.2f} \pm {:.2f}$'        .format(fMAP[k][l].e0[0], fMAP[k][l].e0u[0]),
                    '$lt={:.2f} \pm {:.2f}$'        .format(fMAP[k][l].lt[0], fMAP[k][l].ltu[0]),
                    '$chi2={:.2f} ; bin = ({},{})$' .format(fMAP[k][l].c2[0], k, l)
                    ))

                textstr2 = '\n'.join((
                    '$u_r(e0)$ =  ${:.4f} $'             .format(fMAP[k][l].e0u[0]/fMAP[k][l].e0[0]),
                    '$u_r(lt)$  = ${:.4f} $'             .format(fMAP[k][l].ltu[0]/fMAP[k][l].lt[0]),
                    '$lt / u(lt) = {:.2f}$'             .format(fMAP[k][l].lt[0]/fMAP[k][l].ltu[0])
                    ))


                props = dict(boxstyle='square', facecolor='white', alpha=0.5)
                fig = plt.figure(figsize=(16,6))

                ax      = fig.add_subplot(1, 2, 1)
                plt.errorbar(x[k][l], y[k][l], yu[k][l], fmt="kp")
                plt.plot(x[k][l], fMAP[k][l].e0*np.exp(-x[k][l]/fMAP[k][l].lt), 'yellow', (0,500))
                plt.hist2d(selection_map[k][l].Z, selection_map[k][l].S2e, (30, 30), range=((0,500),(8200,11000)), cmap='coolwarm')
                plt.gca().text(0.02, 1.15, textstr, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
                plt.gca().text(0.7, 1.15, textstr2, transform=plt.gca().transAxes, fontsize=11, verticalalignment='top', bbox=props)
                plt.xlabel('Drift time ($\mu$s)')
                plt.ylabel('S2e (pes)')
                plt.colorbar().set_label("Number of events")

                ax      = fig.add_subplot(1, 2, 2)
                plt.hist(selection_map[k][l].S2e, bins = 30, range = range_e, histtype='stepfilled', color='crimson', label=f' Entries =  {len(selection_map[k][l])}')
                #plt.axvline(x=fMAP[k][l].e0, color='black', alpha=1, linewidth=4, label = f'e0 fit = {fMAP[k][l].e0[0]:.1f}')
                plt.axvline(x=median[k][l] , color='blue', alpha=1, linewidth=4, label = f'median = {median[k][l]:.1f}')
                plt.axvline(x=mean[k][l]   , color='green', alpha=1, linewidth=4, label = f'mean = {mean[k][l]:.1f}')
                plt.legend(loc='upper right')
                plt.xlabel('S2e (pes)')
                plt.ylabel('Entries')
                #print(f'median = {median[k][l]}')

                plt.savefig('/Users/neus/current-work/maps/maps-plots/7878798081/{0}/{1}/{0}_{1}_{2}bins_{3}x_{4}y.png'.format(str(fit).replace('FitType.',''), len(selection_map), len(fMAP), k, l ))
                plt.close(fig)
            else:
                fail += 1
                continue

    print(f'events with more than 1 event: {success}')
    print(f'0 events = {fail}')


    print(f'end of xy bin fit')

    return fMAP


def fit_fcs_in_rphi_sectors_df(sector        : int,
                               selection_map : Dict[int, List[DataFrame]],
                               event_map     : DataFrame,
                               n_time_bins   : int,
                               time_diffs    : np.array,
                               nbins_z       : int,
                               nbins_e       : int,
                               range_z       : Tuple[float, float],
                               range_e       : Tuple[float, float],
                               energy        : str                 = 'S2e',
                               z             : str                 = 'Z',
                               fit           : FitType             = FitType.unbined,
                               n_min         : int                 = 100)->List[FitParTS]:
    """
    Returns fits to a (radial) sector of a RPHI-time series map

        Parameters
        ----------
            sector
                Radial sector where the fit is performed.
            selection_map
                A map of selected events defined as Dict[int, List[KrEvent]]
            event_map
                An event map defined as a DataFrame
            n_time_bins
                Number of time bins for the time series.
            time_diffs
                Vector of time differences for the time series.
            nbins_z
                Number of bins in Z for the fit.
            nbins_e
                Number of bins in energy.
            range_z
                Range in Z for fit.
            range_e
                Range in energy.
            energy:
                Takes two values: S2e (uses S2e field in kre) or E (used E field on kre).
                This field allows to select fits over uncorrected (S2e) or corrected (E) energies.
            fit
                Selects fit type.
            n_min
                Minimum number of events for fit.

        Returns
        -------
            A List[FitParTS], one FitParTs per PHI sector.

        @dataclass
        class FitParTS:             # Fit parameters Time Series
            ts   : np.array          # contains the time series (integers expressing time differences)
            e0   : np.array          # e0 fitted in time series
            lt   : np.array
            c2   : np.array
            e0u  : np.array          # e0 error fitted in time series
            ltu  : np.array

    """

    wedges    =[len(kre) for kre in selection_map.values() ]  # number of wedges per sector
    tfrst     = time_diffs[0]
    tlast     = time_diffs[-1]

    fps =[]
    for i in range(wedges[sector]):
        if event_map[sector][i] > n_min:
            ts, masks =  get_time_series_df(n_time_bins, (tfrst, tlast), selection_map[sector][i])
            fp  = time_fcs_df(ts, masks, selection_map[sector][i],
                              nbins_z, nbins_e, range_z, range_e, energy, z, fit)
        else:
            warnings.warn(f'Cannot fit: events in s/w[{sector}][{i}] ={event_map[sector][i]} < {n_min}',
                         UserWarning)

            dum = np.zeros(len(ts), dtype=float)
            dum.fill(np.nan)
            fp  = FitParTS(ts, dum, dum, dum, dum, dum)

        fps.append(fp)
    return fps
