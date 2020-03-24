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

    dsts = [dst[sel_mask] for sel_mask in masks]

    logging.debug(f' time_fcs_df: len(dsts) = {len(dsts)}')
    fcs =[fit_lifetime(dst[z].values, dst[energy].values,
                       nbins_z, nbins_e, range_z, range_e, fit) for dst in dsts]
    
    
#     print('dst len: ',len(dst),'\n')
#     print('dsts len: ',len(dsts),'\n')
    

    e0s, lts, c2s = pars_from_fcs(fcs)

#     for dst in dsts:
#         x, y, yu = profileX(dst.Z, dst.S2e, 20,range_z,range_e)
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
    for i in range(r):
        fMAP[i] = [fit_fcs_in_xy_bin((i,j), selection_map, event_map, n_time_bins, time_diffs,
                                     nbins_z, nbins_e, range_z,range_e, energy, z, fit, n_min)
                                     for j in range(c) ]
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
