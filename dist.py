from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy import std
from pandas import Series
from scipy.stats import kstest
from scipy import optimize

from ._typing import ContinuousDist

import logging # Import logging
# Import lmoments3
try:
    from lmoments3 import distr as lm_distr
except ImportError:
    lm_distr = None
    logging.warning("lmoments3 not found. L-moment fitting will not be available.")

# Setup basic logger for this file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _weibull_negloglik(params, data):
    c, scale = params
    if scale <= 0 or c <= 0: # Shape parameter c must also be positive
        return np.inf
    from scipy.stats import weibull_min
    return -np.sum(weibull_min.logpdf(data, c=c, scale=scale))

def _gamma_negloglik(params, data):
    a, scale = params
    if a <= 0 or scale <= 0:
        return np.inf
    from scipy.stats import gamma
    return -np.sum(gamma.logpdf(data, a=a, scale=scale))

def _gumbel_negloglik(params, data):
    # Gumbel (right) has shape=0 in scipy extreme value family; scipy has gumbel_r (loc, scale)
    loc, scale = params
    if scale <= 0:
        return np.inf
    from scipy.stats import gumbel_r
    return -np.sum(gumbel_r.logpdf(data, loc=loc, scale=scale))

def _pearson3_negloglik(params, data):
    skew, loc, scale = params
    if scale <= 0:
        return np.inf
    from scipy.stats import pearson3
    return -np.sum(pearson3.logpdf(data, skew, loc=loc, scale=scale))

def _gev_negloglik(params, data):
    c, loc, scale = params
    if scale <= 0:
        return np.inf
    from scipy.stats import genextreme
    # scipy.genextreme takes shape c (note sign conventions with some lmoments packages)
    return -np.sum(genextreme.logpdf(data, c, loc=loc, scale=scale))

# Helper function to estimate initial parameters using method of moments
from scipy.special import gamma as gamma_func
from scipy.stats import skew
from scipy.optimize import root_scalar

# Helper function to estimate Weibull parameters using method of moments
def estimate_weibull_moments(data):
    mean = data.mean()
    var = data.var(ddof=1)
    if mean <= 0 or var <= 0:
        # Fallback values if mean or variance are non-positive
        return 1.0, 0.0, data.std(ddof=1)
    r = var / mean**2

    # Define function for root finding to solve for shape parameter c
    def func(k):
        return (gamma_func(1 + 2/k) / (gamma_func(1 + 1/k)**2)) - 1 - r
    try:
        # Find root of func in interval [0.1, 10]
        sol = root_scalar(func, bracket=[0.5, 10], method='brentq')
        c = sol.root
    except Exception:
        # Fallback shape parameter if root finding fails
        c = 1.0
    # Calculate scale parameter from mean and shape
    scale = mean / gamma_func(1 + 1/c)
    return c, scale

# Helper function to estimate Gumbel parameters using method of moments
def estimate_gumbel_moments(data):
    mean = data.mean()
    std = data.std(ddof=1)
    gamma_const = 0.5772156649  # Euler-Mascheroni constant
    # Scale parameter estimated from std deviation
    scale = std * np.sqrt(6) / np.pi
    # Location parameter estimated from mean and scale
    loc = mean - gamma_const * scale
    return loc, scale

# Helper function to estimate Pearson3 parameters using method of moments
def estimate_pearson3_moments(data):
    mean = data.mean()
    std = data.std(ddof=1)
    skewness = skew(data)
    if skewness == 0:
        # Avoid division by zero in skewness
        skewness = 1e-6
    loc = np.min(data)  # Location estimated as minimum of data
    return skewness, loc, std

# Helper function to estimate GEV parameters using method of moments
def estimate_gev_moments(data):
    mean = data.mean()
    var = data.var(ddof=1)
    sample_skew = skew(data)

    # Theoretical skewness function of shape parameter c
    def skewness_func(c):
        num = gamma_func(1 - 3*c) - 3*gamma_func(1 - c)*gamma_func(1 - 2*c) + 2*gamma_func(1 - c)**3
        den = (gamma_func(1 - 2*c) - gamma_func(1 - c)**2)**(3/2)
        return num / den

    # Objective function for root finding: difference between theoretical and sample skewness
    def objective(c):
        return skewness_func(c) - sample_skew

    try:
        # Find shape parameter c in interval [-0.5, 0.5]
        sol = root_scalar(objective, bracket=[-0.3, 0.3], method='brentq')
        c = sol.root
    except Exception:
        # Fallback shape parameter if root finding fails
        c = 0.1

    # Calculate scale parameter sigma from variance and shape
    sigma = abs(c) * np.sqrt(var / (gamma_func(1-2*c)-gamma_func(1-c)**2))
    # Calculate location parameter mu from mean, scale, and shape
    mu = mean - (sigma / c) * (gamma_func(1 - c) - 1)
    return c, mu, sigma

def estimate_lognorm_moments(data):
    """
    Estimate Lognormal parameters (shape=sigma, loc, scale=exp(mu))
    using method of moments from sample mean and variance.
    """
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    if mean <= 0 or var <= 0:
        return 1.0, 0.0, 1.0

    # Moment relationships:
    # mean = exp(mu + sigma^2 / 2)
    # var = (exp(sigma^2) - 1) * exp(2mu + sigma^2)
    sigma2 = np.log(1 + var / mean**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    scale = np.exp(mu)
    return sigma, 0.0, scale

def estimate_fisk_moments(data):
    """
    Estimate Fisk (Log-logistic) parameters using method of moments.
    The distribution has shape c, loc, scale.
    """
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    if mean <= 0 or var <= 0:
        return 1.0, 0.0, 1.0

    # Theoretical mean and var for fisk(c, loc=0, scale=s):
    # mean = s * (π/c) / sin(π/c)  for c > 1
    # var = s^2 * [ (2π/c) / sin(2π/c) - (π/c / sin(π/c))^2 ]  for c > 2
    # We solve for c numerically, then s from mean.

    def func(c):
        if c <= 2.01:
            return 1e6  # avoid invalid
        term1 = (2 * np.pi / c) / np.sin(2 * np.pi / c)
        term2 = ((np.pi / c) / np.sin(np.pi / c))**2
        theo_var_ratio = term1 - term2
        return theo_var_ratio - (var / mean**2)

    try:
        sol = root_scalar(func, bracket=[2.05, 50], method="brentq")
        c = sol.root
    except Exception:
        c = 5.0  # fallback

    scale = mean * np.sin(np.pi / c) * c / np.pi
    return c, 0.0, scale


@dataclass
class Dist:
    """A wrapper around scipy continuous distributions for SPI/SI usage.

    Behaviour:
    - Fits the provided distribution to the positive (>
      0) portion of the data using L-moments (lmoments3) for initial
      estimates when available and then refines estimates with
      Nelder-Mead (scipy.optimize.minimize) where implemented.
    - Computes zero-probability using the 'centre of mass' formula from
      Stagge et al. (2015): p0 = (m + 1) / (2*(n + 1)). This is used if
      prob_zero=True.

    The API remains compatible with the original dist.py: methods
    `cdf()`, `pdf()` and `ppf()` return pandas.Series with the same
    index as the data used to create the Dist object.
    """

    data: Series = field(init=True, repr=False)
    dist: ContinuousDist = field(init=True, repr=False)
    prob_zero: bool = field(default=False, init=True, repr=True)
    data_window: Series | None = field(default=None, init=True, repr=False)

    # fitted parameters (populated in __post_init__)
    loc: float = field(init=False, repr=True)
    scale: float = field(init=False, repr=True)
    pars: list[float] | None = field(init=False, repr=False)
    p0: float = field(init=False, repr=False, default=0.0)

    def __post_init__(self):
        # select fitting data (data_window overrides)
        data_fit = self.data_window if self.data_window is not None else self.data

        # Call the new fitting method that mimics utilsPY.py
        self.pars, self.loc, self.scale, self.p0 = self.fit_dist(
            data=data_fit,
            dist=self.dist,
            prob_zero=self.prob_zero
        )

    @staticmethod
    def fit_dist(
        data: Series, dist: ContinuousDist, prob_zero: bool
    ) -> tuple[list[float] | None, float, float, float]:
        """
        Fits a Scipy continuous distribution to the data using L-moments for initial
        parameters and then MLE, mimicking the utilsPY.py approach.
        Handles zero precipitation probability as per Stagge et al. (2015).

        Parameters
        ----------
        data : Series
            The input data for fitting.
        dist : ContinuousDist
            The continuous distribution to be fitted.
        prob_zero : bool
            Flag indicating whether to calculate the probability of zero values.

        Returns
        -------
        Tuple
            Tuple containing distribution parameters (pars, loc, scale) and p0.
        """
        arr = np.asarray(data.dropna(), dtype=float)
        n = arr.size

        if n == 0:
            return None, np.nan, np.nan, 0.0 # p0 is 0 if no data

        m = int(np.sum(arr == 0.0)) # Number of zeros
        n_non_zero = n - m # Number of non-zero values

        # Calculate p0 using Stagge et al. (2015) center-of-mass formula
        p0_val = (m / (n + 1)) if prob_zero and m > 0 else 0.0 # p0_val == est

        # --- Parameter Estimation Logic (4 Cases from utilsPY.py) ---
        # Case 4: All values are zero
        if p0_val > 0 and n_non_zero == 0: # Equivalent to (est > 0 and nn == npo)
            logger.debug(f"Case 4: All values are zero for {dist.name}. Parameters set to NaN.")
            return None, np.nan, np.nan, p0_val

        # Data for fitting (only positive values for distributions like Gamma, Weibull)
        pos_data_original = arr[arr > 0.0]
        
        if pos_data_original.size == 0:
            logger.debug(f"No positive data for fitting {dist.name}. Parameters set to NaN.")
            return None, np.nan, np.nan, p0_val

        # Prepare data for fitting, potentially appending a small value
        data_for_fitting = pos_data_original       # faster computing
        name = getattr(dist, 'name', None)

        # Apply the small value appending logic as in utilsPY.py for relevant distributions
        # This covers Case 1 and Case 3 where zeros exist and non-zero data is used
        if prob_zero and p0_val > 0 and n_non_zero > 0 and name in ['gamma', 'weibull_min', 'genextreme', 'pearson3', 'gumbel_r', 'lognorm', 'fisk']:
            # Only append if pos_data_original is not empty
            if pos_data_original.size > 0:
                data_for_fitting = np.concatenate((pos_data_original, [np.min(pos_data_original) * 0.01]))  #faster computing
            # else: data_for_fitting remains empty, handled by previous check
        
        # Ensure data_for_fitting is not empty after potential modifications
        if data_for_fitting.size == 0:
            logger.debug(f"No valid data for fitting after preprocessing for {dist.name}. Parameters set to NaN.")
            return None, np.nan, np.nan, p0_val

        # --- Initial Parameter Estimation ---
        initial_params = None

        # Try L-moments for initial estimates (Case 1 & 2 primary, also fallback for Case 3)
        if lm_distr is not None:
            lm_map = {
                'gamma': 'gam',
                'weibull_min': 'wei',
                'gumbel_r': 'gum',
                'pearson3': 'pe3',
                'genextreme': 'gev',
            }
            lm_name = lm_map.get(name, None)

            if lm_name is None:
                # Skip L-moment fitting for distributions not supported by lmoments3
                logger.debug(f"L-moment fitting not supported for {name}, skipping.")
            else:
                if hasattr(lm_distr, lm_name):
                    lm_func = getattr(lm_distr, lm_name)
                    try:
                        lm_pars = lm_func.lmom_fit(data_for_fitting)  # Use data_for_fitting
                        # --- assign parameter arrays by distribution type ---
                        if name == 'gamma':
                            initial_params = np.asarray([lm_pars['a'], lm_pars['scale']])
                        elif name == 'weibull_min':
                            initial_params = np.asarray([lm_pars['c'], lm_pars['scale']])
                        elif name == 'gumbel_r':
                            initial_params = np.asarray([lm_pars['loc'], lm_pars['scale']])
                        elif name == 'pearson3':
                            initial_params = np.asarray([lm_pars['skew'], lm_pars['loc'], lm_pars['scale']])
                        elif name == 'genextreme':
                            initial_params = np.asarray([-lm_pars['c'], lm_pars['loc'], lm_pars['scale']])

                        logger.debug(f"L-moment initial params for {name}: {initial_params}")

                    except Exception as e:
                        logger.warning(f"L-moment fitting failed for {name}: {e}. Trying method of moments or scipy default.")
                        initial_params = None

            # --- If no zero values, initial parameters were estimated by L-moments ---
            if initial_params is None and p0_val == 0 and lm_name is not None:
                try:
                    lm_func = getattr(lm_distr, lm_name)
                    lm_pars = lm_func.lmom_fit(pos_data_original)  # Use pos_data_original directly

                    if name == 'gamma':
                        initial_params = np.asarray([lm_pars['a'], lm_pars['scale']])
                    elif name == 'weibull_min':
                        initial_params = np.asarray([lm_pars['c'], lm_pars['scale']])
                    elif name == 'gumbel_r':
                        initial_params = np.asarray([lm_pars['loc'], lm_pars['scale']])
                    elif name == 'pearson3':
                        initial_params = np.asarray([lm_pars['skew'], lm_pars['loc'], lm_pars['scale']])
                    elif name == 'genextreme':
                        initial_params = np.asarray([-lm_pars['c'], lm_pars['loc'], lm_pars['scale']])

                    logger.debug(f"Retry L-moment fitting (no zeros) for {name}: {initial_params}")

                except Exception as e:
                    logger.warning(f"L-moment retry failed for {name}: {e}")
                    initial_params = None

        else:
            logger.warning("lmoments3 not installed. Falling back to method of moments or scipy default for initial parameters.")

        # Fallback to Method of Moments for initial guess (Specifically for Case 3, or if L-moments fail)
        # Case 3: Zeros exist but insufficient non-zero data (1 <= n_non_zero <= 3)
        if initial_params is None and (p0_val > 0 and 1 <= n_non_zero <= 3):
            logger.debug(f"Case 3: Insufficient non-zero data for {name}. Using Method of Moments for initial guess.")
            # Use data_for_fitting for moments calculation as the appended value is for fitting stability
            # Main parameter initialization logic
            if name == 'gamma':
                # Calculate mean and variance from data
                m1 = data_for_fitting.mean()
                v1 = data_for_fitting.var(ddof=1)
                # Method of moments estimates for gamma shape and scale
                a0 = m1 * m1 / v1 if v1 > 0 else 1.0
                scale0 = v1 / m1 if m1 > 0 else 1.0
                # Include loc=0 as gamma in scipy has shape, loc, scale
                initial_params = np.asarray([a0, scale0])
            elif name == 'weibull_min':
                # Estimate Weibull shape, scale using method of moments helper
                c0, scale0 = estimate_weibull_moments(data_for_fitting)
                initial_params = np.asarray([c0, scale0])
            elif name == 'gumbel_r':
                # Estimate Gumbel location and scale using method of moments helper
                loc0, scale0 = estimate_gumbel_moments(data_for_fitting)
                initial_params = np.asarray([loc0, scale0])
            elif name == 'pearson3':
                # Estimate Pearson3 skewness, location, scale using method of moments helper
                skew0, loc0, scale0 = estimate_pearson3_moments(data_for_fitting)
                initial_params = np.asarray([skew0, loc0, scale0])
            elif name == 'genextreme':
                # Estimate GEV shape, location, scale using method of moments helper
                c0, loc0, scale0 = estimate_gev_moments(data_for_fitting)
                initial_params = np.asarray([c0, loc0, scale0])
            elif name == 'lognorm':
                sigma0, loc0, scale0 = estimate_lognorm_moments(data_for_fitting)
                initial_params = np.asarray([sigma0, loc0, scale0])
            elif name == 'fisk':
                c0, loc0, scale0 = estimate_fisk_moments(data_for_fitting)
                initial_params = np.asarray([c0, loc0, scale0])
            else:
                # For other distributions, use scipy's fit method to get initial parameters
                try:
                    fit_tuple = dist.fit(data_for_fitting)
                    initial_params = np.asarray(fit_tuple)
                except Exception:
                    # If fitting fails, set initial_params to None
                    initial_params = None
        
        # --- Maximum Likelihood Estimation (MLE) using Nelder-Mead ---
        fitted_pars = None
        fitted_loc = np.nan
        fitted_scale = np.nan

        # Helper function to parse the fit tuple from scipy.fit()
        def _parse_fit_tuple(fit_tup):
            if len(fit_tup) == 2:
                return None, float(fit_tup[0]), float(fit_tup[1])
            elif len(fit_tup) == 3:
                return [float(fit_tup[0])], float(fit_tup[1]), float(fit_tup[2])
            else:
                *pars_list, loc_f, scale_f = fit_tup
                pars_list = [float(x) for x in pars_list] if pars_list else None
                return pars_list, float(loc_f), float(scale_f)

        # === Step 1: Primary MLE estimation (Nelder–Mead method) ===
        try:
            # --- Define the negative log-likelihood objective dynamically using lmoments3 ---
            if name == 'gamma':
                def _obj_fn(x):
                    a, scale = x
                    return lm_distr.gam.nnlf(data_for_fitting, a=a, loc=0.0, scale=scale)

            elif name == 'weibull_min':
                def _obj_fn(x):
                    c, scale = x
                    return lm_distr.wei.nnlf(data_for_fitting, c=c, loc=0.0, scale=scale)

            elif name == 'gumbel_r':
                def _obj_fn(x):
                    loc, scale = x
                    return lm_distr.gum.nnlf(data_for_fitting, loc=loc, scale=scale)

            elif name == 'pearson3':
                def _obj_fn(x):
                    skew, loc, scale = x
                    return lm_distr.pe3.nnlf(data_for_fitting, skew=skew, loc=loc, scale=scale)

            elif name == 'genextreme':
                def _obj_fn(x):
                    # lmoments3 uses the opposite sign convention for c compared to SciPy.
                    c, loc, scale = x
                    return lm_distr.gev.nnlf(data_for_fitting, c=c, loc=loc, scale=scale)
                
            elif name == 'lognorm':
                def _obj_fn(x):
                    s, loc, scale = x
                    from scipy.stats import lognorm
                    return -np.sum(lognorm.logpdf(data_for_fitting, s, loc=0.0, scale=scale)) # loc=0.0 fixed for 2 params
                
            elif name == 'fisk':
                def _obj_fn(x):
                    c, loc, scale = x
                    from scipy.stats import fisk
                    return -np.sum(fisk.logpdf(data_for_fitting, c, loc=0.0, scale=scale)) # loc=0.0 fixed for 2 params
                
            else:
                raise RuntimeError(f"{name} not handled in nnlf-based MLE block")

            # --- Perform MLE optimization using Nelder–Mead ---
            # Ensure proper length of initial parameters
            if name in ['lognorm', 'fisk']:
                if initial_params is None or len(initial_params) != 3:
                    # fallback to method of moments if incomplete
                    if name == 'lognorm':
                        initial_params = np.asarray(estimate_lognorm_moments(data_for_fitting))
                    else:
                        initial_params = np.asarray(estimate_fisk_moments(data_for_fitting))

            res = optimize.minimize(_obj_fn, x0=initial_params, method='Nelder-Mead')

            # --- Extract results if optimization was successful ---
            if res.success and np.all(np.isfinite(res.x)):
                if name == 'gamma':
                    fitted_pars = [float(res.x[0])]
                    fitted_loc = 0.0
                    fitted_scale = float(res.x[1])

                elif name == 'weibull_min':
                    fitted_pars = [float(res.x[0])]
                    fitted_loc = 0.0
                    fitted_scale = float(res.x[1])

                elif name == 'gumbel_r':
                    fitted_pars = None
                    fitted_loc = float(res.x[0])
                    fitted_scale = float(res.x[1])

                elif name == 'pearson3':
                    fitted_pars = [float(res.x[0])]
                    fitted_loc = float(res.x[1])
                    fitted_scale = float(res.x[2])

                elif name == 'genextreme':
                    fitted_pars = [float(res.x[0])]
                    fitted_loc = float(res.x[1])
                    fitted_scale = float(res.x[2])

                elif name == 'lognorm':
                    # Lognormal shape, loc, scale
                    fitted_pars = [float(res.x[0])]     # shape (σ)
                    fitted_loc = float(res.x[1])        # loc fixed at 0.0
                    fitted_scale = float(res.x[2])      # scale = exp(μ)

                elif name == 'fisk':
                    # Fisk (log-logistic) shape, loc, scale
                    fitted_pars = [float(res.x[0])]     # shape (c)
                    fitted_loc = float(res.x[1])        # loc fixed at 0.0
                    fitted_scale = float(res.x[2])      # scale (s)
            else:
                raise RuntimeError(f"Nelder–Mead MLE failed for {name}")

        # === Step 2: If MLE fails, fallback to L-moments  ===
        except Exception as e_mle:
            logger.warning(f"MLE failed for {name}: {e_mle}. Trying L-moments fallback")

            lm_success = False
            if lm_distr is not None:
                # lm_map = {
                #     'gamma': 'gam', 'weibull_min': 'wei', 'gumbel_r': 'gum',
                #     'pearson3': 'pe3', 'genextreme': 'gev',
                # }
                lm_map = {
                    'gamma': 'gam', 'weibull_min': 'wei', 'gumbel_r': 'gum',
                    'pearson3': 'pe3'
                }
                lm_name = lm_map.get(name, None)
                if lm_name is not None and hasattr(lm_distr, lm_name):
                    try:
                        lm_func = getattr(lm_distr, lm_name)
                        lm_pars = lm_func.lmom_fit(data_for_fitting)
                        if name == 'gamma':
                            fitted_pars = [lm_pars['a']]
                            fitted_loc = 0.0
                            fitted_scale = lm_pars['scale']
                        elif name == 'weibull_min':
                            fitted_pars = [lm_pars['c']]
                            fitted_loc = 0.0
                            fitted_scale = lm_pars['scale']
                        elif name == 'gumbel_r':
                            fitted_pars = None
                            fitted_loc = lm_pars['loc']
                            fitted_scale = lm_pars['scale']
                        elif name == 'pearson3':
                            fitted_pars = [lm_pars['skew']]
                            fitted_loc = lm_pars['loc']
                            fitted_scale = lm_pars['scale']
                        # elif name == 'genextreme':
                        #     fitted_pars = [-lm_pars['c']]
                        #     fitted_loc = lm_pars['loc']
                        #     fitted_scale = lm_pars['scale']
                        lm_success = True
                        logger.info(f"L-moments fallback succeeded for {name}.")
                    except Exception as e_lmom:
                        logger.warning(f"L-moment fitting failed for {name}: {e_lmom}. Proceeding to scipy.fit().")

                else:
                    logger.debug(f"L-moments fallback not supported for {name}, skipping to scipy.fit().")

            # If L-moments also fail or unsupported → fallback to scipy.fit()
            if not lm_success:
                # --- Step 3: Try scipy.fit() with initial parameters ---
                try:
                    shape_args = ()
                    fit_kwargs = {}

                    if initial_params is not None:
                        if name == 'gamma':
                            shape_args = (float(initial_params[0]),)
                            fit_kwargs = {'loc': 0.0, 'scale': float(initial_params[1])}
                        elif name == 'weibull_min':
                            shape_args = (float(initial_params[0]),)
                            fit_kwargs = {'loc': 0.0, 'scale': float(initial_params[1])}
                        elif name == 'gumbel_r':
                            fit_kwargs = {'loc': float(initial_params[0]), 'scale': float(initial_params[1])}
                        elif name == 'pearson3':
                            shape_args = (float(initial_params[0]),)
                            fit_kwargs = {'loc': float(initial_params[1]), 'scale': float(initial_params[2])}
                        # elif name == 'genextreme':
                        #     shape_args = (float(initial_params[0]),)
                        #     fit_kwargs = {'loc': float(initial_params[1]), 'scale': float(initial_params[2])}
                        elif name == 'lognorm':
                            shape_args = (float(initial_params[0]),)
                            fit_kwargs = {'loc': float(initial_params[1]), 'scale': float(initial_params[2])}
                            # fit_kwargs = {'loc': 0.0, 'scale': float(initial_params[2])}
                        elif name == 'fisk':
                            shape_args = (float(initial_params[0]),)
                            fit_kwargs = {'loc': float(initial_params[1]), 'scale': float(initial_params[2])}
                            # fit_kwargs = {'loc': 0.0, 'scale': float(initial_params[2])}
                    if shape_args:
                        fit_tuple = dist.fit(data_for_fitting, *shape_args, **fit_kwargs)
                    else:
                        fit_tuple = dist.fit(data_for_fitting, **fit_kwargs) if fit_kwargs else logger.warning(f"scipy.fit with initial params failed for {name}: {e_mle}. Trying scipy.fit() without initials.")

                    fitted_pars, fitted_loc, fitted_scale = _parse_fit_tuple(fit_tuple)
                    logger.info(f"scipy.fit succeeded for {name} using initial params.")

                except Exception as e_fit1:
                    logger.warning(f"scipy.fit with initial params failed for {name}: {e_fit1}. Trying scipy.fit() without initials.")

                    # --- Step 4: Try scipy.fit() without any initial parameters ---
                    try:
                        fit_tuple = dist.fit(data_for_fitting)
                        fitted_pars, fitted_loc, fitted_scale = _parse_fit_tuple(fit_tuple)
                        logger.info(f"scipy.fit succeeded for {name} without initial params.")
                    except Exception as e_fit2:
                        logger.error(f"scipy.fit without initial params also failed for {name}: {e_fit2}. Using initial_params as fallback (non-MLE).")

                        # --- Step 5: Use initial_params as final non-MLE fallback ---
                        if initial_params is not None:
                            if name == 'gamma':
                                fitted_pars = [float(initial_params[0])]
                                fitted_loc = 0.0
                                fitted_scale = float(initial_params[1])
                            elif name == 'weibull_min':
                                fitted_pars = [float(initial_params[0])]
                                fitted_loc = 0.0
                                fitted_scale = float(initial_params[1])
                            elif name == 'gumbel_r':
                                fitted_pars = None
                                fitted_loc = float(initial_params[0])
                                fitted_scale = float(initial_params[1])
                            elif name == 'pearson3':
                                fitted_pars = [float(initial_params[0])]
                                fitted_loc = float(initial_params[1])
                                fitted_scale = float(initial_params[2])
                            elif name == 'genextreme':
                                fitted_pars = [float(initial_params[0])]
                                fitted_loc = float(initial_params[1])
                                fitted_scale = float(initial_params[2])
                            else:
                                # Generic fallback
                                if len(initial_params) >= 2:
                                    fitted_pars = list(initial_params[:-2]) or None
                                    fitted_loc = float(initial_params[-2])
                                    fitted_scale = float(initial_params[-1])
                                else:
                                    fitted_pars = None
                                    fitted_loc = np.nan
                                    fitted_scale = np.nan

                            logger.warning(f"Using initial_params as final fit (non-MLE) for {name}.")
                        else:
                            fitted_pars, fitted_loc, fitted_scale = None, np.nan, np.nan
                            logger.error(f"No valid initial_params available for {name}. Cannot fit distribution.")

        return fitted_pars, fitted_loc, fitted_scale, p0_val


    def cdf(self) -> Series:
        """Return vectorized CDF values for the original data series used to fit."""
        data = self.data
        if self.pars is not None:
            vals = self.dist.cdf(data.values, *self.pars, loc=self.loc, scale=self.scale)
        else:
            vals = self.dist.cdf(data.values, loc=self.loc, scale=self.scale)

        vals = np.asarray(vals, dtype=float)

        if self.prob_zero:
            m = (self.data == 0.0).sum()
            n = len(self.data)

            mask_zero = (data.values == 0.0)
            vals = self.p0 + (1.0 - self.p0) * vals
            vals[mask_zero] = (m + 1.0) / (2.0 * (n + 1.0))  # center of mass for zeros (m + 1.0) / (2.0 * (n + 1.0)) 

        return Series(vals, index=data.index, dtype=float)

    def pdf(self) -> Series:
        data_pdf = self.data.sort_values()
        if self.pars is not None:
            pdf = self.dist.pdf(data_pdf.values, *self.pars, loc=self.loc, scale=self.scale)
        else:
            pdf = self.dist.pdf(data_pdf.values, loc=self.loc, scale=self.scale)

        pdf = np.asarray(pdf, dtype=float)

        if self.prob_zero:
            m = (self.data == 0.0).sum()
            n = len(self.data)

            mask_zero = (data_pdf.values == 0.0)
            pdf = self.p0 + (1.0 - self.p0) * pdf
            pdf[mask_zero] = (m + 1.0) / (2.0 * (n + 1.0))  # center of mass for zeros (m + 1.0) / (2.0 * (n + 1.0)) 

        return Series(pdf, index=data_pdf.index, dtype=float)

    def ppf(self, q: float) -> Series:
        """Return percent point function (inverse CDF) for quantile q as Series."""
        # when prob_zero is used, the continuous part is scaled to (p0,1]
        if self.prob_zero:
            # values below p0 map to zero
            q = np.asarray(q, dtype=float)
            # prepare result array
            out = np.full(self.data.shape, np.nan, dtype=float)
            # indices where q <= p0 -> 0
            mask_zero = q <= self.p0
            out[mask_zero] = 0.0
            # continuous part: map q in (p0,1) to (0,1) for underlying dist
            mask_cont = q > self.p0
            if np.any(mask_cont):
                q_cont = (q[mask_cont] - self.p0) / (1.0 - self.p0)
                if self.pars is not None:
                    out[mask_cont] = self.dist.ppf(q_cont, *self.pars, loc=self.loc, scale=self.scale)
                else:
                    out[mask_cont] = self.dist.ppf(q_cont, loc=self.loc, scale=self.scale)
            out = np.clip(out, -3, 3)  # restrict extreme values to -3 to 3
            return Series(out, index=self.data.index, dtype=float)

        else:
            if self.pars is not None:
                ppf_vals = self.dist.ppf(q, *self.pars, loc=self.loc, scale=self.scale)
            else:
                ppf_vals = self.dist.ppf(q, loc=self.loc, scale=self.scale)
            ppf_vals = np.clip(ppf_vals, -3, 3)  # restrict extreme values to -3 to 3
            return Series(ppf_vals, index=self.data.index, dtype=float)

    def ks_test(
        self,
        method: Literal["auto", "exact", "approx", "asymp"] = "auto",
    ) -> float:
        args = (
            (self.pars, self.loc, self.scale) if self.pars is not None else (self.loc, self.scale)
        )
        # perform KS on positive data only (consistent with fitting) Stagge et al. (2015)
        arr = np.asarray(self.data.dropna(), dtype=float)
        arr = arr[arr > 0]
        if arr.size == 0:
            return float('nan')
        kstest_result = kstest(rvs=arr, cdf=self.dist.name, args=args, method=method)
        return kstest_result.pvalue

    def aic(self) -> dict:
        """
        Compute both log-likelihood-based and MSE-based Akaike Information Criterion (AIC)
        for the fitted distribution.

        Returns
        -------
        dict
            {
                'AIC_loglik': float,
                'AIC_mse': float
            }

        Notes
        -----
        - AIC(loglik) = -2 ln(L) + 2m   (Akaike, 1974; Stagge et al. 2015; Lee et al. 2023)
        - AIC(MSE)    = n ln(MSE) + 2m  (Zhao et al. 2020)
        - Supports zero-inflated data when prob_zero=True.
        """

        data = np.asarray(self.data.dropna())
        n = len(data)

        if n < 3:
            return {'AIC_loglik': np.nan, 'AIC_mse': np.nan}

        # Number of estimated parameters (shape(s) + loc + scale)
        if self.pars is not None:
            m = len(self.pars) + 2
            args = (*self.pars, self.loc, self.scale)
        else:
            m = 2
            args = (self.loc, self.scale)

        # Initialize output values
        aic_loglik = np.nan
        aic_mse = np.nan

        # AIC based on log-likelihood (classical form)
        try:
            if self.prob_zero:
                x_zero = data == 0.0
                x_pos = data > 0

                ll_zero = np.sum(np.log(np.clip(self.p0, 1e-12, 1.0))) if np.any(x_zero) else 0.0
                pdf_pos = self.dist.pdf(data[x_pos], *args)
                ll_pos = np.sum(np.log(np.clip((1 - self.p0) * pdf_pos, 1e-12, None))) if np.any(x_pos) else 0.0
                loglik = ll_zero + ll_pos
            else:
                pdf_vals = self.dist.pdf(data, *args)
                loglik = np.sum(np.log(np.clip(pdf_vals, 1e-12, None)))

            aic_loglik = -2 * loglik + 2 * m

        except Exception as e:
            logger.warning(f"AIC(loglik) failed for {self.dist.name}: {e}")

        # AIC based on MSE (Zhao et al. 2020)
        try:
            sorted_data = np.sort(data)
            n_eff = len(sorted_data)
            F_emp = np.arange(1, n_eff + 1) / (n_eff + 1)

            # Fitted CDF
            if self.pars is not None:
                F_fit = self.dist.cdf(sorted_data, *self.pars, loc=self.loc, scale=self.scale)
            else:
                F_fit = self.dist.cdf(sorted_data, loc=self.loc, scale=self.scale)

            # Apply zero-inflation correction
            if self.prob_zero:
                F_fit = self.p0 + (1.0 - self.p0) * F_fit

            # Compute mean squared error between empirical and fitted CDFs
            mse = np.mean((F_emp - F_fit) ** 2)
            mse = np.clip(mse, 1e-12, None)

            # AIC(MSE) = n ln(MSE) + 2m
            aic_mse = n * np.log(mse) + 2 * m

        except Exception as e:
            logger.warning(f"AIC(MSE) failed for {self.dist.name}: {e}")

        # Return both AIC values as a dictionary
        return {'AIC_loglik': aic_loglik, 'AIC_mse': aic_mse}
