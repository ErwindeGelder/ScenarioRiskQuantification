""" Code for performing the case study for the risk quantification paper.

Creation date: 2021 02 03
Author(s): Erwin de Gelder

Modifications:
"""

import os
from typing import Callable, List, NamedTuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from simulation import SimulationString, KDE, kde_from_file


# Case study options list
CaseStudy = NamedTuple("CaseStudy", [("name", str),
                                     ("n", int),
                                     ("parameters", List[str]),
                                     ("default_parameters", dict),
                                     ("percentile", int),
                                     ("simulator", SimulationString),
                                     ("func_sample", Callable),
                                     ("func_density", Callable),
                                     ("func_validity_check", Callable),
                                     ("func_process_result", Callable)])


def generate_parms(options, func_sample):
    np.random.seed(0)
    pars = np.zeros((options.n, len(options.parameters)))
    tries = np.zeros(options.n)
    i = 0
    while i < options.n:
        pars[i, :] = func_sample()
        tries[i] += 1
        if options.func_validity_check(pars[i, :]):
            i += 1
    df = pd.DataFrame(data=pars, columns=options.parameters)
    df["tries"] = tries
    return df, pars


def do_simulations(df, pars, options):
    result = np.zeros(options.n)
    kpi = np.zeros(options.n)
    for i, par in enumerate(tqdm(pars)):
        par_dict = dict(zip(options.parameters, par))
        par_dict.update(options.default_parameters)
        result[i] = options.simulator.simulation(par_dict)
        kpi[i] = options.func_process_result(result[i])
    df["result"] = result
    df["kpi"] = kpi


def monte_carlo(options, overwrite=False):
    folder = os.path.join("data", "simulation_results")
    filename_df = os.path.join(folder, "{:s}_mc.csv".format(options.name))
    if os.path.exists(filename_df) and not overwrite:
        return pd.read_csv(filename_df, index_col=0)

    df, pars = generate_parms(options, options.func_sample)
    do_simulations(df, pars, options)
    if not os.path.exists(folder):
        os.mkdir(folder)
    df.to_csv(filename_df)
    return df


def mc_result(df_mc):
    print("Monte Carlo:")
    collision = (df_mc["result"] < 0).astype(np.float)
    prob = np.mean(collision)
    sigma = np.sqrt(np.sum((collision - prob)**2)) / len(df_mc)
    print("  Probability of collision: {:.2e} +/- {:.2e}".format(prob, sigma))
    prob = np.mean(df_mc["kpi"])
    sigma = np.sqrt(np.sum((df_mc["kpi"] - prob)**2)) / len(df_mc)
    print("  Probability of injury:    {:.2e} +/- {:.2e}".format(prob, sigma))


def create_is(df_mc, options, overwrite=False):
    folder = os.path.join("data", "kde")
    filename = os.path.join(folder, "{:s}_is.p".format(options.name))
    if os.path.exists(filename) and not overwrite:
        return kde_from_file(filename)

    df_sorted = df_mc.sort_values("result")
    df_is = df_sorted.iloc[:options.n*options.percentile//100]
    data = df_is[options.parameters].values
    kde = KDE(data)
    kde.compute_bandwidth()
    kde.pickle(filename)
    return kde


def importance_sampling(kde, options, overwrite=False):
    folder = os.path.join("data", "simulation_results")
    filename_df = os.path.join(folder, "{:s}_is.csv".format(options.name))
    if os.path.exists(filename_df) and not overwrite:
        return pd.read_csv(filename_df, index_col=0)

    # Generate the parameters.
    df, pars = generate_parms(options, kde.sample)
    df["density_orig"] = options.func_density(pars)
    df["density_is"] = kde.score_samples(pars)
    do_simulations(df, pars, options)

    # Write to file
    df.to_csv(filename_df)
    return df


def is_result(df_is, df_mc):
    print("Importance sampling:")
    values = (df_is["result"] < 0) * df_is["density_orig"] / df_is["density_is"]
    values *= (np.sum(df_mc["tries"]) / len(df_mc)) / (np.sum(df_is["tries"]) / len(df_is))
    prob = np.mean(values)
    sigma = np.sqrt(np.sum((values - prob)**2)) / len(values)
    print("  Probability of collision: {:.2e} +/- {:.2e}".format(prob, sigma))
    values = df_is["kpi"] * df_is["density_orig"] / df_is["density_is"]
    values *= (np.sum(df_mc["tries"]) / len(df_mc)) / (np.sum(df_is["tries"]) / len(df_is))
    prob = np.mean(values)
    sigma = np.sqrt(np.sum((values - prob)**2)) / len(values)
    print("  Probability of injury:    {:.2e} +/- {:.2e}".format(prob, sigma))


def case_study(options, overwrite=False):
    df_mc = monte_carlo(options, overwrite=overwrite)
    mc_result(df_mc)
    kde_is = create_is(df_mc, options, overwrite=overwrite)
    df_is = importance_sampling(kde_is, options, overwrite=overwrite)
    is_result(df_is, df_mc)
    return df_mc, df_is


MU_REACTIONTIME = np.log(.92**2/np.sqrt(.92**2+.28**2))
SIGMA_REACTIONTIME = np.sqrt(np.log(1+.28**2/.92**2))


def sample_reactiontime():
    return np.random.lognormal(MU_REACTIONTIME, SIGMA_REACTIONTIME)


def reactiontime_density(reactiontime):
    return (1/(reactiontime*SIGMA_REACTIONTIME*np.sqrt(2*np.pi))*
            np.exp(-(np.log(reactiontime)-MU_REACTIONTIME)**2/(2*SIGMA_REACTIONTIME**2)))


def get_kpi(result):
    if result < 0:
        speed_change_impact = -result/2
        return 1 / (1 + np.exp(-(-6.068 + 0.1*speed_change_impact - 0.6234)))
    return 0
