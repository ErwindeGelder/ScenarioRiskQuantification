""" Functionalities to estimate the uncertainty of the estimated risk

Creation date: 2022 07 01
Author(s): Erwin de Gelder
"""

import os
import pickle
from typing import Callable, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from domain_model import DocumentManagement
from simulation import KDE, Simulator


OVERWRITE = False
FOLDER = os.path.join("data", "uncertainty_results")
N_HOUR = 63
N_HOURS = np.logspace(np.log10(18), np.log10(63), 29)  # 30 or more goes wrong
N_BOOTSTRAP = 1000
N_MC = 10000
N_CRITICAL = N_MC // 50
N_NIS = 10000
N_SIMULATIONS = np.logspace(np.log10(625), np.log10(10000), 57).astype(int)


class Uncertainty:
    """ Class for determining the uncertainty of the estimated risk. """
    def __init__(self, name: str, scenarios: DocumentManagement, kde: KDE, simulator: Simulator,
                 parameters: List, func_valid: Callable):
        self.name = name
        self.scenarios = scenarios
        self.kde = kde
        self.data = self.kde.data * self.kde.data_helpers.std
        self.simulator = simulator
        self.parameters = parameters
        self.func_valid = func_valid

        self.tstarts = self.get_tstarts()

    def get_tstarts(self) -> np.ndarray:
        """ Determine the starting times of the scenarios.

        :return: numpy array of the starting times in seconds.
        """
        tstarts = []
        for key in self.scenarios.collections["scenario"]:
            scenario = self.scenarios.get_item("scenario", key)
            tstarts.append(scenario.get_tstart())
        return np.array(tstarts)

    def exposure_category(self, plot=True) -> Tuple[np.ndarray, np.ndarray]:
        """ Determine estimated exposure (and uncertainty) for varying hours.

        :param plot: whether to make a plot or not.
        :return: numpy arrays of the exposure and its uncertainty.
        """
        means = np.zeros(len(N_HOURS))
        sigmas = np.zeros_like(means)
        for i, nhour in enumerate(N_HOURS):
            tstarts_sub = [x for x in self.tstarts if x <= nhour*3600]
            _, counts = np.unique(np.floor(np.array(tstarts_sub)/3600), return_counts=True)
            if len(counts) < nhour:
                counts = np.concatenate((counts, np.zeros(np.ceil(nhour).astype(int)-len(counts))))
            means[i] = np.mean(counts)
            sigmas[i] = np.std(counts)/np.sqrt(nhour)

        if plot:
            plt.subplots()
            plt.plot(N_HOURS, means, lw=5, label="Expectation")
            plt.fill_between(N_HOURS, means-sigmas, means+sigmas, label="Standard deviation",
                             color=(1, .7, .7))
            plt.xlabel("Hours of data [h]")
            plt.ylabel("Exposure")
            plt.legend()
            plt.xlim(N_HOURS[0], N_HOURS[-1])

        return means, sigmas

    def generate_parms(self, n_parms: int, kde: KDE) -> Tuple[pd.DataFrame, np.ndarray]:
        """ Generate parameters of a scenario using the provided KDE.

        :param n_parms: number of parameters to be generated.
        :param kde: the probability density function to draw the parameters from.
        :return: dataframe of parameters and numpy array of parameters.
        """
        pars = np.zeros((n_parms, len(self.parameters)))
        tries = np.zeros(n_parms)
        i = 0
        while i < n_parms:
            tries[i] += 1
            pars_tmp = kde.sample()
            if not self.func_valid(pars_tmp)[0]:
                continue
            pars[i] = pars_tmp[0]
            i += 1
        dataframe = pd.DataFrame(data=pars, columns=self.parameters)
        dataframe["tries"] = tries
        return dataframe, pars

    def monte_carlo(self, n_hour: float) -> pd.DataFrame:
        """ Perform Monte Carlo simulation.

        :param n_hour: number of hours of data to be used.
        :return: dataframe with the result.
        """
        filename = os.path.join(FOLDER, "{:s}_mc_{:02.0f}.csv".format(self.name, n_hour))
        if os.path.exists(filename) and not OVERWRITE:
            return pd.read_csv(filename, index_col=0)

        subdata = self.data[self.tstarts <= n_hour*3600]
        kde = KDE(subdata)
        kde.compute_bandwidth()
        np.random.seed(0)
        print("Monte Carlo with {:.1f} hours of data.".format(n_hour))
        results = np.zeros(N_MC)
        dataframe, pars = self.generate_parms(N_MC, kde)
        for i, par in enumerate(tqdm(pars)):
            results[i] = self.simulator.simulation(dict(zip(self.parameters, par)))
        dataframe["result"] = results
        dataframe.to_csv(filename)
        return dataframe

    def importance_sampling(self, n_hour: float, df_mc: pd.DataFrame, n_nis: int = N_NIS,
                            n_critical: int = N_CRITICAL):
        """ Do the Monte Carlo importance sampling.

        :param n_hour: number of hours of data to be used.
        :param df_mc: resulting dataframe from Monte Carlo simulation.
        :param n_nis: number of simulations to be performed.
        :param n_critical: number of critical scenarios to be used.
        :return: dataframe with the result.
        """
        filename = os.path.join(FOLDER, "{:s}_nis_{:02.0f}_{:05d}.csv".format(self.name, n_hour,
                                                                              n_nis))
        if os.path.exists(filename) and not OVERWRITE:
            return pd.read_csv(filename, index_col=0)

        # Create importance density
        df_sorted = df_mc.sort_values("result")
        df_sub = df_sorted.iloc[:n_critical]
        data = df_sub[self.parameters].values
        kde = KDE(data)
        kde.compute_bandwidth()

        # Do the importance sampling simulation runs
        print("Importance sampling with {:.1f} hours of data.".format(n_hour))
        dataframe, pars = self.generate_parms(n_nis, kde)
        dataframe["density_is"] = kde.score_samples(pars)
        results = np.zeros(n_nis)
        for i, par in enumerate(tqdm(pars)):
            results[i] = self.simulator.simulation(dict(zip(self.parameters, par)))
        dataframe["result"] = results
        dataframe.to_csv(filename)
        return dataframe

    def is_result(self, df_is: pd.DataFrame, kde: KDE) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate the result using the simulations from importance sampling.

        :param df_is: resulting dataframe from importance sampling.
        :param kde: probability density function of the scenario parameters.
        :return: estimated risk and its uncertainty.
        """
        mc_tries = 0
        i = 0
        while i < N_MC:
            mc_tries += 1
            if self.func_valid(kde.sample())[0]:
                i += 1
        is_tries = np.sum(df_is["tries"])
        weight = len(df_is) / is_tries / N_MC * mc_tries
        values = weight / df_is["density_is"] * (df_is["result"] <= 0)
        values *= kde.score_samples(df_is[self.parameters])
        prob = np.mean(values)
        sigma = np.sqrt(np.sum((values-prob)**2))/len(values)
        return prob, sigma

    def bootstrap_is_result(self, n_hour: float, n_nis: int = N_NIS, n_critical: int = N_CRITICAL,
                            verbose: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Determine risk uncertainty using bootstrapping.

        :param n_hour: number of hours of data to be used.
        :param n_nis: number of simulations to be performed.
        :param n_critical: number of critical scenarios to be used.
        :param verbose: whether to print information or not.
        :return: estimated risk and its uncertainty.
        """
        filename = os.path.join(FOLDER, "{:s}_bootstrap_{:02.0f}_{:05d}.p".format(self.name, n_hour,
                                                                                  n_nis))
        if os.path.exists(filename) and not OVERWRITE:
            with open(filename, "rb") as file:
                probs, sigmas = pickle.load(file)
            if verbose is None:
                verbose = False
        else:
            probs = np.zeros(N_BOOTSTRAP)
            sigmas = np.zeros_like(probs)
            df_mc = self.monte_carlo(n_hour)
            df_is = self.importance_sampling(n_hour, df_mc, n_nis=n_nis, n_critical=n_critical)
            subdata = self.data[self.tstarts <= n_hour*3600]
            for i in tqdm(range(N_BOOTSTRAP)):
                new_data = subdata[np.random.choice(len(subdata), size=len(subdata), replace=True)]
                kde_new = KDE(new_data)
                kde_new.compute_bandwidth()
                probs[i], sigmas[i] = self.is_result(df_is, kde_new)
            with open(filename, "wb") as file:
                pickle.dump((probs, sigmas), file)

            if verbose is None:
                verbose = True

        if verbose:
            print("Bootstrapping {:.1f} h, {:d} sim: {:.4e} +/- {:.4e} (rel.: {:.4f})"
                  .format(n_hour, n_nis, np.mean(probs), np.std(probs),
                          np.std(probs)/np.mean(probs)))
        return probs, sigmas

    def bootstrap_is_result2(self, n_nis: int = N_NIS, n_critical: int = N_CRITICAL,
                             plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """ Do the bootstrapping for varying number of hours.

        :param n_nis: number of simulations to be performed.
        :param n_critical: number of critical scenarios to be used.
        :param plot: whether to make a plot or not.
        :return: estimated risk and its uncertainty.
        """
        means = np.zeros(len(N_HOURS))
        sigmas = np.zeros(len(N_HOURS))
        for i, n_hour in enumerate(N_HOURS):
            probs, _ = self.bootstrap_is_result(n_hour, n_nis=n_nis, n_critical=n_critical)
            means[i] = np.mean(probs)
            sigmas[i] = np.std(probs)

        if plot:
            plt.subplots()
            plt.plot(N_HOURS, means, lw=5, label="Mean bootstrap")
            plt.fill_between(N_HOURS, means-sigmas, means+sigmas, label="Uncertainty",
                             color=(1, .7, .7))
            plt.xlabel("Hours of data [h]")
            plt.ylabel("Probability of a crash")
            plt.legend()
            plt.xlim(N_HOURS[0], N_HOURS[-1])

        return means, sigmas

    def vary_simulations(self, n_hour: float = N_HOUR, plot: bool = True) \
            -> Tuple[np.ndarray, np.ndarray]:
        """ Estimate the risk (uncertainty) for varying number of simulations.

        :param n_hour: number of hours of data to be used.
        :param plot: whether to make a plot or not.
        :return: estimated risk and its uncertainty.
        """
        filename = os.path.join(FOLDER, "{:s}_vary_simulations.p".format(self.name))
        if os.path.exists(filename) and not OVERWRITE:
            with open(filename, "rb") as file:
                means, sigmas = pickle.load(file)
        else:
            means = np.zeros(len(N_SIMULATIONS))
            sigmas = np.zeros(len(N_SIMULATIONS))
            df_mc = self.monte_carlo(n_hour)
            kde = KDE(self.data[self.tstarts < n_hour*3600])
            kde.compute_bandwidth()
            df_is = self.importance_sampling(n_hour, df_mc)
            for i, n_simulation in enumerate(N_SIMULATIONS):
                means[i], sigmas[i] = self.is_result(df_is.iloc[:n_simulation], kde)
            with open(filename, "wb") as file:
                pickle.dump((means, sigmas), file)

        if plot:
            plt.subplots()
            plt.plot(N_SIMULATIONS, means, lw=5, label="Mean")
            plt.fill_between(N_SIMULATIONS, means-sigmas, means+sigmas, label="Uncertainty",
                             color=(1, .7, .7))
            plt.xlabel("Number of simulations")
            plt.ylabel("Probability of a crash")
            plt.legend()
            plt.xlim(N_SIMULATIONS[0], N_SIMULATIONS[-1])

        return means, sigmas

    def total_variance_hours(self, plot=True, return_terms=False) -> \
            Union[Tuple[np.ndarray, np.ndarray],
                  Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """ Estimate the risk (uncertainty) considering all uncertainty components.

        :param plot: whether to make a plot or not.
        :param return_terms: whether to return each of the three variance terms as well.
        :return: risk, its uncertainty, and (optionally) the three variance terms.
        """
        filename = os.path.join(FOLDER, "{:s}_total_variance_hours.p".format(self.name))
        if os.path.exists(filename) and not OVERWRITE:
            with open(filename, "rb") as file:
                probs, sigmas, term1, term2, term3 = pickle.load(file)
        else:
            mu_exposure, sigma_exposure = self.exposure_category(plot=False)
            sigma_nis_data = self.bootstrap_is_result2(plot=False)[1]
            mu_nis = np.zeros(len(N_HOURS))
            sigma_nis_simulation = np.zeros(len(N_HOURS))
            for i, n_hour in enumerate(N_HOURS):
                df_mc = self.monte_carlo(n_hour)
                df_is = self.importance_sampling(n_hour, df_mc)
                kde = KDE(self.data[self.tstarts < n_hour*3600])
                kde.compute_bandwidth()
                mu_nis[i], sigma_nis_simulation[i] = self.is_result(df_is, kde)
            variance_nis = sigma_nis_simulation**2 + sigma_nis_data**2
            probs = mu_exposure * mu_nis
            term1 = mu_exposure**2 * variance_nis
            term2 = mu_nis**2 * sigma_exposure**2
            term3 = sigma_exposure**2 * variance_nis
            sigmas = np.sqrt(term1 + term2 + term3)
            with open(filename, "wb") as file:
                pickle.dump((probs, sigmas, term1, term2, term3), file)

        if plot:
            plt.subplots()
            plt.plot(N_HOURS, probs, lw=5, label="Risk")
            plt.fill_between(N_HOURS, probs-sigmas, probs+sigmas, label="Uncertainty",
                             color=(1, .7, .7))
            plt.xlabel("Number of hours")
            plt.ylabel("Risk [h$^-1$]")
            plt.legend()
            plt.xlim(N_HOURS[0], N_HOURS[-1])

        if return_terms:
            return probs, sigmas, term1, term2, term3
        return probs, sigmas
