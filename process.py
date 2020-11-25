import pickle
import numpy as np
from scipy import stats
import plot


def load_results():
    with open('results_0.05inc_10rep.p', 'rb') as f:
        return pickle.load(f)


def padded_results(results):
    pd_results = {}
    for key in results:
        histories = results[key]
        max_len = max(map(len, histories))
        for history in histories:
            for _ in range(max_len - len(history)):
                history.append(history[-1])
        pd_results[key] = histories
    return pd_results


def average_histories(histories):
    max_len = max(map(len, histories))
    for history in histories:
        for _ in range(max_len - len(history)):
            history.append(history[-1])
    histories = np.asarray(histories)
    average_history = np.sum(histories, axis=0) / len(histories)
    return average_history


def get_averaged_results(results):
    averaged_results = {}
    for key in results:
        averaged_results[key] = average_histories(results[key]).tolist()
    return averaged_results


def load_averaged_results():
    with open('results_0.05inc_10rep_avg.p', 'rb') as f:
        return pickle.load(f)


def get_infection_rates(results):
    results = padded_results(results)
    infection_rates = {}
    for key in results:
        histories = np.array(results[key])
        infection_rates_unaveraged = histories[:, -1][:, 2] / 1000
        infection_rates[key] = infection_rates_unaveraged
    return infection_rates


def get_average_infection_rates(avg_results):
    avg_infection_rates = {}
    for key in avg_results:
        avg_infection_rates[key] = avg_results[key][-1][2] / 1000
    return avg_infection_rates


def sample_variance(sample):
    return sum((sample - (sample / len(sample))) ** 2) / (len(sample) - 1)


def sample_average(sample):
    return sum(sample) / len(sample)


def margin_of_error(sample, confidence=0.9):
    var_sample = sample_variance(sample)
    p_val = (1 - confidence) / 2
    return (var_sample / len(sample)) ** 0.5 * stats.t.isf(p_val, len(sample) - 1)


def load_90_results():
    with open('90ec_100rep.p', 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    results_90 = load_90_results()
    avg_results_90 = get_averaged_results(results_90)
    for key in avg_results_90:
        plot.plot_history(avg_results_90[key], str(key))

