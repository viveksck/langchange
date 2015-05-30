import numpy as np
from scipy.stats import ranksums, ttest_ind

from googlengram import util

def get_word_means(words_time_series, words):
    word_means = {}
    for word in words:
        if word not in words_time_series:
            continue
        word_mean = np.array(words_time_series[word].values()).mean()
        if np.isnan(word_mean):
            continue
        word_means[word] = word_mean 
    return word_means

def _make_series_mat(words_time_series, words, one_minus=True, start_year=1900, end_year=2000):
    series_list = []
    for word in words:
        if word not in words_time_series:
            continue
        word_array = np.array([value for year, value in words_time_series[word].items() if year >= start_year and year <=end_year])
        if one_minus:
            word_array = 1 - word_array
        if np.isnan(word_array.sum()):
            continue
        series_list.append(word_array)
    series_mat = np.array(series_list)
    return series_mat

def get_series_mean_conf(words_time_series, words, one_minus=False, start_year=1900, end_year=2000):
    series_mat = _make_series_mat(words_time_series, words, one_minus=one_minus, start_year=start_year, end_year=end_year)
    means = series_mat.mean(0)
    return means, series_mat.std(0) / np.sqrt(len(means))

def get_power_series(word_degree_series, words, year=1999):
    series_mat = _make_series_mat(word_degree_series, words, one_minus=False, start_year=year, end_year=year)
    degs = []
    probs = []
    sum = series_mat.sum()
    for i in range(1, series_mat.max() + 1):
        count = (series_mat == i).sum()
        if count != 0:
            probs.append(float(count) / float(sum))
            degs.append(i)
    return degs, probs

def p_value_series(words_time_series, word_set1, word_set2,  one_minus=True):
    series_mat1 = _make_series_mat(words_time_series, word_set1, one_minus=True)
    series_mat2 = _make_series_mat(words_time_series, word_set2, one_minus=True)
    assert series_mat1.shape[1] == series_mat2.shape[1]
    p_series = np.empty(series_mat1.shape[1])
    for timepoint in xrange(series_mat1.shape[1]):
        _, p_series[timepoint] = ttest_ind(series_mat1[timepoint,:], series_mat2[timepoint,:], equal_var=False)
    return p_series

def series_mean_ranksums(word_set1, word_set2, words_time_series, one_minus=True):
    word_means1 = np.array(get_word_means(words_time_series, word_set1).values())
    word_means2 = np.array(get_word_means(words_time_series, word_set2).values())
    if one_minus:
        word_means1 = 1 - word_means1
        word_means2 = 1 - word_means2
    z,p = ranksums(word_means1, word_means2)
    return {"z" : z, 
            "p" : p, 
            "set1_size" : len(word_means1), 
            "set2_size" : len(word_means2), 
            "set1_med" : np.median(word_means1),
            "set2_med" : np.median(word_means2)}

if __name__ == '__main__':
    DATA_DIR = "/dfs/scratch0/google_ngrams/info/"
    abstract_words = util.load_word_list(DATA_DIR + "abstractwords.txt")
    concrete_words = util.load_word_list(DATA_DIR + "concretewords.txt")
    words_times = util.load_pickle(DATA_DIR + "ppmi_cosine_f1900-vols.pkl")
    print series_mean_ranksums(abstract_words, concrete_words, words_times)
    
