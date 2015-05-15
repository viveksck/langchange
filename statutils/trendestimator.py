import operator
import numpy as np
import scipy as sp
import statsmodels.api as sm
import collections

FREQ_FILE = "/dfs/scratch0/google_ngrams/info/relevantwords.txt"

def trend_estimate(year_series, start_year=1850, end_year=2000):
    y = np.array([val for year, val in sorted(year_series.items(), key=operator.itemgetter(0)) if year >= start_year and year <= end_year and val != -1])
    zeros = np.where(y==0)[0]
    if len(zeros) == 0:
        start = 0
    else:
        start = zeros[-1] + 1
    y = y[start:]
    if len(y) <= 1:
        return None
    X = np.arange(len(y))
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res

def get_trend_estimates(word_series, start_year=1850, end_year=2000):
    res_series = {}
    for word, year_series in word_series.iteritems():
        trend_est = trend_estimate(year_series, start_year=start_year, end_year=end_year)
        if not trend_est == None:
            res_series[word] = trend_est
    return res_series

def process_trend_estimates(trend_estimates, freq_file=FREQ_FILE, p_value_thresh=0.001, slope_thresh = 10**(-5)):
    freq_dict = {}
    try:
        freq_fp = open(freq_file, "r")
        for line in freq_fp:
            word = line.strip().split()[0]
            if word in trend_estimates:
                freq_dict[word] = len(freq_dict)
    except IOError:
        freq_dict = collections.defaultdict(int)
        pass
    narrowing = {}
    broadening = {}
    nochange = {}
    for word, res in trend_estimates.iteritems():
        if res.nobs < 20 or word not in freq_dict or len(word) == 1:
            continue
        word_info = {"slope" : res.params[1], 
                "intercept" : res.params[0],
                "r2" : res.rsquared,
                "pvalue" : res.pvalues[1],
                "fpvalue" : res.f_pvalue,
                "freq_rank" : freq_dict[word]}
        if word_info["pvalue"] < p_value_thresh and abs(word_info["slope"]) > slope_thresh:
            if word_info["slope"] > 0:
                narrowing[word] = word_info
            else:
                broadening[word] = word_info
        else:
            nochange[word] = word_info

    return {"broadened" : broadening, "narrowed" : narrowing, "nochange" : nochange}

def prune_trend_infos(trend_infos, stat, descending, thresh):
    if stat == None:
        return trend_infos
    if descending:
        mod = 1
    else:
        mod = -1
    return [(word, word_info) for (word, word_info) in trend_infos if word_info[stat]*mod > mod*thresh] 

def sort_trend_infos(trend_infos, stat, descending=True, freq_rank_thresh=10000, r2_thresh=0.5):
    pruned = prune_trend_infos(trend_infos.items(), "freq_rank", False, freq_rank_thresh)
    pruned = prune_trend_infos(pruned, "r2", True, r2_thresh)
    return sorted(pruned, key= lambda (word, word_info) : word_info[stat], reverse=descending)

def get_densefreq_corr(density_trends_p, freq_trends_p, p_value_thresh = 0.001):
    a = []
    b = []
    density_trends = {}
    freq_trends = {}
    for word in density_trends_p:
        if word in freq_trends_p:
            density_trends[word] = density_trends_p[word]
            freq_trends[word] = freq_trends_p[word]
    get_sig_slope = lambda info : info.params[1] if info.pvalues[1] < p_value_thresh else 0
    for word in density_trends.keys(): 
        if density_trends[word].nobs < 20 or freq_trends[word].nobs < 20:
            continue
        a.append(-1 * get_sig_slope(density_trends[word]))
        b.append(get_sig_slope(freq_trends[word]))
    return {"spearman" : sp.stats.spearmanr(a, b), "kendall" : sp.stats.kendalltau(a, b), "pearson" : sp.stats.pearsonr(a, b)}

