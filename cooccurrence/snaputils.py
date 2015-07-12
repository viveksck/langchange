import snap

import scipy.stats as stat
from wilsonconf import check_conf

def make_snap_graph(indices, coo_mat):
    graph = snap.TUNGraph.New()
    for entry in indices:
        graph.AddNode(entry)
    for i in xrange(len(coo_mat.row)):
        if (graph.IsNode(int(coo_mat.row[i])) and graph.IsNode(int(coo_mat.col[i])) 
                and coo_mat.col[i] != coo_mat.row[i] and coo_mat.data[i] > 0):
            graph.AddEdge(int(coo_mat.row[i]), int(coo_mat.col[i]))
    return graph

def make_conf_graph(indices, coo_mat, alpha, eff_sample_size, strict=True):
    coo_mat = coo_mat / coo_mat.sum()
    csr_mat = coo_mat.tocsr()
    graph = snap.TUNGraph.New()
    for entry in indices:
        graph.AddNode(entry)
    z = stat.norm.ppf(1 - alpha / 2.0)
    print "Going through", len(coo_mat.row), "entries"
    for i in xrange(len(coo_mat.row)):
        if i % 10000 == 0:
            print "Processed entry", i
        if (graph.IsNode(int(coo_mat.row[i])) and graph.IsNode(int(coo_mat.col[i])) 
                and coo_mat.col[i] != coo_mat.row[i] and coo_mat.data[i] > 0):
            freq_one = csr_mat[coo_mat.row[i], :].sum()
            freq_two = csr_mat[coo_mat.col[i], :].sum()
            if check_conf(coo_mat.data[i], freq_one, freq_two, z, z**2.0, eff_sample_size, strict=strict):
                graph.AddEdge(int(coo_mat.row[i]), int(coo_mat.col[i]))
    return graph
