from sklearn.neighbors import LSHForest

class EmbeddingNetworkBuilder:
    """ Basically a wrapper around sklearns LSH forest """

    def __init__(self, lsh_init=None):
        if lsh_init == None:
            self._lsh_forest = LSHForest(n_estimators=25, n_candidates=1000)
        else:
            self._lsh_forest = lsh_init 
        self.iw = None
        self.m = None

    def fit_lsh_forest(self, embedding):
        self._lsh_forest.fit(embedding.m)
        self._embedding = embedding

    def extract_nn_network(self, nn=20):
        dir_graph_mat = self._lsh_forest.kneighbors_graph(X=self._embedding.m, n_neighbors=nn+1)
        return dir_graph_mat

    def make_undirected(self, dir_graph_mat):
        nodes = set(range(dir_graph_mat.shape[0]))
        edges = set([])
        for node_i in dir_graph_mat.shape[0]:
            for node_j in dir_graph_mat[node_i].nonzero()[1]:
                edges.add((node_i, node_j))
        return nodes, edges

    def get_forest(self):
        return self._lsh_forest
    
    def get_node_to_word(self):
        return self.iw


