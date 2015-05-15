from embedding import SVDEmbedding, EnsembleEmbedding, Embedding
from explicit import Explicit


def create_representation(args):
    rep_type = args['<representation>']
    path = args['<representation_path>']
    w_c = args['--w+c']
    eig = float(args['--eig'])
    
    if rep_type == 'PPMI':
        if w_c:
            raise Exception('w+c is not implemented for PPMI.')
        else:
            return Explicit.load(path, True)
        
    elif rep_type == 'SVD':
        if w_c:
            return EnsembleEmbedding(SVDEmbedding(path, False, eig, False), SVDEmbedding(path, False, eig, True), True)
        else:
            return SVDEmbedding(path, True, eig)
        
    else:
        if w_c:
            return EnsembleEmbedding(Embedding.load(path + '.words', False), Embedding.load(path + '.contexts', False), True)
        else:
            return Embedding.load(path, True)
