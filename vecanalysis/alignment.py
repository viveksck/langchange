import numpy as np
from vecanalysis.representations.embedding import Embedding

""" Some methods for aligning embeddings spaces """

def intersection_align(embed1, embed2):
    """ 
        Get the intersection of two embeddings.
        Returns embeddings with common vocabulary and indices.
    """
    common_vocab = filter(set(embed1._vocab).__contains__, embed2._vocab) 
    new_vecs1 = np.empty((len(common_vocab), embed1._vecs.shape[1]))
    new_vecs2 = np.empty((len(common_vocab), embed2._vecs.shape[1]))
    for i in xrange(len(common_vocab)):
        new_vecs1[i] = embed1._vecs[embed1._w2v[common_vocab[i]]]
        new_vecs2[i] = embed2._vecs[embed2._w2v[common_vocab[i]]]
    return Embedding(new_vecs1, common_vocab), Embedding(new_vecs2, common_vocab)
    
def procrustes_align(base_embed, other_embed):
    """ 
        Align other embedding to base embeddings via Procrustes.
        Returns best distance-preserving aligned version of other_embed
        NOTE: Assumes indices are aligned
    """
    base_vecs = base_embed._vecs
    other_vecs = other_embed._vecs
    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    fixed_vecs = other_vecs.dot(ortho)
    return Embedding(fixed_vecs, other_embed._vocab)

def linear_align(base_embed, other_embed):
    """
        Align other embedding to base embedding using best linear transform.
        NOTE: Assumes indices are aligned
    """
    base_vecs = base_embed._vecs
    other_vecs = other_embed._vecs
    fixed_vecs = other_vecs.dot(np.pinv(other_vecs)).dot(base_vecs)
    return Embedding(fixed_vecs, other_embed.vocab)



