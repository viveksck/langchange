import numpy as np
from vecanalysis.representations.embedding import Embedding

""" Some methods for aligning embeddings spaces """

def explicit_intersection_align(embed1, embed2):
    common_vocab = filter(set(embed1.iw).__contains__, embed2.iw) 
    return embed1.get_subembed(common_vocab), embed2.get_subembed(common_vocab)
    
def intersection_align(embed1, embed2):
    """ 
        Get the intersection of two embeddings.
        Returns embeddings with common vocabulary and indices.
    """
    common_vocab = filter(set(embed1.iw).__contains__, embed2.iw) 
    newvecs1 = np.empty((len(common_vocab), embed1.m.shape[1]))
    newvecs2 = np.empty((len(common_vocab), embed2.m.shape[1]))
    for i in xrange(len(common_vocab)):
        newvecs1[i] = embed1.m[embed1.wi[common_vocab[i]]]
        newvecs2[i] = embed2.m[embed2.wi[common_vocab[i]]]
    return Embedding(newvecs1, common_vocab), Embedding(newvecs2, common_vocab)

def smart_procrustes_align(base_embed, other_embed, post_normalize=False):
    in_base_embed, in_other_embed = intersection_align(base_embed, other_embed)
    base_vecs = in_base_embed.m
    other_vecs = in_other_embed.m
    m = other_vecs.T.dot(base_vecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    return Embedding((other_embed.m).dot(ortho), other_embed.iw, normalize = post_normalize)

def procrustes_align(base_embed, other_embed):
    """ 
        Align other embedding to base embeddings via Procrustes.
        Returns best distance-preserving aligned version of other_embed
        NOTE: Assumes indices are aligned
    """
    basevecs = base_embed.m - base_embed.m.mean(0)
    othervecs = other_embed.m - other_embed.m.mean(0)
    m = othervecs.T.dot(basevecs)
    u, _, v = np.linalg.svd(m) 
    ortho = u.dot(v)
    fixedvecs = othervecs.dot(ortho)
    return Embedding(fixedvecs, other_embed.iw)

def linear_align(base_embed, other_embed):
    """
        Align other embedding to base embedding using best linear transform.
        NOTE: Assumes indices are aligned
    """
    basevecs = base_embed.m
    othervecs = other_embed.m
    fixedvecs = othervecs.dot(np.linalg.pinv(othervecs)).dot(basevecs)
    return Embedding(fixedvecs, other_embed.iw)



