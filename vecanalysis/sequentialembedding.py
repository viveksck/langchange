import collections
import random

from sklearn.decomposition import PCA

from vecanalysis.representations.embedding import Embedding
from vecanalysis.dimreduce import reduce_dim
from vecanalysis.alignment import smart_procrustes_align

INPUT_PATH = "/dfs/scratch0/googlengrams/archive/sglove-vecs-smallrel-aligned-seq/{year}-300vecs"

class SequentialEmbedding:

    def __init__(self, years, top_freq=None):
        self.embeds = collections.OrderedDict()
        for year in years:
            self.embeds[year] = Embedding.load(INPUT_PATH.format(year=year))

    @classmethod
    def from_ordered_dict(cls, year_embeds):
        new_seq_embeds = cls([])
        new_seq_embeds.embeds = year_embeds
        return new_seq_embeds

    def get_embed(self, year):
        return self.embeds[year]

    def get_seq_neighbour_set(self, word, n=3):
        neighbour_set = set([])
        for embed in self.embeds.itervalues():
            closest = embed.closest(word, n=n)
            for _, neighbour in closest:
                neighbour_set.add(neighbour)
        return neighbour_set

    def get_word_subembeds(self, word, n=3, num_rand=None, word_list=None):
        if word_list == None:
            word_set = self.get_seq_neighbour_set(word, n=n)
            if num_rand != None:
                word_set = word_set.union(set(random.sample(self.embeds.values()[-1].iw, num_rand)))
            word_list = list(word_set)
        year_subembeds = collections.OrderedDict()
        for year,embed in self.embeds.iteritems():
            year_subembeds[year] = embed.get_subembed(word_list)
        return SequentialEmbedding.from_ordered_dict(year_subembeds)

    def get_dim_reduced(self, dim=2, normalize=False):
        year_reduced_embeds = collections.OrderedDict()
        for year,embed in self.embeds.iteritems():
            year_reduced_embeds[year] = reduce_dim(embed, dim=2, post_normalize=False)
        return SequentialEmbedding.from_ordered_dict(year_reduced_embeds)

    def get_dim_reduced_avg(self, dim=2, normalize=False):
        first_iter = True
        for year,embed in self.embeds.iteritems():
            if first_iter:
                align_words = set(embed.iw)
                first_iter = False
            else:
                align_words = align_words.intersection(set(embed.iw))
        first_iter = True 
        for year,embed in self.embeds.iteritems():
            if first_iter:
                avg_mat = embed.get_subembed(list(align_words)).m
                first_iter = False
            else:
                avg_mat += embed.get_subembed(list(align_words)).m
        avg_mat /= float(len(self.embeds))
        pca = PCA(n_components=2)
        pca.fit(avg_mat)
        year_reduced_embeds = collections.OrderedDict()
        for year,embed in self.embeds.iteritems():
            proj_embed_mat = pca.transform(embed.m)
            year_reduced_embeds[year] = Embedding(proj_embed_mat, embed.iw, normalize=normalize) 
        return SequentialEmbedding.from_ordered_dict(year_reduced_embeds)

    def get_dim_reduced_fixed(self, dim=2, normalize=False, basis=2000):
        pca = PCA(n_components=2)
        pca.fit(self.embeds[basis].m)
        year_reduced_embeds = collections.OrderedDict()
        for year,embed in self.embeds.iteritems():
            proj_embed_mat = pca.transform(embed.m)
            year_reduced_embeds[year] = Embedding(proj_embed_mat, embed.iw, normalize=normalize) 
        return SequentialEmbedding.from_ordered_dict(year_reduced_embeds)

    def get_word_path(self, word, n=3, dim=2, num_rand=None, word_list=None, basis_year=2000):
       subembeds = self.get_word_subembeds(word, n=n, num_rand=num_rand, word_list=word_list) 
       pca = PCA(n_components=dim)
       basis = pca.fit_transform(subembeds.embeds[basis_year].m)
       basis = Embedding(basis, subembeds.embeds[basis_year].iw, normalize=False)
       word_path = collections.OrderedDict()
       for year, embed in self.embeds.iteritems():
           word_path[year] = pca.transform(embed.represent(word)).flatten()
       return word_path, basis

    def get_word_paths(self, word_list):
        word_paths = collections.defaultdict(collections.OrderedDict)
        for year, embed in self.embeds.iteritems():
            for word in word_list:
                word_paths[word][year] = embed.represent(word)
        return word_paths
    
    def get_aligned(self, normalize=False):
        year_aligned_embeds = collections.OrderedDict()
        first_iter = True
        base_embed = None
        for year,embed in self.embeds.iteritems():
            if first_iter:
                year_aligned_embeds[year] = embed
                first_iter = False
            else:
                year_aligned_embeds[year] = smart_procrustes_align(base_embed, embed, post_normalize=False)
            base_embed = year_aligned_embeds[year]

        return SequentialEmbedding.from_ordered_dict(year_aligned_embeds)

    def get_reduced_word_subembeds(self, word, n=3, dim=2, num_rand=None, basis=2000, word_list=None):
        return self.get_word_subembeds(word, n=n, num_rand=num_rand, word_list=word_list).get_dim_reduced_fixed(dim=dim, basis=basis).get_aligned()
