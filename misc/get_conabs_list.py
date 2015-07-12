import numpy as np
import cPickle as pickle

INPUT_FILE = "/dfs/scratch0/googlengrams/2012-eng-fic/info/commonnonstop-1900-2000-8-6.pkl"
in_fp = open(INPUT_FILE, "rb") 
words = pickle.load(in_fp)
infp = open("concreteness_ratings.csv")
infp.readline()
concrete = []
abstract = []
ABS_THRESH = 25
CON_THRESH = 75
scores = {}
for line in infp:
    line_info = line.split(",")
    scores[line_info[0]] = float(line_info[2])

abs_thresh = np.percentile(scores.values(), 25)
con_thresh = np.percentile(scores.values(), 75)
consfp = open("concretewords.txt", "w")
absfp = open("abstractwords.txt", "w")
for word in scores:
    if word not in words:
        continue
    if scores[word] < abs_thresh:
        absfp.write(word + "\n")
    elif scores[word] > con_thresh:
        consfp.write(word + "\n")
