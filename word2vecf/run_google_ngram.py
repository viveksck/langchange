import subprocess
import sys

DATA_DIR = '/dfs/scratch0/google_ngrams/'
INPUT_DIR = DATA_DIR + 'w2vtrain-tmp/'
OUTPUT_DIR = DATA_DIR + 'vecs-tmp/'

year = sys.argv[1]
subprocess.call(['./word2vecf', 
        '-train',  INPUT_DIR + str(year) + '-train.txt', 
        '-wvocab',  INPUT_DIR + str(year) + '-wv.txt', 
        '-cvocab',  INPUT_DIR + str(year) + '-cv.txt', 
        '-output',  OUTPUT_DIR + str(year) + '-300vecs', 
        '-size', '300',
        '-neg', '15',
        '-threads',  '70', 
        '-iters', '10',
        '-dumpcv',  OUTPUT_DIR + str(year) + '-300ctxvecs'])
