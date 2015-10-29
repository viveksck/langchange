import argparse

def write_word_list(filename, word_list):
    out_fp = open(filename, "w")
    print >> out_fp, "\n".join(word_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Makes valid word list for file pattern lexicon file")
    parser.add_argument("out_file")
    parser.add_argument("in_file", help="Path to pattern lexicon")
    parser.add_argument("bad_tag", help="Language specific proper noun tag")
    args = parser.parse_args()
    bad_tag = args.bad_tag
    fp = open(args.in_file)
    good_words = set([])
    for line in fp:
        if line.startswith(";;;"):
            continue
        info = line.split()
        word = info[0].lower()
        tag = info[1]
        if tag != bad_tag:
            good_words.add(word)
    write_word_list(args.out_file, list(good_words))
