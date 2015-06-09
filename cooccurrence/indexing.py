def word_to_id(word, index):
    word = word.decode('utf-8').lower()
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_cached_id(word, index):
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_fixed_id(word, index):
    word = word.strip("\"")
    try:
        return index[word]
    except KeyError:
        id_ = len(index)
        index[word] = id_
        return id_

def word_to_static_id(word, index):
    return index[word]

def word_to_static_id_pass(word, index):
    try:
        return index[word]
    except KeyError:
        return -1;
