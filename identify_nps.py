from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
import spacy


def get_continuous_chunks(text, chunk_func=ne_chunk):
    with open(text, "r", encoding='utf-8') as file_open:
        txt = file_open.readlines()
        return txt

    chunked = chunk_func(pos_tag(word_tokenize(txt)))
    print(chunked)
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    return continuous_chunk


def extract_entity(text):
    nps = get_continuous_chunks(text)
    nps = ', '.join(nps)
    test_str = nps
    prod_nlp = spacy.load('product extraction')
    ner = prod_nlp.get_pipe("ner")
    move_names = list(ner.move_names)
    doc = prod_nlp(test_str)

    for ent in doc.ents:
        return ent.label_, ent.text


