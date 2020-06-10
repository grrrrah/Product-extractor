from __future__ import unicode_literals, print_function

import plac
import random
import warnings
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABEL = "PRODUCT"

TRAIN_DATA = [
    (
        "Franck Sofa – Sutherland Furniture",
        {"entities": [(0, 11, LABEL)]},
    ),
    (
        "ROMA SOFA - PEBBLE – Najarian Furniture Skip to content",
        {"entities": [(0, 9, LABEL)]},
    ),
    (
        "Big Cay Desk | Maine Cottage",
        {"entities": [(0, 12, LABEL)]},
    ),
    (
        "Jack Stools — Lostine",
        {"entities": [(0, 11, LABEL)]},
    ),
    (
        "Patio Set Covers (Small Set) — Wholesale Furniture Brokers",
        {"entities": [(0, 16, LABEL)]},
    ),
    (
        "Muuto | Tip Lamp | Shop online at someday designs",
        {"entities": [(9, 16, LABEL)]},
    ),
    (
        "Muuto | Tip Lamp | Shop online at someday designs",
        {"entities": [(9, 16, LABEL)]},
    ),
    (
        "Esme Natural Rattan Day Bed – Koko Collective",
        {"entities": [(0, 27, LABEL)]},
    ),
    (
        "Esme Natural Rattan Day Bed – Koko Collective",
        {"entities": [(0, 27, LABEL)]},
    ),
    (
        "Rattan Day Bed – Koko Collective",
        {"entities": [(0, 14, LABEL)]},
    ),
    (
        "Wooster Convertible Crib – Karla Dubois",
        {"entities": [(0, 24, LABEL)]},
    ),
    (
        "Day Bed – Koko Collective",
        {"entities": [(0, 7, LABEL)]},
    ),
    (
        "Sola desk with solid hardwood top and electronic lift system. - Five Elements Furniture",
        {"entities": [(0, 9, LABEL)]},
    ),
    (
        "SLEEP Mattress, Mattresses by HipVan | HipVan",
        {"entities": [(0, 14, LABEL)]},
    ),
    (
        "Bunk Bed – Koko Collective",
        {"entities": [(0, 8, LABEL)]},
    ),
    (
        "AVA 4.3 - 2 Sliding door wardrobe with LED Lights and the best separat – Wardrobe Bunk Bed Sofa",
        {"entities": [(13, 49, LABEL)]},
    ),
    (
        "AVA 4.3 - 2 Sliding door wardrobe with LED Lights and the best separat – Wardrobe Bunk Bed Sofa",
        {"entities": [(13, 49, LABEL)]},
    ),
    (
        "Wardrobe Bunk Bed Sofa",
        {"entities": [(0, 22, LABEL)]},
    ),
    (
        "Chair",
        {"entities": [(0, 5, LABEL)]},
    ),
    (
        "Tetrad Patna Chair - Hopewells",
        {"entities": [(0, 18, LABEL)]},

    ),


]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="product", output_dir='product extraction', n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("SOMETHING")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes) and warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "modern armchair with minimal design"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)