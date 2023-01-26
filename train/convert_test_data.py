import globals
from corpus_utils import Pair, Statement
import corpus_utils
import baselines

import random
import logging

def main(args):
    """Convert a test corpus to 3 files
    - shuffled statements
    - shuffled proofs
    - corresponding keys and arxiv id
    """
    random.seed(23896)
    if args.seed is not None:
        random.seed(args.seed)

    logging.info("Reading corpus")
    corpus, metadata, num_doc = corpus_utils.read_txt_corpus_with_metadata(args.source, subset=args.subset)

    arxiv_ids = [p.arxiv_id for p in corpus]
    statements = [p.statement for p in corpus]
    proofs = list(enumerate([p.proof for p in corpus]))
    
    random.shuffle(proofs)

    logging.info("Writing to files...")
    keys = {}
    for i, e in enumerate(proofs):
        state_id, p = e
        keys[state_id] = i

    logging.info("Statements...")
    with open("{}_statements".format(args.target_prefix), "w") as f:
        f.write("{}\n\n".format(num_doc))
        for s in statements:
            s.write(f)
            f.write("\n")

    logging.info("Proofs...")
    with open("{}_proofs".format(args.target_prefix), "w") as f:
        f.write("{}\n\n".format(num_doc))
        for i, p in proofs:
            p.write(f)
            f.write("\n")

    logging.info("Keys and metadata...")
    with open("{}_keys".format(args.target_prefix), "w") as f:
        for k, v in sorted(keys.items()):
            art_id = arxiv_ids[k]
            meta = metadata[art_id]
            cats, title = meta

            f.write("{}\t{}\t{}\t{}\t{}\n".format(k, v, art_id, "|".join(cats), title))

    logging.info("Done, exiting...")



def niam(args):

    logging.info("Loading statements...")
    statements, num_s = corpus_utils.load_raw("{}_statements".format(args.target_prefix))

    logging.info("Loading proofs...")
    proofs, num_p = corpus_utils.load_raw("{}_proofs".format(args.target_prefix))

    assert(num_p == num_s)

    keys = {}
    metadata = {}
    arxiv_ids = []

    logging.info("Loading keys and metadata...")
    with open("{}_keys".format(args.target_prefix)) as f:
        for line in f:
            line = line.strip().split("\t")
            k, v, arxiv_id, cats, title = line
            k = int(k)
            v = int(v)
            keys[k] = v
            arxiv_ids.append(arxiv_id)
            metadata[arxiv_id] = (cats.split("|"), title)
    
    logging.info("Reconstructing dataset...")
    dataset = []
    for i, statement in enumerate(statements):
        proof = proofs[keys[i]]
        arxiv_id = arxiv_ids[i]
        pair = Pair(arxiv_id)
        pair.statement = statement
        pair.proof = proof
        dataset.append(pair)
    
    logging.info("Exporting dataset")

    corpus_utils.export_dataset(dataset, metadata, len(metadata), args.source + "_reconstructed")

    logging.info("Done, exiting...")


if __name__ == "__main__":

    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("source", help="Specify source file")
    parser.add_argument("target_prefix", help="Specify target prefix")
    parser.add_argument("--subset", "-S", default=None, type=int, help="Only reads N first documents")
    parser.add_argument("--revert", action="store_true", help="Revert transformation (sanity check))")

    parser.add_argument("--seed", type=int, help="Random seed for shuffling corpus")

    args = parser.parse_args()

    if args.revert:
        niam(args)
    else:
        main(args)

