#!/usr/bin/env python3

"""TF-IDF baseline for statement-proof matching task."""

from collections import defaultdict
import sys
import glob
import os
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lap
import scipy.stats as stats
import bipartite_matching

#import mkl
#mkl.set_num_threads(2)

from scipy.special import expit
from globals import *
import globals
from corpus_utils import *


def extract_best_ranking(simvec, k=500):
    """
    Input:
        array of similarities
    Returns:
        indexes of k best highest ranking similarities, sorted by decreasing similarities
        the corresponding similarities
    """
    sim_ranks = stats.rankdata(simvec, method="min")
    N = len(simvec)
    highest_ranking = np.argwhere(sim_ranks > N-k).flatten()
    
    ranks_of_highest = sorted(highest_ranking, key = lambda x: simvec[x], reverse=True)

    return ranks_of_highest, simvec[ranks_of_highest]


def dice(statement, proofs):
    # Dice(X, Y) = 2*|X intersection Y| / (|X| + |Y|)

    intersection = 2 * np.sum(statement.multiply(proofs) > 0, axis=1)
    x_n = np.sum(statement > 0)
    y_n = np.sum(proofs > 0, axis = 1)
    y_n += x_n

    y_n = y_n + 1e-10 # numerical stability, sim should be zero anyway, since |X| + |Y| == 0 => X intersection Y is empty
    similarities = intersection / y_n

    return similarities

def evaluate(statement_vecs, proof_vecs, method="cosine"):
    """Match statements to proofs and evaluate with Mean Reciprocal Rank
    
    Arguments:
        statement_vecs: list of numpy arrays
        proofs_vecs: list of numpy arrays

    Returns:
        mrr, list of ranks for qualitative evaluation (per category), accuracy
    """
    sim_function = cosine_similarity

    if method == "dice":
        sim_function = dice

    mrr = 0
    ranks = []
    acc = 0

    N_examples = statement_vecs.shape[0]
    assert(N_examples == proof_vecs.shape[0])

    for i in range(N_examples):
        if i % 100 == 0:
            logging.getLogger().info("Evaluation: {:.2f} %".format(i * 100 / statement_vecs.shape[0]))
        row = statement_vecs.getrow(i)
        
        similarities = sim_function(row, proof_vecs)

        # for some reason, bug when using dice, need to copy :(
        similarities = np.array(similarities).squeeze()
        # bug: no reshaping method works on the original array, working on a copy works fine (still don't know why)
#        print("shape", similarities.shape, similarities)
#        print("sim.flatten", np.array(similarities).flatten())
#        print("sim.reshape", np.array(similarities).reshape(-1))
#        print("sim.squeeze", np.array(similarities).squeeze())
#        print("sim ravel", similarities.ravel())
        gold_score = similarities[i]

        # Boolean array: which proof have a higher score than gold
        above_score = similarities >= gold_score
        # Rank is just the sum of boolean (True==1), ex-aequo are assigned the worst rank
        rank = above_score.sum() #+ 1  bc >= 
        
        ranks.append(rank)
        
        mrr += 1 / rank
        if rank == 1:
            acc += 1

    return mrr / N_examples, ranks, acc / N_examples

def evaluate_and_output_ranks(outfile, vectorizer, statements, proofs, method="cosine"):
    """Blind evaluation, output $k$-best ranking ids for each statement for future reranking.
    
    Arguments:
        outfile: where to print a ranking and similarities for each statement
        vectorizer: a vectorizer (feature extractor from sklearn)
        statements: list of Statement
        proofs: list of Statement

    Statement and proof lists are not aligned (i.e. the gold proof for statements[i] is not proofs[i])
    """
    
    statement_vecs = vectorizer.transform(statements)
    proof_vecs = vectorizer.transform(proofs)


    sim_function = cosine_similarity
    if method == "dice":
        sim_function = dice

    N_examples = statement_vecs.shape[0]
    assert(N_examples == proof_vecs.shape[0])

    with open(outfile, "w") as f:
        for i in range(N_examples):
            if i % 100 == 0:
                logging.getLogger().info("Evaluation: {:.2f} %".format(i * 100 / statement_vecs.shape[0]))
            row = statement_vecs.getrow(i)
            
            similarities = sim_function(row, proof_vecs)

            similarities = np.array(similarities).squeeze()
            # check neural_model for the rest of function
            ranks, sims = extract_best_ranking(similarities)

            # if all proofs have same score (i.e. mode math, if either the proof or the statement has no math)
            # assign a random proof
            # Note: i is random bc proofs are shuffled + make sure that each randomly choosen proof is assigned once at most
            # (for reranking)
            if len(ranks) == 0:
                ranks = [i]
                sims = [similarities[i]]

            f.write("{}\t{}\t{}\n".format(i,
                             " ".join(map(str, ranks)),
                             " ".join(map(lambda x: str(round(x, 4)), sims))))




def evaluate_dataset_tfidf(vectorizer, dataset, input_type):
    
    statements = [pair.statement.get_statement(input_type) for pair in dataset]
    proofs = [pair.proof.get_statement(input_type) for pair in dataset]
    
    statement_vecs = vectorizer.transform(statements)
    proof_vecs = vectorizer.transform(proofs)
    
    logging.getLogger().info("Tfidf:Evaluation in progress")
    mrr = evaluate(statement_vecs, proof_vecs)
    logging.getLogger().info("Tfidf:Evaluation done")
    return mrr

def tfidf_baseline(args, corpus, dev, test, dev_meta, dev_primary_counts):
    logging.getLogger().info("Tfidf:Vectorizing documents")
    vectorizer = TfidfVectorizer()

#    if args.transductive:
#        statements = [pair.statement.get_statement(args.input) for pair in dev + test]
#        proofs = [pair.proof.get_statement(args.input) for pair in dev + test]
#        corpus = corpus + statements + proofs

    vectorizer.fit(corpus)  # corpus = list of documents

    mrr_dev, ranks_dev, acc_dev = evaluate_dataset_tfidf(vectorizer, dev, args.input)
    mrr_test, ranks_test, acc_test = evaluate_dataset_tfidf(vectorizer, test, args.input)
    
    print("* tfidf-results-MRR-dev {}".format(round(mrr_dev * 100, 1)))
    print("* tfidf-results-MRR-test {}".format(round(mrr_test * 100, 1)))

    print("* tfidf-results-acc-dev {}".format(round(acc_dev * 100, 1)))
    print("* tfidf-results-acc-test {}".format(round(acc_test * 100, 1)))


    with open("{}/baseline_tfidf_voc.log".format(globals.rightnow), "w") as f:
        for k, v in sorted(vectorizer.vocabulary_.items(), key = lambda x : x[1]):
            f.write("{} {}\n".format(k, v))

    logging.info("Tfidf:Dev evaluation by categories")
    evaluate_mrr_by_categories(dev, ranks_dev, dev_meta, dev_primary_counts, "tfidf", primary=True)

    return vectorizer

def evaluate_dice(vectorizer, dataset, input_type):

    statements = [pair.statement.get_statement(input_type) for pair in dataset]
    proofs = [pair.proof.get_statement(input_type) for pair in dataset]

    statement_vecs = vectorizer.transform(statements)
    proof_vecs = vectorizer.transform(proofs)

    logging.info("Dice:Evaluation in progress")
    mrr = evaluate(statement_vecs, proof_vecs, method = "dice")
    logging.info("Dice:Evaluation done")
    return mrr

def dice_baseline(args, corpus, dev, test, dev_meta, dev_primary_counts):
    logging.getLogger().info("Dice:Vectorizing documents")
    vectorizer = CountVectorizer()

    # just word overlap -> count dev and test tokens in vocabulary
    statements = [pair.statement.get_statement(args.input) for pair in dev + test]
    proofs = [pair.proof.get_statement(args.input) for pair in dev + test]
    
    vectorizer.fit(corpus + statements + proofs)
    
    mrr_dev, ranks_dev, acc_dev = evaluate_dice(vectorizer, dev, args.input)
    mrr_test, ranks_test, acc_test = evaluate_dice(vectorizer, test, args.input)
    
    print("* dice-results-MRR-dev {}".format(round(mrr_dev * 100, 1)))
    print("* dice-results-MRR-test {}".format(round(mrr_test * 100, 1)))

    print("* dice-results-acc-dev {}".format(round(acc_dev * 100, 1)))
    print("* dice-results-acc-test {}".format(round(acc_test * 100, 1)))

    with open("{}/baseline_dice_voc.log".format(globals.rightnow), "w") as f:
        for k, v in sorted(vectorizer.vocabulary_.items(), key = lambda x : x[1]):
            f.write("{} {}\n".format(k, v))

    logging.getLogger().info("Dice:Dev evaluation by categories")
    evaluate_mrr_by_categories(dev, ranks_dev, dev_meta, dev_primary_counts, 'dice', primary=True)

    return vectorizer

def initialize_corpus(filename, corpus_type, subset):
    if 'anno' in corpus_type:
        logging.info("Reading {} corpus {}".format(corpus_type, filename))

        dataset, _, _ = read_anno_txt_corpus_with_metadata(filename, subset=subset)
        # assert (len(metadata) == num_doc)
        # assert (set([doc.arxiv_id for doc in dataset]) == set(metadata))

        # all_category_counts, primary_category_counts = compute_category_stats(metadata, corpus_type)

        # print("* {}:data:num-documents {}".format(corpus_type, num_doc))
        print("* {}:data:num-pairs {}".format(corpus_type, len(dataset)))

        return dataset, None, None, None

    logging.info("Reading {} corpus {}".format(corpus_type, filename))

    dataset, metadata, num_doc = read_anno_txt_corpus_with_metadata(filename, subset=subset)
    # assert(len(metadata) == num_doc)
    # assert(set([doc.arxiv_id for doc in dataset]) == set(metadata))

    # all_category_counts, primary_category_counts = compute_category_stats(metadata, corpus_type)

    print("* {}:data:num-documents {}".format(corpus_type, num_doc))
    print("* {}:data:num-pairs {}".format(corpus_type, len(dataset)))

    # return dataset, metadata, all_category_counts, primary_category_counts
    return dataset, metadata, None, None

def main(args):
    """Statement-proof matching tasks -- baselines

    Evaluate two baselines in several settings:

    (i) text and math
    (ii) text only
    (iii) math only

    Baselines:
        TF-IDF
        Dice overlap
    """
    train, metatrain, train_cats, train_primary_cats = initialize_corpus(args.train, "train", args.subset)
    dev, metadev, dev_cats, dev_primary_cats = initialize_corpus(args.dev, "dev", args.subset)
    test, metatest, test_cats, test_primary_cats = initialize_corpus(args.test, "test", args.subset)


    corpus = []
    for pair in train:
        corpus.append(pair.statement.get_statement(args.input))
        corpus.append(pair.proof.get_statement(args.input))
    
    tfidf_vectorizer = tfidf_baseline(args, corpus, dev, test, metadev, dev_primary_cats)

    bow_vectorizer = dice_baseline(args, corpus, dev, test, metadev, dev_primary_cats)

    logger = logging.getLogger()
    for t, filename in [("dev", args.dev), ("test", args.test)]:
        s_file = filename + "_raw_statements"
        p_file = filename + "_raw_proofs"
        if not os.path.isfile(s_file) or not os.path.isfile(p_file):
            logger.warning("Found no file called {} or no file called".format(s_file, p_file))
            continue

        logger.info("Loading statements...")
        statements, num_statements = load_raw(s_file)

        logger.info("Loading proofs...")
        proofs, num_proofs = load_raw(p_file)
        assert(num_statements == num_proofs)

        statements = [s.get_statement(args.input) for s in statements]
        proofs = [p.get_statement(args.input) for p in proofs]

        if args.subset is not None:
            logger.warning("Using a subset of the corpus")
            statements = statements[:args.subset]
            proofs = proofs[:args.subset]

        outtf = "{}/ranks_{}_tfidf".format(args.outputdir, t)
        outdice = "{}/ranks_{}_dice".format(args.outputdir, t)
        evaluate_and_output_ranks(outtf, tfidf_vectorizer, statements, proofs, method="cosine")
        evaluate_and_output_ranks(outdice, bow_vectorizer, statements, proofs, method="dice")

if __name__ == "__main__":

    import argparse
    
    usage = main.__doc__
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("train", help="Train corpus")
    parser.add_argument("dev", help="Dev corpus")
    parser.add_argument("test", help="Test corpus")
    parser.add_argument("outputdir", help="Output directory")

    parser.add_argument("--subset", "-S", type=int, default=None, help="Use only X first training examples")
    parser.add_argument("--input", type=str, default=BOTH, choices=[BOTH, TEXT, FORMULAE])
    #parser.add_argument("--transductive", action="store_true", help="use test and dev to construct vocabulary")

    parser.add_argument("--threads", type=int, default=2, help="MKL num threads")

    args = parser.parse_args()

    globals.rightnow = args.outputdir
    #globals.rightnow = "baseline_{}_".format(args.input) + globals.rightnow
    logging.info("Logs in {}".format(globals.rightnow))
    os.makedirs(globals.rightnow, exist_ok = True)

    with open("{}/baseline_command_line".format(globals.rightnow), "w") as f:
        f.write("{}\n".format(" ".join(sys.argv)))

    #mkl.set_num_threads(args.threads)
    main(args)


