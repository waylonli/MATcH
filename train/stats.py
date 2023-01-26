#!/usr/bin/env python3

from string import punctuation
from collections import defaultdict
from globals import *
import globals

import numpy as np


"""
Number of documents int
pairs per doc: min, max, avg, std

tokens per statements: min, max, avg, std
    math tokens per statement 
    text tokens per statement

    outliers per category

    type token ratio on text

    proportion of text / formulae

    MI: between terms and categories
"""

def compute_and_export_statistics(dataset, metadata, num_doc):

    text_tokens = [defaultdict(list), defaultdict(list)]
    math_tokens = [defaultdict(list), defaultdict(list)]

    #num_types = [defaultdict(set), defaultdict(set)]

    cats = set()

    for pair in dataset:
        cat = metadata[pair.arxiv_id][0][0] # recall: {arxivid: [list of categories, title]}
        cats.add(cat)

        for i, statement in enumerate([pair.statement, pair.proof]):
        
            doc_text_tokens = 0
            doc_math_tokens = 0

            for typet, string in statement.text:

                if typet == FORMULAE:
                    tokens = string.split()
                    doc_math_tokens += len(tokens)

                if typet == TEXT:
                    tokens = string.split()
                    doc_text_tokens += len(tokens)

#                    word_types = [tok.strip(punctuation) for tok in tokens]
#                    num_types[i][cat] |= set(word_types)
#                    num_types[i]["all"] |= set(word_types)

            math_tokens[i][cat].append(doc_math_tokens)
            math_tokens[i]["all"].append(doc_math_tokens)

            text_tokens[i][cat].append(doc_text_tokens)
            text_tokens[i]["all"].append(doc_text_tokens)


    funs = [np.min, np.max, np.mean, np.std]

    with open("{}/cu_statistics.csv".format(globals.rightnow), "w") as f:
        f.write("var\tmin\tmax\tmean\tstd\n")
        for cat in ["all"] + sorted(cats):
            for i, typet in enumerate("statement proof".split()):


                ttokens = np.array(text_tokens[i][cat])
                mtokens = np.array(math_tokens[i][cat])
                #all_toks = [n1 + n2 for n1, n2 in zip(ttokens, mtokens)]
                all_toks = ttokens + mtokens
                props = mtokens / all_toks * 100
    
                #num_types_cat = num_types[i][cat]
                
                for var, vec in zip(["all", "text", "math", "math_prop"], 
                                    [all_toks, ttokens, mtokens, props]):

                    if len(vec) > 0:
                        vals = [fun(vec) for fun in funs]
                        vals[0] = round(vals[0], 1)
                        vals[1] = round(vals[1], 1)
                        f.write("{}.{}.{}\t{}\t{}\t{:.1f}\t{:.1f}\n".format(
                            cat,
                            typet,
                            var,
                            *vals))

    with open("{}/cu_shortest_examples.txt".format(globals.rightnow), "w") as f:
        cat = "all"

        tokens_statement = np.array(text_tokens[0][cat]) + np.array(math_tokens[0][cat])
        tokens_proof     = np.array(text_tokens[1][cat]) + np.array(math_tokens[1][cat])
        
        assert(len(tokens_statement) == len(tokens_proof))
        assert(len(dataset) == len(tokens_statement))
        tokens = [(i, ts, tp) for i, ts, tp in zip(range(len(dataset)), tokens_statement, tokens_proof)]
        
        by_state = sorted(tokens, key = lambda x: x[1])
        by_proof = sorted(tokens, key = lambda x: x[2])

        for i, ts, tp in by_state:
            if ts > 6:
                break
            arxivid = dataset[i].arxiv_id
            cats, title = metadata[arxivid]
            cat = cats[0]
            f.write("{} {} {}\n".format(arxivid, cat, title))
            dataset[i].write(f)

        f.write("\n\n=================================\n\n")

        for i, ts, tp in by_proof:
            if tp > 30:
                break
            arxivid = dataset[i].arxiv_id
            cats, title = metadata[arxivid]
            f.write("{} {} {}\n".format(arxivid, cat, title))
            dataset[i].write(f)

    with open("{}/cu_longest_examples.txt".format(globals.rightnow), "w") as f:
        cat = "all"

        tokens_statement = np.array(text_tokens[0][cat]) + np.array(math_tokens[0][cat])
        tokens_proof     = np.array(text_tokens[1][cat]) + np.array(math_tokens[1][cat])
        
        assert(len(tokens_statement) == len(tokens_proof))
        assert(len(dataset) == len(tokens_statement))
        tokens = [(i, ts, tp) for i, ts, tp in zip(range(len(dataset)), tokens_statement, tokens_proof)]
        
        by_state = sorted(tokens, key = lambda x: x[1], reverse=True)
        by_proof = sorted(tokens, key = lambda x: x[2], reverse=True)

        for i, ts, tp in by_state:
            if ts < 1000:
                break
            arxivid = dataset[i].arxiv_id
            cats, title = metadata[arxivid]
            cat = cats[0]
            f.write("{} {} {}\n".format(arxivid, cat, title))
            dataset[i].write(f)

        f.write("\n\n=================================\n\n")

        for i, ts, tp in by_proof:
            if tp < 2000:
                break
            arxivid = dataset[i].arxiv_id
            cats, title = metadata[arxivid]
            f.write("{} {} {}\n".format(arxivid, cat, title))
            dataset[i].write(f)


    return text_tokens, math_tokens


