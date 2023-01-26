import logging
import sys

def read_keys(filename):
    res = []
    with open(filename) as f:
        for line in f:
            g, pid, arxivid, cat, title = line.strip().split("\t")
            res.append((int(g), int(pid), arxivid, cat.split("|")[0], title))
    return res

def read_pred(filename):
    with open(filename) as f:
        for line in f:
            split_line = line.strip().split("\t")
            if len(split_line) == 3:
                i, ranks, sims = split_line
            else:
                i = split_line[0]
                ranks = sims = "0"
            i = int(i)
            ranks = [int(j) for j in ranks.split()]
            sims = [float(j) for j in sims.split()]
            yield i, ranks, sims
            

def main(args, logger):
    keys = read_keys(args.goldkeys)
    c = 0
    acc = 0
    mrr = 0

    N = len(keys)
    for gold, pred in zip(keys, read_pred(args.predranks)):
        i1, proof_id, arxiv_id, cat, title = gold

        i2, ranks, sims = pred
        assert(i1 == i2)
        assert(i1 == c)

        if proof_id == ranks[0]:
            acc += 1
        
        rank = N
        for k in range(len(ranks)):
            if ranks[k] == proof_id:
                rank = k + 1
                break

        logger.info("{} rank: {}".format(c, rank))
        
        mrr += 1 / rank
        c += 1

    print("* mrr:{}".format(round(mrr/N * 100, 2)))
    print("* acc:{}".format(round(acc/N * 100, 2)))


if __name__ == "__main__":
    import argparse

    usage = main.__doc__
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    

    # train corpora
    parser.add_argument("goldkeys", help="Gold ref")
    parser.add_argument("predranks", help="model output")
    parser.add_argument("--verbose", "-v", type=int, default=10, help="Logger verbosity, higher is quieter")

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    
    logger = logging.getLogger()
    logger.setLevel(args.verbose)

    main(args, logger)


