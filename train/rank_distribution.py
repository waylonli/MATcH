from collections import defaultdict
import sys
import evaluator

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
            i, ranks, sims = line.strip().split("\t")
            i = int(i)
            ranks = [int(j) for j in ranks.split()]
            sims = [float(j) for j in sims.split()]
            yield i, ranks, sims
            

def main(args):
    """Check if same same proof is assigned to several statements"""

    best = defaultdict(int)
    all_best = defaultdict(int)

    N = 0
    for i, ranks, sims in evaluator.read_pred(args.predranks):
        N += 1
        best[ranks[0]] += 1
        for r in ranks:
            all_best[r] += 1

    N_ranks = len(best)
    vals = [20, 10, 5, 2]

    items = sorted(best.items(), key = lambda x: x[1], reverse = True)
    
    res = {}
    for v in vals:
        effectif = len([s[0] for s in items if s[1] >= v])
        res[v] = effectif

    res[1] = len([s[0] for s in items if s[1] == 1])

    for k in res:
        if k != 1:
            print(r"$\geq {}$  & {} & {} \\".format(k, res[k], round(res[k] * 100 / N, 1)))
    print(r"$=1$  & {} & {} \\".format(res[1],round(res[1]*100 / N, 1)))
    print(r"$<1$  & {} & {} \\".format(N-N_ranks,round((N-N_ranks)*100 / N, 1)))

#    print(N)
#    print(N_ranks)
#    print("Never assigned", N-N_ranks)
    


if __name__ == "__main__":
    import argparse

    usage = main.__doc__
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("predranks", help="Predicted ranks")
    args = parser.parse_args()
    
    main(args)


