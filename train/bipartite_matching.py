import lap
import logging
import numpy as np
import sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def construct_input_for_lapmod(lines):
    """
    lines: (i, js, ws)
        i is an index
        js is a list of indices (ranking from best to worst)
        ws is a list of corresponding similarities (sorted, same size as js)
    return:
        n: number of lines
        cc, ii, kk: see help(lap.lapmod)
        max_col: max number of columns
    """
    n = len(lines)
    cc = []
    ii = [0]
    kk = []
    for i, js, ws in lines:
        ii.append(ii[-1] + len(js))
        cc.extend(ws)
        kk.extend(js)

    cc = np.array(cc)
    kk = np.array(kk)

    max_col = kk.max()
    #print("n = {} max_col = {}".format(n, max_col))
    n = max(max_col, n)

    while len(ii) < n + 1:
        ii.append(ii[-1])

    ii = np.array(ii)

    # lap / lapmod solve min cost with positive values on edges -> invert the costs
    cc = -(cc - np.max(cc))

    return n, cc, ii, kk



def main(args):
    """
    """
    logging.info("Reading data")
    lines = []
    with open(args.input) as f:
        for line in f:
            line = line.strip().split("\t")
            #assert( len(line) == 3 )
            i = int(line[0])
            if len(line) == 3:
                js = list(map(int, line[1].split()))
                ws = list(map(float, line[2].split()))
            else:
                js = [0]
                ws = [-100]

            sorted_jw = sorted(zip(js, ws), key=lambda x: x[0])
            js, ws = list(zip(*sorted_jw))
            lines.append((i, js, ws))

    logging.info("Reading data: done")

    logging.info("Constructing lapmod input")
    n, cc, ii, kk = construct_input_for_lapmod(lines)

    logging.info("Matching...")
    cost, x, y = lap.lapmod(n, cc, ii, kk)
    logging.info("Matching: done")

    with open(args.output, "w") as f:
        for i, j in enumerate(x):
            f.write("{}\t{}\t0.0\n".format(i,j))


if __name__ == "__main__":

    import argparse

    usage = main.__doc__
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("input", help="Ranks file as output by neural_model.py")
    parser.add_argument("output", help="Output file (same format)")
    args = parser.parse_args()
    main(args)

