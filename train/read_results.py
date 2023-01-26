    
from collections import defaultdict

def baseline(args):
    """Read logs and output latex table"""

    res = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    order = {"Both": 0, "Text": 1, "Math": 2}

    with open(args.filename) as f:
        for line in f:
            if line.strip():
                filen, line = line.strip().split(":*")

                metric, value = line.strip().split(":")
                value = round(float(value), 1)

                folder, filename = filen.rsplit("/", 1)
                input = folder.split("_")[-1].capitalize()
                assert( input in order )

                if "_lap" in filename:
                    decoding = "global"
                else:
                    decoding = "local"
            
                _, corpus, method, *rest = filename.split("_")

                if method == "tfidf": method = "TF-IDF"
                if method == "dice": method = "Dice"

#                metric = id[2]
#                corpus = id[3]
                res[corpus][(order[input], input, method)][(metric, decoding)] = value

    dev_lines = []
    for k, v in sorted(res["dev"].items()):
        _, input, method = k
        mrr = v[("mrr", "local")]
        acc = v[("acc", "local")]
        acc_global = v[("acc", "global")]
        line = "        {} & {} & {} & {} & {} \\\\ ".format(input, method, mrr, acc, acc_global)
        dev_lines.append(line)

    test_lines = []
    for k, v in sorted(res["test"].items()):
        _, input, method = k
        mrr = v[("mrr", "local")]
        acc = v[("acc", "local")]
        acc_global = v[("acc", "global")]
        line = "        {} & {} & {} & {} & {} \\\\ ".format(input, method, mrr, acc, acc_global)
        test_lines.append(line)

    table=r"""{header}
    \toprule
{title}
    Input & Method & MRR & Accuracy & Accuracy\\
    \midrule
    {mid1}
    \midrule
{dev}
    \midrule
    {mid2}
    \midrule
{test}
    \bottomrule
{footer}
    """.format(header=r"\begin{tabular}{llrrr}",
               footer=r"\end{tabular}",
               title=r"& & \multicolumn{2}{c}{Local decoding} & Global decoding \\",
               dev="\n".join(dev_lines), 
               test="\n".join(test_lines),
               mid1=r"\multicolumn{5}{l}{Dev} \\ ",
               mid2=r"\multicolumn{5}{l}{Test} \\")

    print(table)

def conv(args):

    res = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    order = {"Both": 0, "Text": 1, "Math": 2}
    with open(args.filename) as f:
        for line in f:
            if line.strip():
                filen, line = line.strip().split(":*")
                metric, v = line.strip().split(":")

                input = filen.split("_")[2].capitalize()
                filename = filen.split("/")[1]
                
                corpus = filename.split("_")[1]
                
                if "lap" in filename:
                    decoding = "global"
                else:
                    decoding = "local"

                res[corpus][(order[input], input)][(metric,decoding)] = round(float(v), 1)

    dev_lines = []
    for k, v in sorted(res["dev"].items()):
        _, input = k
        mrr_loc = v["mrr","local"]
        acc_loc = v["acc","local"]
        acc_glo = v["acc","global"]

        line = "        {} & {} & {} & {} \\\\ ".format(input, mrr_loc, acc_loc, acc_glo)
        dev_lines.append(line)

    test_lines = []
    for k, v in sorted(res["test"].items()):
        _, input = k
        mrr_loc = v["mrr","local"]
        acc_loc = v["acc","local"]
        acc_glo = v["acc","global"]

        line = "        {} & {} & {} & {} \\\\ ".format(input, mrr_loc, acc_loc, acc_glo)
        test_lines.append(line)

    table=r"""
    {open}
    \toprule
    {multi1}
            & MRR & Accuracy & Accuracy\\
    \midrule
    {multi2} 
    \midrule
{linesdev}
    \midrule
    {multi3}
    \midrule
{linestest}
    \bottomrule
    {close}
    """.format(open=r"\begin{tabular}{lccc}",
               close=r"\end{tabular}",
               multi1=r"Input & \multicolumn{2}{c}{Local decoding} & Global decoding \\",
               multi2=r"Dev & & & \\",
               multi3=r"Test & & & \\",
               linesdev="\n".join(dev_lines),
               linestest="\n".join(test_lines))
    print(table)


def main(args):
    if args.type == "baseline":
        baseline(args)
    else:
        conv(args)

if __name__ == "__main__":

    import argparse

    usage = main.__doc__
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("type", choices=["baseline", "convolution", "baseline_old"])
    parser.add_argument("filename", help="Ouput of 'grep [a-z] logs/log_baseline_*' or grep [a-z] models*/eval*")
    args = parser.parse_args()
    main(args)
