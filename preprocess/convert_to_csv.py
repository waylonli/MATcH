import random
import pandas as pd
from tqdm import tqdm
from globals import *


random.seed(RANDOM_SEED)

def convert_to_csv(filename, output_path, text_only=False, math_only=False):
    with open(filename, encoding='utf-8') as f:
        document = f.read().split("\n\n")

        # number of docs
        # num_doc = int(document[0])
        #
        # # metadata
        # metadata = {}
        # for line in document[1].split("\n"):
        #     arxivid, cats, title = line.split("\t")
        #     cats = cats.split("|")
        #     #metadata.append((arxivid, cats, title))
        #     metadata[arxivid] = (cats, title)
        #
        # assert(len(metadata) == num_doc)

        df = pd.DataFrame(columns=("theory" ,"proof", "meta"))
        for chunk in tqdm(document[2:]):
            try:
                docid, stat_lines, proof_lines = chunk.split(SEPARATOR)
            except:
                continue
            arxiv_id = docid.strip()
            if not math_only and not text_only:
                stat = " ".join([line.replace("text\t","").replace("math\t","") for line in stat_lines.strip().split("\n")])
                proof = " ".join([line.replace("text\t","").replace("math\t","") for line in proof_lines.strip().split("\n")])
            elif text_only:
                stat = " ".join(
                    [line.replace("text\t", "") if line.startswith("text") else "" for line in stat_lines.strip().split("\n")])
                proof = " ".join(
                    [line.replace("text\t", "") if line.startswith("text") else "" for line in proof_lines.strip().split("\n")])
            elif math_only:
                stat = " ".join(
                    [line.replace("math\t", "") if line.startswith("math") else "" for line in
                     stat_lines.strip().split("\n")])
                proof = " ".join(
                    [line.replace("math\t", "") if line.startswith("math") else "" for line in
                     proof_lines.strip().split("\n")])

            df = df.append(
                [{"meta": arxiv_id, "theory": stat, "proof": proof}])
    df.to_csv(output_path)
    f.close()
    return



if __name__ == "__main__":
    import argparse
    # anno = ['adver', 'full']
    # split = ['train', 'dev', 'test']
    # mode = ['text', 'math']
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help="Specify source directory")
    parser.add_argument("target_file", help="Specify target filename")
    parser.add_argument("--text_only", action="store_true", help="Only use text")
    parser.add_argument("--math_only", action="store_true", help="Only use math")
    args = parser.parse_args()
    # for a in anno:
    #     for s in split:
    #         for m in mode:
    #             source_file = "./tasks/theory-proof-matching/anonymized_dataset/{}_anno_{}".format(a, s)
    #             target_file = "./tasks/theory-proof-matching/anonymized_dataset/{}_anno_{}_{}_only.csv".format(a, s, m)
    #             convert_to_csv(source_file, target_file, text_only=(m=='text'), math_only=(m=='math'))

    convert_to_csv(args.source_file, args.target_file, text_only=args.text_only, math_only=args.math_only)













