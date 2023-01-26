from collections import defaultdict
import logging
import glob
import re
import sys
import random
import spacy

# from polyglot.detect import Detector
Detector = None
from tqdm import tqdm

from stats import *

from globals import *
import globals
import pickle



def data_split(data, percentage):
    random.seed(RANDOM_SEED)
    random.shuffle(data)
    # if not os.path.exists("split_id.txt"):
    #     arxiv_ids = set([ex.arxiv_id for ex in data])
    #     dev_arxiv_ids = set(random.sample(arxiv_ids, int(len(arxiv_ids) * percentage)))
    #     test_arxiv_ids = set(random.sample(arxiv_ids - dev_arxiv_ids, int(len(arxiv_ids) * percentage)))
    #     train_arxiv_ids = arxiv_ids - dev_arxiv_ids - test_arxiv_ids
    #     with open("split_id.txt", 'wb') as f:
    #         pickle.dump((train_arxiv_ids, dev_arxiv_ids, test_arxiv_ids), f)
    #         f.close()
    # else:
    with open("/bask/projects/j/jlxi8926-auto-sum/waylon/mathbert/tasks/theory-proof-matching/anonymized_dataset/split_id", 'rb') as f:
        train_arxiv_ids, dev_arxiv_ids, test_arxiv_ids = pickle.load(f)
        f.close()

    train = []
    dev = []
    test = []
    for ex in data:
        if ex.arxiv_id in train_arxiv_ids:
            train.append(ex)
        elif ex.arxiv_id in dev_arxiv_ids:
            dev.append(ex)
        elif ex.arxiv_id in test_arxiv_ids:
            test.append(ex)
        else:
            raise Exception("This should not happen")
    return train, dev, test





######### Evaluation

def evaluate_mrr_by_categories(dev, ranks, metadata, category_counts, method, primary=True):
    rank_dict = defaultdict(list)
    
    for ex, rank in zip(dev, ranks):
        cats = metadata[ex.arxiv_id][0]
        if primary:
            rank_dict[cats[0]].append(rank)
        else:
            for c in cats:
                rank_dict[c].append(rank)
    
    with open("{}/baseline_results_by_category_{}.log".format('win_test', method), "w") as f:
        for cat, count in sorted(category_counts.items(), key = lambda x : (x[1], x[0]), reverse = True):
            if count > 100:
                rank_list = rank_dict[cat]
                mrr = sum([1 / i for i in rank_list]) / len(rank_list)
                mrr = round(mrr * 100, 1)
                f.write("{} {} {}\n".format(cat, mrr, count))







############# Data classes

class Pair:
    options = None
    def __init__(self, arxiv_id, pair_element=None):
        self.arxiv_id = arxiv_id
        self.statement = Statement()
        self.proof = Statement()
        self.statement_id = None
        if pair_element is not None:
            self.statement = Statement(pair_element.find(STATEMENT_TAG))
            self.proof = Statement(pair_element.find(PROOF_TAG))
            self.statement_id = self.statement.statement_id
    
    def write(self, stream, no_proof = False, no_statement = False):
        stream.write("{}|{}\n".format(self.arxiv_id, self.statement_id))
        if not no_statement:
            stream.write("{}\n".format(SEPARATOR))
            self.statement.write(stream)
        if not no_proof:
            stream.write("{}\n".format(SEPARATOR))
            self.proof.write(stream)
        stream.write("\n")
    
    def load(self, chunk):
        docid, stat_lines, proof_lines = chunk.split(SEPARATOR)
        self.arxiv_id = docid.strip()
        self.statement.load(stat_lines.strip().split("\n"))
        self.proof.load(proof_lines.strip().split("\n"))


class Statement:
    tag_statistics = defaultdict(int)
    tag_notext = defaultdict(int)
    token_statistics = defaultdict(lambda : defaultdict(int))
    attributes = defaultdict(lambda : defaultdict(int))
    tokenizer = spacy.load("en_core_web_sm")

    def __init__(self, root=None):
        self.text = []
        self.statement_id = None
        if root is not None:
            self.text = process_doc(root)
            self.statement_id = root.getchildren()[0].get('id')

    def __str__(self):
        return "\n".join([ "\t".join(pair) for pair in self.text ])

    def get_statement(self, ttype):
        if ttype == BOTH:
            return " ".join([text for _, text in self.text])
        assert(ttype in {TEXT, FORMULAE})
        return " ".join([text for T, text in self.text if T == ttype])

    def get_statement_tokenized(self, ttype):
        texts = self.text
        res = []
        for t, text in texts:
            if t == TEXT:
                if ttype == FORMULAE:
                    continue
                #"""
                for token in Statement.tokenizer.tokenizer(text):
                    res.append(token.text)
                """
                res.extend(list(text))
                res.append(" ")
                """
            if t == FORMULAE:
                if ttype == TEXT:
                    continue
                for token in text.split():
                    if token.startswith("math_"):
                        token = token[5:]
                    stoken = token.split("____")
                    if len(stoken) == 1:
                        res.append(("math", stoken[0]))
                    elif len(stoken) == 2:
                        res.append(("math", stoken[0], stoken[1]))
                    else:
                        assert(False)
                #"""
                res.append(" ")
                #"""
        return res

    def write(self, stream):
        for t, text in self.text:
            stream.write("{}\t{}\n".format(t, text))
    
    def load(self, lines):
        for line in lines:
            sline = line.strip().split("\t")
            self.text.append((sline[0], sline[1]))

    @staticmethod
    def dump_statistics():
        with open("{}/cu_tag_stats.log".format('win_test'), "w") as f:
            for t, v in sorted(Statement.tag_statistics.items(), reverse=True):
                f.write("{} {}\n".format(t, v))

        with open("{}/cu_tag_notext_stats.log".format('win_test'), "w") as f:
            for t, v in sorted(Statement.tag_notext.items(), reverse=True):
                with_text = Statement.tag_statistics[t] - v
                f.write("{} {} {}\n".format(t, v, with_text))

        with open("{}/cu_tag_attributes_stats.log".format('win_test'), "w") as f:
            triples = set()
            for t, v in Statement.attributes.items():
                for k, v2 in sorted(v.items(), reverse=True):
                    if v2 == 1:
                        k = (k[0], UNIQUE)
                    triples.add((t, '='.join(k), v2))
            for triple in sorted(triples):
                f.write("{} {} {}\n".format(*triple))

        with open("{}/cu_token_stats.log".format('win_test'), "w") as f:
            for t, v in Statement.token_statistics.items():
                for k, v2 in sorted(v.items(), reverse=True):
                    f.write("{} {} {}\n".format(t, k, v2))






############ Processing functions

def obfuscate_variables(root):
    # TODO:  ----> just remove <mi>
#    i = 0
#    for element in root.iter():
#        if element.tag in { "{{{}}}{}".format(namespace["m"], VARIABLE_TAG),  
#                            "{{{}}}{}".format(namespace["m"], CONTENT_V_TAG)}:
#            element.text = "__VAR__{}".format(i)
#            i += 1
    return root

def remove_multiple_spaces(s):
    return " ".join([string for string in s.strip().split() if string])
    #return re.sub(r" +", " ", s.strip().replace("\t", " "))



def process_math_opening_tag(element, l):
    if "b" in Pair.options:
        if element.tag == "{{{}}}mfenced".format(namespace["m"]):
            if "open" in element.attrib:
                l.append(element.attrib["open"])
            else:
                l.append("(")
    
    tag = element.tag.split("}")[-1]
    if "p" in Pair.options and tag in SILENT_TAGS:
        l.append("TAG_{}".format(tag))

def process_math_closing_tag(element, l):
    if "b" in Pair.options:
        if element.tag == "{{{}}}mfenced".format(namespace["m"]):
            if "close" in element.attrib:
                l.append(element.attrib["close"])
            else:
                l.append(")")

def process_font(element, text):
    if "mathvariant" in element.attrib:
        font = element.attrib["mathvariant"]
        if font != "normal":
            text = text.split()
            text = ["{}____{}".format(t, font) for t in text]
            return " ".join(text)
    return text


def process_math_rec(element, l):
    process_math_opening_tag(element, l)

    Statement.tag_statistics[element.tag] += 1
    
    if element.text is None or not element.text.strip():
        Statement.tag_notext[element.tag] += 1

    for k, v in element.attrib.items():
        if k != "id":
            Statement.attributes[element.tag][(k,v)] += 1

    for child in element.xpath("child::node()"):
        if isinstance(child, str):
            text = remove_multiple_spaces(child)
            if text.strip():
                if "f" in Pair.options:
                    text = process_font(element, text)
                l.append(text)
                for token in text.split():
                    Statement.token_statistics[element.tag][token] += 1
        else:
            process_math_rec(child, l)

    process_math_closing_tag(element, l)

def process_math(element):
    element = transform(element)
    math_string = []
    process_math_rec(element.getroot(), math_string)
    return " ".join(math_string)

def process_doc_rec(element, res):
    if element.tag == "{{{}}}math".format(namespace["m"]):
        math_text = process_math(element).strip()
        if math_text:
            if "t" in Pair.options:
                math_text = " ".join(["math_{}".format(tok) for tok in math_text.split()])
            res.append((FORMULAE, math_text))
    else:
        for child in element.xpath("child::node()"):
            if isinstance(child, str):
                text = remove_multiple_spaces(child).strip()
                if text:
                    res.append((TEXT, text))
            else:
                process_doc_rec(child, res)

def process_doc(element):
    l = []
    process_doc_rec(element, l)
    return l

def get_metadata(root):
    
    meta_node = root.find("meta")
    
    try:
        arxivid = meta_node.attrib["arxivid"]
        categories = meta_node.attrib["category"]
        title = meta_node.attrib["title"]
        return (arxivid, categories.split(" "), title)

    except:
        root_node = root.getroot()
        logging.warning("Could not find metadata for {}".format(root_node.attrib["path"]))
        return None


def compute_category_stats(metadata, corpus_type):
    count = defaultdict(int)
    primary = defaultdict(int)
    avg = 0
    for categories, _ in metadata.values():
        primary[categories[0]] += 1
        for c in categories:
            count[c] += 1
        avg += len(categories)
    
    with open("{}/cu_category_stats_{}.log".format('win_test', corpus_type), "w") as f:
        
        f.write("* Avg-num-cat-per-article {}\n".format(avg / len(metadata)))
        f.write("* Num-categories {}\n".format(len(count)))

        f.write("# Stats-all-categories\n")
        for cat, num in sorted(count.items(), key=lambda x : x[1], reverse=True):
            f.write("{} {}\n".format(cat, num))

        f.write("# Stats-primary-categories\n")
        for cat, num in sorted(primary.items(), key=lambda x : x[1], reverse=True):
            f.write("{} {}\n".format(cat, num))
    
    return count, primary




def diagnostic(pair, non_english, encoding, langs, min_length=20, max_length=2000):

    if pair.arxiv_id in non_english and non_english[pair.arxiv_id] > 1:
        return 3
    elif pair.arxiv_id in encoding and encoding[pair.arxiv_id] > 1:
        return 4

    statement = pair.statement.get_statement(TEXT)
    proof = pair.proof.get_statement(TEXT)

    # try:
    #     language = Detector(" ".join([statement, proof]))
    #     if language.language.name != "English":
    #         langs[pair.arxiv_id] = language.language.name
    #         non_english[pair.arxiv_id] += 1
    #         logging.info("Detected {} for {}".format(language.language.name, pair.arxiv_id))
    #         return 3
    # except:
    #     encoding[pair.arxiv_id] += 1
    #     logging.warning("Language detector broke on {}".format(pair.arxiv_id))
    #     return 4

    statement = pair.statement.get_statement(BOTH)
    statement_tokens = statement.strip(" .").split()
    if len(statement_tokens) < min_length:
        return 1
    if len(statement_tokens) > max_length:
        return 5
    proof = pair.proof.get_statement(BOTH)
    proof_tokens = proof.split()
    if len(proof_tokens) < min_length:
        return 2
    if len(proof_tokens) > max_length:
        return 5

    return 0


def filter_dataset(dataset, metadata, args):

    ### shortest examples
    ### non-english articles
    ### to print in external files

    non_english = defaultdict(int)
    encoding = defaultdict(int)
    languages = {}

    with open("{}/cu_discarded_short_statements.txt".format('win_test'), "w") as fshorts,\
         open("{}/cu_discarded_short_proofs.txt".format('win_test'), "w") as fshortp,\
         open("{}/cu_discarded_nonenglish.txt".format('win_test'), "w") as fnonenglish,\
         open("{}/cu_discarded_encoding_crash.txt".format('win_test'), "w") as fencoding:
        
        new_dataset = []
        for pair in dataset:

            diag = diagnostic(pair, non_english, encoding, languages, min_length=args.m, max_length=args.M)
            if args.no_filter:
                new_dataset.append(pair)
            else:
                if diag == 0:
                    new_dataset.append(pair)
                elif diag == 1:
                    logging.warning("Discarding {}: short statement".format(pair.arxiv_id))
                    pair.write(fshorts, no_proof = True)
                elif diag == 2:
                    logging.warning("Discarding {}: short proof".format(pair.arxiv_id))
                    pair.write(fshortp, no_statement = True)
                elif diag == 3:
                    logging.warning("Discarding {}: non english".format(pair.arxiv_id))
                    pair.write(fnonenglish)
                elif diag == 4:
                    logging.warning("Discarding {}: encoding".format(pair.arxiv_id))
                    pair.write(fencoding)
                elif diag == 5:
                    logging.warning("Discarding {}: statement or proof too long".format(pair.arxiv_id))
                else:
                    assert(False)

    article_ids = {pair.arxiv_id for pair in new_dataset}
    newmetadata = {k: v for k, v in metadata.items() if k in article_ids}
    # assert(len(article_ids) == len(newmetadata))

    with open("{}/cu_discarded_arxivid.txt".format('win_test'), "w") as farx:
        for arxivid in sorted(non_english):
            farx.write("{} {} {}\n".format(arxivid, (arxivid in article_ids), languages[arxivid]))

        for arxivid in sorted(encoding):
            farx.write("{} {} {}\n".format(arxivid, (arxivid in article_ids), "encoding"))

    return new_dataset, newmetadata, len(article_ids)





######## Read functions

def read_xml_corpus_with_metadata(dataset_directory, subset=None):

    dataset = []
    metadata = {}
    num_doc = 0
    
    statistics = defaultdict(int)

    filenames = sorted(glob.glob("{}/*.xhtml".format(dataset_directory)))
    for filename in filenames:

        if subset is not None and num_doc > subset:
            break
        root = parse_xhtmlfile(filename)

        get_meta_output = get_metadata(root)
        if get_meta_output != None:

            #metadata.append(get_meta_output)
            arxivid = get_meta_output[0]
            assert(arxivid not in metadata)
            metadata[arxivid] = get_meta_output[1:]
        
            for pair_element in root.findall(PAIR_TAG):
                dataset.append(Pair(arxivid, pair_element))
            
            num_doc += 1
            assert(len(metadata) == num_doc)
        else:
            logging.warning("No metadata for {}".format(filename))
        
        if num_doc % 10 == 0:
            max_doc = subset if subset else len(filenames)
            logging.info("Doc {} out of {} ({} %)".format(num_doc, max_doc, round(num_doc / max_doc * 100, 2)))

    Statement.dump_statistics()

    return dataset, metadata, num_doc

def read_txt_corpus_with_metadata(filename, subset = None):
    with open(filename) as f:
        document = f.read().split("\n\n")
        
        # number of docs
        num_doc = int(document[0])

        # metadata
        metadata = {}
        for line in document[1].split("\n"):
            arxivid, cats, title = line.split("\t")
            cats = cats.split("|")
            #metadata.append((arxivid, cats, title))
            metadata[arxivid] = (cats, title)

        assert(len(metadata) == num_doc)

        # dataset
        dataset = []

        arxivids = set()
        for chunk in document[2:]:
            if chunk.strip():
                p = Pair(None)
                p.load(chunk)
                dataset.append(p)
                arxivids.add(p.arxiv_id)
                if subset is not None and len(arxivids) == subset + 1:
                    dataset.pop()
                    arxivids.remove(p.arxiv_id)
                    num_doc = subset
                    metadata = {k : v for k, v in metadata.items() if k in arxivids}
                    break
                

        return dataset, metadata, num_doc


def read_anno_txt_corpus_with_metadata(filename, subset=None):
    with open(filename) as f:
        document = f.read().split("\n\n")

        # number of docs
        num_doc = int(document[0])

        # dataset
        dataset = []

        arxivids = set()
        for chunk in document[2:]:
            if chunk.strip():
                p = Pair(None)
                p.load(chunk)
                dataset.append(p)
                arxivids.add(p.arxiv_id)

        return dataset, None, num_doc


def read_anno_xml_corpus(dataset_directory, subset=None):
    dataset = []
    metadata = {}
    num_doc = 0

    statistics = defaultdict(int)

    filenames = sorted(glob.glob("{}/*.html".format(dataset_directory)))

    # for filename in tqdm(filenames):
    #     print(filename)
    #     with open(filename) as f:
    #         text = f.read()
    #     f.close()
    #     with open(filename, 'w') as f_out:
    #         f_out.write(text.replace("<mi", "<m:mi").replace("</mi>", "</m:mi>"))
    #     f_out.close()
    #     # os.remove(filename)
    # filenames = sorted(glob.glob("{}/*.html".format(dataset_directory)))
    for filename in tqdm(filenames):

        if subset is not None and num_doc > subset:
            break
        # try:
        root = parse_xhtmlfile(filename)
        # except:
        #     with open(filename) as f:
        #         text = f.read()
        #         new_text = '<pair>\n' + text
        #     f.close()
        #     with open(filename, 'w') as f_out:
        #         f_out.write(new_text)
        #     f_out.close()
        #     root = parse_xhtmlfile(filename)

        # get_meta_output = get_metadata(root)
        # if get_meta_output != None:
        #
        #     # metadata.append(get_meta_output)
        #     arxivid = get_meta_output[0]
        #     assert (arxivid not in metadata)
        #     metadata[arxivid] = get_meta_output[1:]

        for pair_element in root.findall(PAIR_TAG):
            dataset.append(Pair(num_doc, pair_element))

        num_doc += 1
        #     assert (len(metadata) == num_doc)
        # else:
        #     logging.warning("No metadata for {}".format(filename))

        # if num_doc % 10 == 0:
        #     max_doc = subset if subset else len(filenames)
        #     logging.info("Doc {} out of {} ({} %)".format(num_doc, max_doc, round(num_doc / max_doc * 100, 2)))

    Statement.dump_statistics()

    return dataset, metadata, num_doc

def load_raw(filename):
    # Load corpus in raw format (only statements or only proofs)
    statements = []
    with open(filename) as f:
        s = f.read().split("\n\n")
        num_doc = int(s[0])
        for chunk in s[1:]:
            chunk = chunk.strip()
            if chunk:
                statement = Statement()
                statement.load(chunk.split("\n"))
                statements.append(statement)
    return statements, num_doc



## Write function
def export_dataset(dataset, metadata, num_doc, filename):
    # assert(num_doc == len(metadata))
    with open(filename, "w") as f:
        f.write("{}\n\n".format(num_doc))
        
        for arxivid, v in sorted(metadata.items()):
            cats, title = v
            f.write("{}\t{}\t{}\n".format(arxivid, "|".join(cats), title))

        f.write("\n")
        
        for p in dataset:
            p.write(f)


def main(args):
    """Read the whole statement-proof dataset in xml format
    and dumps it in a fast-readable format."""

    # dataset, metadata, num_doc = read_xml_corpus_with_metadata(args.source_dir, subset=args.subset)
    dataset, metadata, num_doc = read_anno_xml_corpus(args.source_dir, subset=args.subset)
    print(args.source_dir)
    print("len dataset before filtering", len(dataset))
    dataset, metadata, num_doc = filter_dataset(dataset, metadata, args)
    print("len dataset after filtering", len(dataset))

    # text_token_counts, math_token_counts = compute_and_export_statistics(dataset, metadata, num_doc)

    export_dataset(dataset, metadata, num_doc, args.target_file)

    # assert(num_doc == len(metadata))

    train, dev, test = data_split(dataset, args.s)
    for corpus, name in zip([train, dev, test], ["train", "dev", "test"]):
        newids = {pair.arxiv_id for pair in corpus}
        newmetadata = {k:v for k, v in metadata.items() if k in newids}
        export_dataset(corpus, newmetadata, len(newmetadata), "{}_{}".format(args.target_file, name))


#    data2, meta2, num_doc2 = read_txt_corpus_with_metadata(args.target_file)
#    export_dataset(dataset, metadata, num_doc, args.target_file + "__")

if __name__ == "__main__":

    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("source_dir", help="Specify source directory")
    parser.add_argument("target_file", help="Specify target filename")
    parser.add_argument("--subset", "-S", default=None, type=int, help="Only reads N first documents")


    parser.add_argument("--variables", action="store_true", help="Neutralize mathematical identifiers variables")
    parser.add_argument("--preprocessing", type=str, default="b", help="arg: [bpft]+ (b)rackets (p)ositional (t)ypes (f)onts")
    parser.add_argument("--no_filter", type=bool, default=False, help="apply filter or not")
    parser.add_argument("-m", type=int, default=20, help="min length")
    parser.add_argument("-M", type=int, default=500, help="max length")
    parser.add_argument("-s", type=float, default=0.1, help="size of dev and test (in [0, 1] interval)")

    args = parser.parse_args()

    if args.variables:
        args.preprocessing += "v"

    globals.rightnow = "cu_{}_".format(args.preprocessing) + 'win_test'

    logging.info("Logs in {}".format('win_test'))
    os.makedirs('win_test', exist_ok = True)

    with open("{}/cu_command_line".format('win_test'), "w") as f:
        f.write("{}\n".format(" ".join(sys.argv)))


    Pair.options = args.preprocessing

    main(args)












