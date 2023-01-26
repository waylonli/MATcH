from collections import defaultdict


physics_ids={"astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph",
             "hep-th", "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph"}

def main(filename):
    """
    Load corpus and compute category stats such as:
        number of articles
        most frequent categories (articles)
        most frequent categories (pairs)
    """
    metadata = {}
    pairs = []

    with open(filename) as f:
        num_doc = int(f.readline())
        f.readline()

        buffer = f.readline()
        while buffer.strip():
            arxivid, cats, title = buffer.split("\t")
            cats = cats.split("|")
            metadata[arxivid] = (cats, title)
            buffer = f.readline()

        for line in f:
            if line.startswith("arXiv"):
                pairs.append(line.strip())
    
    category_stats_articles = defaultdict(int)
    coarse_category_articles = defaultdict(int)
    n_coarse_categories = defaultdict(set)

    category_stats_pairs = defaultdict(int)
    coarse_category_pairs = defaultdict(int)

    num_primary_cats = set()
    num_all_cats = set()
    num_cats_per_article = []

    for arxivid in metadata:
        cats, title = metadata[arxivid]
        category_stats_articles[cats[0]] += 1

        num_primary_cats.add(cats[0])
        for cat in cats:
            num_all_cats.add(cat)

        coarse = cats[0].split(".")[0]
        coarse_category_articles[coarse] += 1
        n_coarse_categories[coarse].add(cats[0])
        num_cats_per_article.append(len(cats))

        if coarse in physics_ids:
            category_stats_articles["ALL_PHYSICS"] += 1
            n_coarse_categories["ALL_PHYSICS"].add(cats[0])


    for arxivid in pairs:
        cats, title = metadata[arxivid]
        category_stats_pairs[cats[0]] += 1
        coarse = cats[0].split(".")[0]
        coarse_category_pairs[coarse] += 1

    print("Articles")
    cats = []
    for cat, val in sorted(category_stats_articles.items(), key= lambda x: x[1], reverse=True):
        print(cat, val, category_stats_pairs[cat], sep="\t")
        cats.append(cat)

    print()
    print("Pairs")
    for cat, val in sorted(category_stats_pairs.items(), key= lambda x: x[1], reverse=True):
        print(cat, val)

    print()
    print("Coarse articles")
    for cat, val in sorted(coarse_category_articles.items(), key = lambda x:x[1], reverse=True):
        print(cat, val)

    print()
    print("Coarse category, pairs")
    for cat, val in sorted(coarse_category_pairs.items(), key = lambda x:x[1], reverse=True):
        print(cat, val)

    print()
    print("Sub cat per coarse cat")
    for cat, val in n_coarse_categories.items():
        print(cat, len(val), sorted(val), sep="\t")

    print()
    print("Total number of pairs = ", len(pairs))
    print("Total number of articles = ", len(metadata), len(set(pairs)))
    print("Num primary categories = ", len(num_primary_cats), num_primary_cats)
    print("Num all categories = ", len(num_all_cats), num_all_cats)
    print("Avg number of cat per article = ", sum(num_cats_per_article) / len(num_cats_per_article))

if __name__ == "__main__":

    import argparse

    usage = main.__doc__

    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("corpus")

    args = parser.parse_args()

    main(args.corpus)

