# MATcH - A Dataset for the Mathematical Statement-Proof Matching Problem

## Dataset Description

- **Homepage:** https://bollin.inf.ed.ac.uk/match.html (download page)
- **Repository:** https://github.com/waylonli/MATcH
- **Paper:** BERT is not The Count: Learning to Match Mathematical Statements with Proofs (EACL 2023)
- **Point of Contact:** Waylon Li, Yftah Ziser, Shay Cohen

### Dataset Summary

This dataset includes a set of mathematical statements with their corresponding proofs. The dataset is extracted from a set of arXiv articles, based on the MREC corpus (see below). The dataset includes two variants: an "easy" one, in which statement-proofs pairs from the same article may appear in all training/validation/test sets, and a "hard" one, in which statement-proofs pairs from the same article appear only in one set.

### Supported Tasks

The dataset supports the task of matching mathematical statements to proofs. The paper that describes the task and several algorithms to solve it appeared in EACL 2023.

### Languages

The dataset mostly includes text in English (with notation), but there are a few cases in which statement/proofs pairs are in a different language.

### Dataset Structure

The dataset includes three subsets: `mixed/`, `unmixed/` and `pb/`. `mixed` and `unmixed` stand for the cases where statement-pairs from the same article can span several splits (train, dev, test) or not, respectively, and pb stands for the probability domain restriction of mixed.

Each directory includes the following types of files:

- `{dirname}_{conservation,partial,trans,full}_{train,dev,test}`

where `dirname` is either `mixed`, `unmixed`, or `pb`, the second descriptor determines the level of symbol replacement and finally the corresponding split.

Each such file includes two parts: the upper parts specifies the articles from which the statement-proof pairs are taken, and a second part that includes the actual pairs.

### Source Data

Our dataset is based on the Mathematical REtrieval Collection corpus (MREC; Liska et al., 2011) as a source.
See our EACL 2023 paper (Section 4) for more details, or see the following page https://mir.fi.muni.cz/MREC/.

### Citation Information

> Weixian Li, Yftah Ziser and Shay B. Cohen (2023). BERT is not The Count: Learning to Match Mathematical Statements with Proofs. Proceedings of EACL.



(This description is based on the dataset card template from HuggingFace)

