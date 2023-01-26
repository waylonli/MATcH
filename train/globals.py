#!/usr/bin/env python3

"""Global constants"""

from lxml import etree
import datetime
import os
import logging
import sys
import glob


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

namespace = {"":"http://www.w3.org/1999/xhtml",
             "m":"http://www.w3.org/1998/Math/MathML",
             "svg":"http://www.w3.org/2000/svg",
             "dct":"http://purl.org/dc/terms/",
             "arx":"http://arxmliv.kwarc.info/ns/"}

RANDOM_SEED=42

UNIQUE="UNIQUE"

SEPARATOR="____________"

FORMULAE="math"
TEXT = "text"
BOTH = "both"

META_TAG="meta"
PAIR_TAG="pair"
STATEMENT_TAG="theorem"
PROOF_TAG="proof"
ROOT_TAG="root"

SILENT_TAGS = {"msqrt", "mover", "munder", "mmultiscripts",
               "mprescripts", "mfrac", "mroot", "msqrt",
                "msup", "msubsup", "msub"}
#PSILENT_TAGS = {"{{{}}}{}".format(namespace["m"], tag) for tag in SILENT_TAGS}

#VARIABLE_TAG="mi"
#CONTENT_V_TAG="ci"


def now():
    return str(datetime.datetime.now()).replace(" ", "_")

def get_class(element):
    if "class" in element.attrib:
        return element.attrib["class"]
    return None

def parse_xhtmlfile(f):
    tree = etree.parse(f)
    return tree

rightnow = now()


# search = "*"
# dirs = glob.glob(search)
# while not any([item.endswith("mathml") for item in dirs]):
#     search = f"../{search}"
#     dirs = glob.glob(search)
#
# dirs = [item for item in dirs if item.endswith("mathml")][0]

xslt_root = parse_xhtmlfile(f"./content-to-presentation.xsl")
# xslt_root = parse_xhtmlfile(f"./tasks/theory-proof-matching/maximin_ori/content-to-presentation.xsl")

transform = etree.XSLT(xslt_root)




