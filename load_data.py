#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assume a small dataset! 

text files --> input vectors in character space.
dimensionality = number of characters in the input text, caps not included.
"""

import numpy as np

shakespeare_source = "data/karpathy_shakespeare_sample.txt"
shakespeare_text = open(shakespeare_source).read().lower()

cstore = list(set(shakespeare_text))
char_to_int = dict([(cstore[i], i) for i in range(len(cstore))])
int_to_char=cstore

def to_int(s):
    return [char_to_int[c] for c in s]

def to_vecs(int_list):
    """Returns the whole dataset in one go, so be careful."""
    vecs = np.zeros((len(cstore), len(int_list)))
    for i in range(len(int_list)):
        vecs[int_list[i], i]=1
    return vecs

shakespeare_vecs = to_vecs(to_int(shakespeare_text))
