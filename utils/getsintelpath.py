# SHOULD BE IN UTILS
from pathlib import Path
from itertools import chain
import re


# getid = lambda x: [*map(int, re.findall(r'\d+', x.name))][0]
# getdirdata = lambda x: [*map(lambda y: y.as_posix(), sorted(x.glob('*.png'), key=getid))]
# alldata = chain.from_iterable(map(chunker, map(getdirdata, Path(sintelroot).glob('*'))))


def getSintelPairFrame(root):
    def chunker(lst):
        return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]
    def getid(x):
        return [*map(int, re.findall(r'\d+', x.name))][0]
    def getdirdata(x):
        return [*map(lambda y: y.as_posix(), sorted(x.glob('*.png'), key=getid))]

    subroot = Path(root).glob('*')
    datalist = chain.from_iterable(map(chunker, map(getdirdata, subroot)))

    return [*datalist]


"""uses
sintelroot = "/data/keshav/sintel/training/final"
files = getSintelPairFrame(sintelroot)
"""
