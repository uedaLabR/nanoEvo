import numpy as np


def returnReadPos(cigar):

    cigseqlen = 0
    cnovN = 0
    readpos = 0
    refpos = 0
    rlist = []

    for cigaroprator, cigarlen in cigar:

        if  cigaroprator == 0:

            rlist.extend(list(range(readpos, readpos + cigarlen)))
            readpos += cigarlen
            refpos  +=  cigarlen


        elif  cigaroprator == 1: #Ins

            readpos +=  cigarlen

        elif  cigaroprator == 2: #Del

            rlist.extend([readpos] * cigarlen)
            refpos  += cigarlen



    return rlist

def convertRefToReadPos(n,cigar):

    cigseqlen = 0
    cnovN = 0
    readpos = 0
    refpos = 0
    for cigaroprator, cigarlen in cigar:

        if  cigaroprator == 0:

            if n<= refpos + cigarlen:
                return readpos + (n-refpos)

            readpos += cigarlen
            refpos  +=  cigarlen

        elif  cigaroprator == 1: #Ins

            readpos +=  cigarlen

        elif  cigaroprator == 2: #Del

            if n <= refpos + cigarlen:
                return readpos

            refpos  += cigarlen

    return cnovN
