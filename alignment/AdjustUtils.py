import numpy as np
from numba import jit
import numpy as np
from numba import jit, f8, i8, b1, void
import functools
import numpy as np
import numba

UNIT = 16
@jit(nopython=True,nogil=True)
def extractSig(tp, signal):

    if tp is None:
        return np.zeros(UNIT,dtype=np.uint8)

    data = signal[tp[0]:tp[1]]
    if len(data) == 0:
        data = signal[tp[0]:tp[0]+1]

    return formatSig(data)

@jit(nopython=True,nogil=True)
def extractSigMean(tp, signal):

    if tp is None:
        return 0

    data = signal[tp[0]:tp[1]]
    if len(data) == 0:
        data = signal[tp[0]:tp[0]+1]
    return np.nanmean(data)

def formatSig2(data):

    UNIT = 16
    sg = np.around(data.astype(np.float),2)
    if len(sg) ==0:
        return np.zeros(UNIT)
    sg = signal.resample(sg, UNIT)

    return sg
@jit()
def extractSig2(tp, signal):
    return signal[tp[0]*5:tp[1]*5]

def getBinAveSignal(moveboundary, signal):


    m0 = list(moveboundary)
    m0.pop(-1)
    m1 = list(moveboundary)
    m1.pop(0)
    mIndex = zip(m0,m1)

    sig = functools.partial(extractSig2, signal=signal)
    signals = list(map(sig,mIndex))
    binsignals = list(map(formatSig2,signals))
    binsignals = np.array(binsignals).flatten()
    return binsignals

import alignment.RealignUtils as util
from scipy import signal
@numba.njit
def objective(params, a, b):
    scale, add = params
    diff = scaled_diff(a, b, scale, add)
    return np.sum(np.abs(diff))
@numba.njit
def scaled_diff(a, b, scale, add):
    return a - (b * scale + add)

from scipy.optimize import minimize
def alignForRecalib(ref,formatSignal,theory,cigar_str,UNIT=16):

    maxN = len(ref)
    print("lenformat",len(formatSignal),len(formatSignal)/16,maxN,len(ref))
    print(cigar_str)

    ary = np.zeros(maxN*UNIT)
    #
    a = pysam.AlignedSegment()
    a.cigarstring = cigar_str
    theorylist = []
    signalmeanslist = []
    null_index = []
    #Copy move value
    for n in range(maxN):

        c_s = util.convertRefToReadPos(n,a.cigar)
        c_e = util.convertRefToReadPos(n+1, a.cigar)
        diff = c_e - c_s
        if diff > 0:

            try:

                unit_interval = formatSignal[(c_s * UNIT):((c_e) * UNIT)]
                if len(unit_interval) > UNIT:
                    unit_interval = signal.resample(unit_interval, UNIT)
                ary[(n * UNIT):((n + 1) * UNIT)] = unit_interval
                theorylist.append(theory[n])
                signalmeanslist.append(np.mean(unit_interval))

            except:

               print('Error')


        else:

            null_index.append(n)

    # out = predictShift(signalmeanslist,theorylist)
    # print("out",out)

    # recalibrate against theoretical val
    # a,b = calcMean_Variance(signalmeanslist,theorylist)
    theorylist = np.array(theorylist , dtype=np.float32)
    signalmeanslist = np.array(signalmeanslist, dtype=np.float32)
    result = minimize(objective, [1, 0], args=(theorylist, signalmeanslist))
    scale, add = result.x

    print("scale,add", scale, add)
    print(np.mean(ary))
    ary = (ary * scale) + add
    print(np.mean(ary))
    null_index = to_interval(null_index)
    return ary, scale, add, null_index

def getFormatSignal(traceintervals, signal,toBytes=True):

    exsig = functools.partial(extractSig, signal=signal)
    binsignals = np.array(list(map(exsig,traceintervals)))
    binsignals = np.concatenate(binsignals,axis=None)
    if toBytes:
        binsignals = binsignals.tobytes()
    return binsignals

def getMeanSignal(traceintervals, signal):

    sigmean = functools.partial(extractSigMean, signal=signal)
    avesignals = np.array(list(map(sigmean,traceintervals)))
    return avesignals


from scipy import interpolate
def resample(sg,UNIT):

    num = len(sg)
    x = np.linspace(0, UNIT-1, num)
    f = interpolate.interp1d(x, sg)
    xx = np.arange(UNIT)
    yy = f(xx)
    return yy

from numba import njit, prange
@numba.jit(nopython=True, nogil=True)
def linear_interp(x, y, x_new):
    y_new = np.empty_like(x_new)
    for i in prange(x_new.shape[0]):
        # Find the index of the closest value in x to x_new[i]
        idx = np.searchsorted(x, x_new[i])

        # Handle edge cases
        if idx == 0:
            y_new[i] = y[0]
        elif idx == len(x):
            y_new[i] = y[-1]
        else:
            # Perform linear interpolation
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            y_new[i] = y0 + (y1 - y0) * (x_new[i] - x0) / (x1 - x0)
    return y_new


@njit(parallel=True)
def linear_interp_parallel(x, y, x_new):
    y_new = np.empty_like(x_new)
    for i in prange(x_new.shape[0]):
        # Find the index of the closest value in x to x_new[i]
        idx = np.searchsorted(x, x_new[i])

        # Handle edge cases
        if idx == 0:
            y_new[i] = y[0]
        elif idx == len(x):
            y_new[i] = y[-1]
        else:
            # Perform linear interpolation
            x0, x1 = x[idx - 1], x[idx]
            y0, y1 = y[idx - 1], y[idx]
            y_new[i] = y0 + (y1 - y0) * (x_new[i] - x0) / (x1 - x0)
    return y_new

@numba.jit(nopython=True, nogil=True)
def resample_array(arr, target_length):
    """
    Resample the given array to the specified length.

    :param arr: The array to be resampled.
    :param target_length: The target length of the array.
    :return: The resampled array.
    """
    original_length = len(arr)
    original_indices = np.linspace(0, original_length - 1, original_length)
    target_indices = np.linspace(0, original_length - 1, target_length)
    resampled_array = linear_interp(original_indices, arr, target_indices)
    # resampled_array = linear_interp_parallel(original_indices, arr, target_indices)
    return resampled_array

from numba import jit, f8, i8, b1, void
from numba import jit
import numpy as np
import os
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
import numba
@jit('f4[:](f4[:], i4)', nogil=True)
def downsample_array(arr, target_length):
    """
    Downsample the given array to the specified length by taking the average of each segment.

    :param arr: The array to be downsampled.
    :param target_length: The target length of the array.
    :return: The downsampled array.
    """
    original_length = len(arr)
    if target_length >= original_length:
        # If the target length is greater than or equal to the original length, return the original array
        return arr
    else:
        # Calculate the size of each segment to be averaged
        segment_size = original_length / target_length
        downsampled_arr = np.empty(target_length, dtype=np.float32)
        for i in range(target_length):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            # Compute the average of the current segment
            downsampled_arr[i] = np.mean(arr[start_idx:end_idx])
        return downsampled_arr



@jit('u1[:](f4[:])',nogil=True)
# @jit(nopython=True,nogil=True)
def formatSig(data):

    # UNIT = 10
    # lst = np.split(data, UNIT)
    # return list(map(np.mean,lst))
    UNIT = 16
    if data is None or len(data)==0:
        return np.zeros(UNIT,dtype=np.uint8)

    if len(data) >= UNIT*2:
        data = downsample_array(data, UNIT*2)
        # print("doensample with size",len(data))

    # sg = signal.resample(sg, UNIT)
    sg = resample_array(data,UNIT)

    #format
    low_limit = 30
    high_limit = 300
    # sg = np.clip(sg, low_limit, high_limit)
    sg = np.where(sg < low_limit, low_limit, sg)
    sg = np.where(sg > high_limit, high_limit, sg)
    sg = sg.astype(np.float32)
    sg = ((sg - low_limit) / (high_limit - low_limit)) * 255
    sg = np.rint(sg).astype(np.uint8)

    return sg



def format(sg):
    low_limit = 30
    high_limit = 250

    # sg = np.clip(sg, low_limit, high_limit)
    sg = np.where(sg < low_limit, low_limit, sg)
    sg = np.where(sg > high_limit, high_limit, sg)

    sg = ((sg - low_limit) / (high_limit - low_limit)) * 255
    sg = np.around(sg).astype(np.uint8)

    return sg


@jit
def intervalToAbsolute(intervals):

    ret = []
    cnt = 0
    sum = 0
    b4 = 0
    UNIT = 10
    for n in intervals:

        n = n *  UNIT
        if cnt == 0:
            ret.append((b4, n))
            sum = sum + n
            cnt += 1
            continue
        # else
        b4 = sum
        sum = sum + n
        ret.append((b4, sum))

    return ret

unit = 10
import pysam
def getrelpos(cigar,pos):
    a = pysam.AlignedSegment()
    a.cigarstring = cigar
    refpos = 0
    relpos = 0
    for cigaroprator, cigarlen in a.cigar:

        if cigaroprator == 0:  # match

            if refpos + cigarlen > pos:
                return relpos + (pos - refpos)

            relpos = relpos + cigarlen
            refpos = refpos + cigarlen

        elif cigaroprator == 2:  # Del
            refpos = refpos + cigarlen
        elif cigaroprator == 1 or cigaroprator == 3:  # Ins or N
            relpos = relpos + cigarlen

    return 0

def getQstQen(cigar):

    a = pysam.AlignedSegment()
    a.cigarstring = cigar
    cglist = a.cigar
    S = 4
    q_st = 0
    cigaroprator, cigarlen = cglist[0]
    if cigaroprator == S:
        q_st = cigarlen

    q_en = 0
    cigaroprator, cigarlen = cglist[-1]
    if cigaroprator == S:
        q_en = cigarlen


    return q_st,q_en

def toReadIdx(tracebackPath):

    d = {}
    for n,m in tracebackPath:
        d[n] = m


    max = tracebackPath[-1][0]
    ret = np.zeros(max, dtype = int)
    prev = 0
    for n in range(max):

        if n in d:
            ret[n] = d[n]
            prev = d[n]
        else:
            ret[n] = prev
    return ret

def to_interval(lst):
    if not lst:
        return []

    lst = sorted(lst)
    intervals = []
    start = end = lst[0]

    for num in lst[1:]:
        if num == end + 1:
            end = num
        else:
            intervals.append((start,start) if start == end else (start, end))
            start = end = num

    intervals.append((start,start) if start == end else (start, end))
    return intervals

def toInterval(ref,tracebackPath,moveboundary,granular_level):


    UNIT_5 = 5
    intervals = []
    nullindex = []
    allreadpos = toReadIdx(tracebackPath)
    print("allpos", ""+str(allreadpos))
    divunit = granular_level
    for n in range(len(ref)):


        if (n + 1) * divunit <= len(allreadpos) - 1:
            c_s = allreadpos[n * divunit]
            c_e = allreadpos[(n + 1) * divunit]
        else:
            break

        start = 0
        end = 0
        d = c_e - c_s
        if (d > 0) :

            c_s0 = c_s // divunit
            modS = c_s % divunit
            start = moveboundary[c_s0] * UNIT_5
            if modS > 0:
                itvaldiv = ((moveboundary[c_s0+1]-moveboundary[c_s0]) * UNIT_5) / divunit
                start = start + int(itvaldiv*modS)

            c_e0 = c_e // divunit
            modE = c_e % divunit
            end = moveboundary[c_e0] * UNIT_5
            if modE > 0:
                itvaldiv = ((moveboundary[c_e0+1] - moveboundary[c_e0]) * UNIT_5) / divunit
                end = end + int(itvaldiv*modE)



        if end > start:

            intervals.append((start,end))

        else:

            # print(c_s,c_e,end,start)
            nullindex.append(n)
            intervals.append(None)

    nullindex = to_interval(nullindex)
    return intervals,nullindex


def getMeans(traceintervals, signal):

    traceintervals = traceintervals[0:-5]
    try:
        if len(traceintervals) > 0:
            sig = functools.partial(extractSigMean, signal=signal)
            meansignals = list(map(sig, traceintervals))
            return meansignals
    except:
        return []



def theoryMean(fmerDict,lgenome,strand,dna):

    if dna:
        return theoryMeanDNA(fmerDict,lgenome,strand)

    else:

        return theoryMeanRNA(fmerDict,lgenome,strand)


def theoryMeanDNA(fmerDict,lgenome,strand):

    means = []
    rg = lgenome
    for n in range(0, len(rg) - 6):
        fmer = rg[n:n + 6]
        if "N" in fmer:
            fmer = fmer.replace('N', 'A')
        cv = fmerDict[fmer]
        means.append(cv)
    return means

def extract5merMean(n,rg,fmerDict):

    fmer = rg[n:n + 5]
    if "N" in fmer:
        fmer = fmer.replace('N', 'A')
    cv = fmerDict[fmer]
    return cv

# def theoryMeanRNA(fmerDict,lgenome,strand):
#
#     if strand == "-":
#         rg = lgenome
#     else:
#         rg = lgenome[::-1]
#
#     l1 = list(range(0, len(rg) - 5))
#     ext = functools.partial(extract5merMean, rg=rg, fmerDict=fmerDict)
#     means = list(map(ext, l1))
#     if strand != "-":
#         means.reverse()
#
#     return means


def theoryMeanRNA(fmerDict,lgenome,strand):

    means = []
    if strand == "-":

        rg = lgenome
        for n in range(0, len(rg) - 5):
            fmer = rg[n:n + 5]
            if "N" in fmer:
                fmer = fmer.replace('N', 'A')
            cv = fmerDict[fmer]
            means.append(cv)

    else:

        #plus strand
        rg = lgenome[::-1]
        for n in range(0,len(rg)-5):

           fmer = rg[n:n+5]
           if "N" in fmer:
               fmer = fmer.replace('N', 'A')
           cv = fmerDict[fmer]
           means.append(cv)

        means.reverse()

    return means

def getCurrentDict(fmercurrent):
    a = {}
    with open(fmercurrent) as f:
        for line in f:
            if "#" not in line:
                data = line.split()
                if data[1] =="level_mean":
                    pass
                else:
                    a[data[0]] = float(data[1])

    return a
def countmatch(g,r):

    cnt = 0
    for m in range(5):
        if g[m] == r[m]:
            cnt += 1
    return cnt


def findMismatchInterval(gseq,readseq,margin):

    ml = []
    l = min(len(gseq),len(readseq))
    for n in range(l-5):
        m = countmatch(gseq[n:n+5],readseq[n:n+5])
        if m <= 2:
            ml.append(n)
        elif readseq[n+2] =='-':
            ml.append(n)
        elif gseq[n + 2] == '-':
            ml.append(n)

    return interval_extract(ml,margin)


def interval_extract(list,margin):
    if len(list) ==0:
        return []

    list = sorted(set(list))
    range_start = previous_number = list[0]

    for number in list[1:]:
        if number - previous_number < margin:
            previous_number = number
        else:
            yield [range_start+1, previous_number+margin]
            range_start = previous_number = number
    yield [range_start+1, previous_number+margin]

# gseq =     "CAAGAAGAAGAAGGACGCTGGAAAGTCGGCCAAGAAAGACAAAGACCCAGTGAACAAATCCGGGGGCAAGGCCAAAAAGAAGAAGTGGTCCAAAGGCAAAGTTCGGGACAAGCTCAATAACTTAGTCTTGTTTGACAAAGCTACCTATGATAAACTCTGTAAGGAAGTTCCCAACTATAAACTTATAACCCCAGCTGTGGTCTCTGAGAGACTGAAGATTCGAGGCTCCCTGGCCAGGGCAGCCCTTCAGGAGCTCCTTAGTAAAGGACTTATCAAACTGGTTTCAAAGCACAGAGCTCAAGTAATTTACACCAGAAATACCAAGGGTGGAGATGCTCCAGCTGCTGGTGAAGATGCATGAATAGGTCCAACCAGCTGTACATTTGGAAAAATAAAACT"
# traceseq = "TAAGAAGAAGAAGGACGCTGGAAAGTCGGCCAAGAAAGACAAAGACCCAGTGAACAAATCCGGGGGCAAGGCCAAAAAGAAGAAGTGGTCCAAAGGCAAAGTTCGGGACAAGCTCAATAACTTAGTTTCGTTTGACAAAGCTACCTATGATAAACTCTGTTGGGAAGTTCCCAACTATAAACTTATAACCCCAGCTGTGGTCTTTGAGAGACTGAAGATTCGAA---CCCTGGCCAGGGCAGCCCTTCAGGAGCTCCTTAGTAAAGGACCTCCCAAACTGGTTTCAAAGCACAGAGCTCAAGTAATTCATACCAGAAATTCTAAGGGTGGAGATGCTCCAGCTGCTGGTGAAGATGCATGA-TAGGTCCAACCAGCTGTACATTTGGAAAAATAAAAC"

gseq =     "TGCATGAATAGGTCCAACCAGCTGTACATTTGGAAAAATAAAAC"
traceseq = "TGCATGA-TAGGTCCAACCAGCTGTACATTTGGAAAAATAAAAC"


itvl = findMismatchInterval(gseq, traceseq,4)

for i in itvl:
    print(i)

print(itvl)

