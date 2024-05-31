import gc

import numpy as np
from numba import jit, f8, i8, b1, void
import functools
import fornewpore.resquiggle.RealignUtils as util

@jit()
def extractSig(tp, signal):
    return signal[tp[0]*5:tp[1]*5]

# @jit
from scipy import signal
def formatSig(data):

    UNIT = 16
    sg = np.around(data.astype(np.float),2)
    if len(sg) ==0:
        return np.zeros(UNIT)
    sg = signal.resample(sg, UNIT)

    return sg


def getFormatSignalMean(moveboundary, signal):


    m0 = list(moveboundary)
    m0.pop(-1)
    m1 = list(moveboundary)
    m1.pop(0)
    mIndex = zip(m0,m1)

    sig = functools.partial(extractSig, signal=signal)
    signals = list(map(sig,mIndex))
    binsignals = list(map(np.nanmean,signals))
    binsignals = np.array(binsignals).flatten()
    return binsignals


def moda(a,shift):

    if shift ==0:
        return a
    elif shift > 0:
        return a[shift:]
    else:
        a = list(a)
        amean10 = np.nanmean(a[0:10])
        for n in range(abs(shift)):
            a.insert(0, amean10)
        return np.array(a)

def predictShift(a,b):

    maxn = 0
    max = 0
    ar = None
    br = None
    #shift = -4
    for n in range(-9,9): # Fix Range to -2

        bmod = moda(b,n)
        if len(a) > len(bmod):
            a = a[0:len(bmod)]
        if len(bmod) > len(a):
            bmod = bmod[0:len(a)]

        v = np.dot(a,bmod)
        if v > max:
            max = v
            maxn = n
            ar = a
            br = bmod

    return maxn,ar,br


# def calcNormalizeScaleLMS(signalmeans,theorymean):
#
#     try:
#
#         y1 = 0
#         y2 = 0
#         y2_pow2 = 0
#         y1y2 = 0
#         n_len = len(signalmeans)
#         for n in range(len(signalmeans)):
#
#             y1 = y1 + (theorymean[n])
#             y2 = y2 + signalmeans[n]
#             y1y2 = y1y2 + y1*y2
#             y2_pow2 = y2_pow2 + y2*y2
#
#         a = [[y2_pow2, y2], [y2, n_len]]
#         ainv = np.linalg.inv(a)
#         b = np.array([[y1y2], [y1]])
#         return np.dot(ainv, b)
#
#     except:
#         return None

# def calcMean_Variance(signalmeans,theory):
#
#
#     scaleshift = calcNormalizeScaleLMS(signalmeans, theory)
#     if scaleshift is None:
#         return signal
#
#     variance = scaleshift[0][0]
#     mean = scaleshift[1][0]
#
#     return variance,mean



import numpy as np
def average_every_n(arr, n=5):

    end = len(arr) - len(arr) % n
    return np.nanmean(arr[:end].reshape(-1, n), axis=1)

import pysam
def align(ref,formatSignal,theory,cigar_str):

    maxN = len(ref)

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
                data = formatSignal[c_s:c_e]
                if len(data)==0:
                    null_index.append(n)
                    continue

                me = np.nanmean(data)
                theorylist.append(theory[n])
                signalmeanslist.append(me)

            except:

               # print('Error')
                pass


        else:

            null_index.append(n)

    # out = predictShift(signalmeanslist,theorylist)
    # print("out",out)

    # recalibrate against theoretical val
    # print(signalmeanslist)
    # print(theorylist)
    theorylist = np.array(theorylist , dtype=np.float32)
    signalmeanslist = np.array(signalmeanslist, dtype=np.float32)
    result = minimize(objective, [1, 0], args=(theorylist, signalmeanslist))
    scale, add = result.x
    # a,b = calcMean_Variance(signalmeanslist,theorylist)

    # print("a,b", a, b)

    null_index = to_interval(null_index)
    return  scale, add, null_index


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
            intervals.append((start-1,start+1) if start == end else (start-1, end+1))
            start = end = num

    intervals.append((start-1,start+1) if start == end else (start-1, end+1))
    return intervals

def theoryMeanRNA(fmerDict,genome,factor,pre,toPlusVal=True):

    means = []
    #plus strand
    rg = genome[::-1]
    for n in range(0,len(rg)-9):

       fmer = rg[n:n+9]
       if "N" in fmer:
           fmer = fmer.replace('N', 'A')
       cv = fmerDict[fmer]
       means.append(cv)
       if factor > 1:
           l = factor
           while l > 1:
               means.append(cv)
               l = l-1

    for m in range(pre):

        means.append(0)
        if factor > 1:
            l = factor
            while l > 1:
                means.append(0)
                l = l - 1

    means.reverse()

    if toPlusVal:
        means = (np.array(means)+2)*60

    return means

def quisiEqual(nuc1,nuc2):

    if nuc1 == "A" or nuc1 == "G":
        if nuc2 == "A" or nuc2 == "G":
            return True
        else:
            return False
    else:
        if nuc2 == "T" or nuc2 == "C":
            return True
        else:
            return False

def evtendsLeft(n,ref,max_entention):

    if n <=0:
        return 0

    m = n
    cnt = 0
    nuc = ref[n]
    while cnt< max_entention:
        cnt += 1
        if m <= 0:
            return 0

        m -= 1
        nucL = ref[m]
        if not quisiEqual(nuc,nucL):
            break

    return m

def evtendsRight(n,ref,max_entention):

    if n <= len(ref):
        return n
    m = n
    cnt = 0
    nuc = ref[n]
    while cnt < max_entention:
        cnt += 1
        m += 1
        if m >= len(ref):
            return len(ref)-1
        nucR = ref[m]
        if not quisiEqual(nuc,nucR):
            break

    return m

def merge_intervals(intervals):

    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for i in range(1, len(intervals)):
        current_start, current_end = intervals[i]
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged

def entendInterval(ref,null_index):

    max_entention = 2
    retlist = []
    for start,end in null_index:

        s = evtendsLeft(start,ref,max_entention)
        e = evtendsRight(end,ref,max_entention)
        retlist.append((s,e))

    return merge_intervals(retlist)

def formatPath(path):

    intervals = {}
    for start, end in path:
        if start not in intervals:
            intervals[start] = (end, end)
        else:
            min_val, max_val = intervals[start]
            intervals[start] = (min(min_val, end), max(max_val, end))

    # print(intervals.keys())
    return list(intervals.values())

def norm(ary):

    max_val = max(ary)
    min_val = min(ary)
    if max_val == min_val:
        return np.zeros_like(ary)

    normalized = (ary - min_val) / (max_val - min_val)
    return normalized

#for debug purpose
def copyTheory(theory,s,e,theoryDisp):

    UNIT = 16
    for n in range(s,e):

        t = theory[n]
        for m in range(UNIT):
            theoryDisp[((n*UNIT)+m)] = t



from numba import jit, f8, i8, b1, void
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from copy import copy
import numba


def redivide_interval(start, end, n):


    total_length = end - start
    base_interval_length = total_length // n
    remainder = total_length % n

    return [(start + (base_interval_length + 1) * i,
             min(start + (base_interval_length + 1) * (i + 1), end) if i < remainder
             else start + base_interval_length * (i + 1) + remainder)
            for i in range(n)]


def smoothIv(ref,s,itvls):

    try:
        return _smoothIv(ref, s, itvls)
    except:
        return itvls

@numba.jit(nopython=True)
def _smoothIv(ref,s,itvls):

    ret = []
    l = 0
    dmax = 0
    lmax = 0
    for sidx, eidx in itvls:

        d = (eidx-sidx)
        if d > dmax:
            dmax = d
            lmax = l

        l += 1

    nuc = ref[s+lmax]
    s0 = lmax
    while s0>0:
        s0 = s0 - 1
        nuc2 = ref[s0 + lmax]
        if nuc!=nuc2:
            break
    e0 = lmax
    while e0 < len(itvls)-1:
        e0 = e0 + 1
        nuc2 = ref[e0 + lmax]
        if nuc!=nuc2:
            break

    start_index =  itvls[s0][0]
    end_index = itvls[e0][1]
    n = e0-s0
    if n > 1 and (end_index-start_index) > n:
        reiv = redivide_interval(start_index, end_index, n+1)
    else:
        return itvls

    # print(s0,e0,start_index,end_index,n,reiv)
    l = 0
    recnt = 0
    for v in itvls:

        if l<s0 or l>e0:

            ret.append(v)

        else:

            ret.append(reiv[recnt])
            recnt+=1

        l += 1

    return ret


def mindGap(ref,theory,null_index_interval,recalibsignal,intervals):

    for s,e in null_index_interval:

        #
        if s < 0:
            s = 0
        if e >= len(intervals)-1:
            e = len(intervals) - 1

        start = intervals[s]
        end = intervals[e]
        if start == None or end == None:
            continue


        theoryPart = theory[s:e]
        # theoryPart = norm(theoryPart)
        partSignal = recalibsignal[start[0]:end[1]]
        # actualpart = norm(partSignal)

        distance, path = fastdtw(theoryPart, partSignal, dist=euclidean)
        itvls = formatPath(path)

        smoothInterval = smoothIv(ref,s,itvls)
        # smoothInterval = itvls

        l=0
        for sidx, eidx in smoothInterval:

            si = start[0]+sidx
            ei = start[0]+eidx
            intervals[s+l] = (si,ei)
            l+=1


    return intervals


import fornewpore.resquiggle.SigAlgin as SigAlgin

#slow needto think about how to do
def align2(ref, recalbsignal,moveboundary,cigar_str):

    UNIT_5 = 5

    a = pysam.AlignedSegment()
    a.cigarstring = cigar_str

    intervals = []
    nullindex = []

    allreadpos = util.returnReadPos(a.cigar)
    for n in range(len(ref)-1):

        if (n + 1) * 2 <= len(allreadpos) - 1:
            c_s = allreadpos[n * 2]
            c_e = allreadpos[(n + 1) * 2]
        else:
            continue


        # take half
        if (c_e - c_s) == 1:

            c_s = c_s // 2
            start = moveboundary[c_s] * UNIT_5
            end = moveboundary[c_s + 1] * UNIT_5
            half = (start + end) // 2
            if c_s%2 == 0:
                #first harf
                end = half
            else:
                start =  half

        else:

            c_s = c_s//2
            c_e = c_e//2
            start = moveboundary[c_s] * UNIT_5
            end = moveboundary[c_e] * UNIT_5

        if end > start:

            intervals.append((start,end))

        else:

            # print(c_s,c_e,end,start)
            nullindex.append(n)
            intervals.append(None)

    nullindex = to_interval(nullindex)
    return intervals,nullindex

def toList(itvl):

    l = list(map(lambda x: 0 if x is None else x[1]-x[0], itvl))
    return l


import ruptures as rpt
import numpy as np
import time
import gc
def recalibandrealign(reads,fmerdict):

    print("start recalibandrealign")
    t1 = time.time()
    data = []
    cnt = 0
    for read in reads:
        #process
        ret = _recalibandrealign(read,fmerdict)
        data.append(ret)

        if cnt % 100 == 0:
            t2 = time.time()
            print("time elaspe", t2 - t1, len(data))
        cnt+=1

    t2 = time.time()
    gc.collect()
    print("finish time elaspe", t2-t1,len(data))
    return data

from functools import partial
UNIT = 16
def genFormatSignal(itvl,resignal):

    if itvl is None:
        return np.zeros(UNIT)
    s = itvl[0]
    e = itvl[1]
    sub = resignal[s:e+1]
    return signal.resample(sub, UNIT)

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

def toInterval(ref,tracebackPath,moveboundary,granular_level):


    UNIT_5 = 5
    intervals = []
    nullindex = []
    allreadpos = toReadIdx(tracebackPath)
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

import numpy as np
from scipy.optimize import minimize
import numba

def normDiff(a, b):

    ary, scale, add  = optimize_difference(a, b)
    ary = np.clip(ary,-120,120) + 120
    return ary

@numba.njit
def scaled_diff(a, b, scale, add):
    return a - (b * scale + add)

@numba.njit
def objective(params, a, b):
    scale, add = params
    diff = scaled_diff(a, b, scale, add)
    return np.sum(np.abs(diff))

def optimize_difference(a, b):
    result = minimize(objective, [1, 0], args=(a, b))
    scale, add = result.x
    diff = scaled_diff(a, b, scale, add)
    return diff, scale, add


import adjust.AdjustUtils as au
def _recalibandrealign(nread,fmerdict):


    factor = 1

    ss = nread.q_st
    if ss < 0:
        ss = 0
    ee = nread.q_en
    if ee >= len(nread.traceboundary)-1:
        ee = len(nread.traceboundary)-1

    start = nread.traceboundary[ss]
    end = nread.traceboundary[ee]

    startT = time.time()

    moveboundary = nread.traceboundary[ss:ee]
    moveboundary = np.array(moveboundary)
    moveboundary = moveboundary - moveboundary[0]

    signal = nread.signal[start * 5:end * 5]
    signal = np.clip(signal, 0, 300)

    pre = 4
    theory = theoryMeanRNA(fmerdict, nread.refgenome, factor,pre)
    meanSignal = getFormatSignalMean(moveboundary, signal)
    # time_n1 = time.time()

    ref = nread.refgenome
    # print(nread.q_st,nread.q_en,nread.r_st,nread.r_en)
    scale, add, null_index = align(ref, meanSignal,theory, nread.cigar_str)
    recalbsignal = (signal*scale)+add
    #

    # time_n2 = time.time()

    granular_level = 3
    theory_for_algin = theoryMeanRNA(fmerdict, nread.refgenome, granular_level, pre)
    readseq = nread.sequence[nread.q_st:nread.q_en]
    theoryRead = theoryMeanRNA(fmerdict, readseq, granular_level, pre)

    # time_n3 = time.time()

    newcigar, tracebackPath = SigAlgin.viterbi(theory_for_algin, theoryRead, nread.cigar_str, granular_level)
    time_n4 = time.time()

    intervals,nullindex = toInterval(ref,tracebackPath,moveboundary,granular_level)

    time_n5 = time.time()
    if len(nullindex) > 0:
        null_index_interval = entendInterval(ref,nullindex)
        intervals  = mindGap(nread.refgenome,theory,null_index_interval,recalbsignal,intervals)
    #
    # time_n6 = time.time()

    #format signal
    gapresolve = au.getFormatSignal(intervals, recalbsignal, toBytes=False)

    # time_n7 = time.time()

    meansignal = au.getMeanSignal(intervals, recalbsignal)
    minlen = min(len(meansignal), len(theory))
    diffsig = normDiff(meansignal[0:minlen], theory[0:minlen])

    duration = list(map(lambda x: 0 if x is None else x[1] - x[0], intervals))


    #recalbsignal,gapresolve, intervals
    nread.signal = recalbsignal
    nread.signalboundary  = toList(intervals)
    nread.formatSignal = gapresolve
    nread.diffsignal = diffsig
    nread.duration = duration

    end = time.time()
    # print("time",len(ref),end-startT, time_n1-startT,time_n2-time_n1,time_n3-time_n2,time_n4-time_n3,time_n5-time_n4,time_n6-time_n5,time_n7-time_n6,end-time_n7)
    tp = nread.toTupleForTrain()
    nread = None
    del nread
    return tp