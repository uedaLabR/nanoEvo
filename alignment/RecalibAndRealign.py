import numpy as np
from numba import jit, f8, i8, b1, void
import functools


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


def getFormatSignal(moveboundary, signal):


    m0 = list(moveboundary)
    m0.pop(-1)
    m1 = list(moveboundary)
    m1.pop(0)
    mIndex = zip(m0,m1)

    sig = functools.partial(extractSig, signal=signal)
    signals = list(map(sig,mIndex))
    binsignals = list(map(formatSig,signals))
    binsignals = np.array(binsignals).flatten()
    return binsignals




def moda(a,shift):

    if shift ==0:
        return a
    elif shift > 0:
        return a[shift:]
    else:
        a = list(a)
        amean10 = np.mean(a[0:10])
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


def calcNormalizeScaleLMS(signalmeans,theorymean):

    try:

        y1 = 0
        y2 = 0
        y2_pow2 = 0
        y1y2 = 0
        n_len = len(signalmeans)
        for n in range(len(signalmeans)):

            y1 = y1 + (theorymean[n])
            y2 = y2 + signalmeans[n]
            y1y2 = y1y2 + y1*y2
            y2_pow2 = y2_pow2 + y2*y2

        a = [[y2_pow2, y2], [y2, n_len]]
        ainv = np.linalg.inv(a)
        b = np.array([[y1y2], [y1]])
        return np.dot(ainv, b)

    except:
        return None

def calcMean_Variance(signalmeans,theory):


    scaleshift = calcNormalizeScaleLMS(signalmeans, theory)
    if scaleshift is None:
        return signal

    variance = scaleshift[0][0]
    mean = scaleshift[1][0]

    return variance,mean



import numpy as np
def average_every_n(arr, n=5):

    end = len(arr) - len(arr) % n
    return np.mean(arr[:end].reshape(-1, n), axis=1)

import pysam
def align(ref,formatSignal,theory,cigar_str,UNIT=16):

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

import numba
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

    ret = n-max_entention
    if ret <= 0:
        ret = 0
    return ret
    # m = n
    # cnt = 0
    # nuc = ref[n]
    # while cnt< max_entention:
    #     cnt += 1
    #     if m <= 0:
    #         return 0
    #
    #     m -= 1
    #     nucL = ref[m]
    #     # if not quisiEqual(nuc,nucL):
    #     #     break
    #
    # return m

def evtendsRight(n,ref,max_entention):

    ret = n+max_entention
    if ret >= len(ref):
        ret = len(ref)-1
    return ret
    # m = n
    # cnt = 0
    # nuc = ref[n]
    # while cnt < max_entention:
    #     cnt += 1
    #     m += 1
    #     if m >= len(ref):
    #         return len(ref)-1
    #     nucR = ref[m]
    #     # if not quisiEqual(nuc,nucR):
    #     #     break
    #
    # return m

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

def entendInterval(ref,null_index,intervals):

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




from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from copy import copy
def _mindGap(ref,theory,recalbsignal,moveboundary,null_index_interval,alginSignal,cigar_str):

    UNIT = 16
    gapresolve = copy(alginSignal)
    shoten_signal = average_every_n(recalbsignal)

    a = pysam.AlignedSegment()
    a.cigarstring = cigar_str

    # for debug
    theoryDisp = np.zeros(len(ref) * UNIT)
    rawDisp = np.zeros(len(recalbsignal))

    for s,e in null_index_interval:

        theoryPart = theory[s:e]
        copyTheory(theory,s,e,theoryDisp)
        theoryPart = norm(theoryPart)

        s_read = util.convertRefToReadPos(s,a.cigar)
        e_read = util.convertRefToReadPos(e,a.cigar)
        sIdx = moveboundary[s_read]
        eIdx = moveboundary[e_read]
        actualpart = norm(shoten_signal[sIdx:eIdx])
        #change point is so frequent, so we down sample if there enough signal
        downsampled = False
        if (len(actualpart)>=len(theoryPart)*2):
            actualpart = average_every_n(actualpart,n=2)
            downsampled = True

        distance, path = fastdtw(theoryPart, actualpart, dist=euclidean)
        itvls = formatPath(path)

        l = 0
        unit = 5
        sidx_raw = sIdx * unit
        eidx_raw = eIdx * unit
        partSignal = recalbsignal[sidx_raw:eidx_raw]

        #debug
        rawDisp[sidx_raw:eidx_raw] = partSignal

        for sidx,eidx in itvls:

            if downsampled:
                sidx = sidx*2
                eidx = eidx*2

            onenucSignal = partSignal[sidx * unit: (eidx + 1) * unit]
            onenucSignal = signal.resample(onenucSignal, UNIT)
            gapresolve[(s+l)*UNIT:(s+l+1)*UNIT] = onenucSignal

            l+=1


    return gapresolve, theoryDisp, rawDisp

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

import numba
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
    if n > 1:
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


def mindGap(ref,newalgin,theory,null_index_interval,recalibsignal,intervals):

    UNIT = 16
    gapresolve = copy(newalgin)
    takesignal = np.zeros(len(recalibsignal))
    print("null",null_index_interval)
    for s,e in null_index_interval:

        #
        start = intervals[s]
        end = intervals[e]
        if start == None or end == None:
            continue


        theoryPart = theory[s:e]
        # theoryPart = norm(theoryPart)
        partSignal = recalibsignal[start[0]:end[1]]
        print(theoryPart)
        print(partSignal)

        takesignal[start[0]:end[1]] = recalibsignal[start[0]:end[1]]
        # actualpart = norm(partSignal)

        distance, path = fastdtw(theoryPart, partSignal, dist=euclidean)
        itvls = formatPath(path)
        print("itvl", itvls)
        smoothInterval = smoothIv(ref,s,itvls)
        print("smoothInterval",smoothInterval)
        # smoothInterval = itvls
        l=0
        for sidx, eidx in smoothInterval:

            # print("interval",sidx, eidx)
            onenucSignal = partSignal[sidx: (eidx + 1)]
            onenucSignal = signal.resample(onenucSignal, UNIT)
            print(onenucSignal,type(onenucSignal),type(gapresolve))
            # gapresolve[(s+l)*UNIT:(s+l+1)*UNIT] = onenucSignal[0:UNIT]
            si = start[0]+sidx
            ei = start[0]+eidx
            intervals[s+l] = (si,ei+1)
            l+=1


    return gapresolve, intervals,takesignal


def alginTheoryCurrent(theory,theoryRead,read,cigar_str):

    return SigAlgin.viterbi(theory, theoryRead,read,cigar_str)

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


from functools import partial
UNIT = 16
def genFormatSignal(itvl,resignal):

    if itvl is None:
        return np.zeros(UNIT)
    s = itvl[0]
    e = itvl[1]
    sub = resignal[s:e+1]
    return signal.resample(sub, UNIT)

import numpy as np
from scipy.optimize import minimize
import numba

def normDiff(a, b):

    ary, scale, add = optimize_difference(a, b)
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

import numpy as np
import itertools

import time
def recalibandrealign(nread,fmerdict):


    factor = 1
    start = nread.traceboundary[nread.q_st]
    end = nread.traceboundary[nread.q_en]
    moveboundary = nread.traceboundary[nread.q_st:nread.q_en]
    moveboundary = np.array(moveboundary)
    moveboundary = moveboundary - moveboundary[0]

    signal = nread.signal[start * 5:end * 5]
    signal = np.clip(signal, 0, 300)

    pre = 4
    theory = theoryMeanRNA(fmerdict, nread.refgenome, factor,pre)
    theoryExtend = theoryMeanRNA(fmerdict, nread.refgenome, 16,pre)


    formatSignal = getFormatSignal(moveboundary, signal)
    ref = nread.refgenome
    # print(nread.q_st,nread.q_en,nread.r_st,nread.r_en)
    alginSignal, a,b, null_index = align(ref, formatSignal,theory, nread.cigar_str)
    recalbsignal = (signal*a)+b
    #
    granular_level = 3
    theory_for_algin = theoryMeanRNA(fmerdict, nread.refgenome, granular_level, pre)
    readseq = nread.sequence[nread.q_st:nread.q_en]
    theoryRead = theoryMeanRNA(fmerdict, readseq, granular_level, pre)

    newcigar,tracebackPath = SigAlgin.viterbi(theory_for_algin, theoryRead, nread.cigar_str,granular_level)
    intervals,nullindex = toInterval(ref,tracebackPath,moveboundary,granular_level)


    newalgin = au.getFormatSignal(intervals, recalbsignal, toBytes=False)

    print("newalgin",type(newalgin),newalgin)

    null_index_interval = entendInterval(ref,nullindex,intervals)
    print(newcigar)
    print("null_interval",null_index_interval)
    print(intervals)
    gapresolve0, intervals,takesignal  = mindGap(nread.refgenome,newalgin,theory,null_index_interval,recalbsignal,intervals)
    t1 = time.time()
    gapresolve = au.getFormatSignal(intervals, recalbsignal, toBytes=False)
    meansignal = au.getMeanSignal(intervals, recalbsignal)
    minlen = min(len(meansignal),len(theory))
    diffsig = normDiff(meansignal[0:minlen], theory[0:minlen])
    duration = list(map(lambda x:0 if x is None else x[1]-x[0],intervals))



    return recalbsignal,moveboundary,newalgin,theoryExtend,formatSignal,alginSignal,gapresolve,theory_for_algin,theoryRead,intervals,theory,diffsig,duration,meansignal