import glob

import numpy as np
import pysam
from ont_fast5_api.fast5_interface import get_fast5_file


def getMoveDict(bamf):

    movedict = {}
    bamfile = pysam.AlignmentFile(bamf, 'r', check_sq=False)
    firstReadid = None
    for read in bamfile:

        read_id = read.query_name
        if firstReadid is None:
           firstReadid = read_id

        seq = read.query_sequence
        # qual = read.query_qualities
        qual = ''.join(map(lambda x: chr(x + 33), read.query_qualities))
        # print(seq)
        # print(qual)
        tags = read.get_tags()
        for tag in tags:
            if tag[0] == "mv":
                move = tag[1]

        movedict[read_id] = (seq,qual,move)
    #
    return movedict


import pysam
def getGenomeSeq(aligner,chrom,strand,r_st,cigar):

    #print(chrom,r_st,cigar)
    a = pysam.AlignedSegment()
    a.cigarstring = cigar

    refpos = 0
    relpos = 0
    lgenome = ""
    en = r_st
    for cigaroprator, cigarlen in a.cigar:

         if cigaroprator == 3: #N

            seq = aligner.seq(chrom, start=r_st, end=en)
            if seq is not None:
                lgenome = lgenome + seq
            r_st = en + cigarlen
            en = r_st

         elif cigaroprator == 0 or cigaroprator == 2:  #
             en = en + cigarlen

    seq = aligner.seq(chrom, start=r_st, end=en)
    if seq is not None:
        lgenome = lgenome + seq

    if strand == -1:
        lgenome = mp.revcomp(lgenome)

    return lgenome

import mappy as mp


def toBamRecord(reflist, nread):

    a = pysam.AlignedSegment()
    a.query_name = nread.read_id
    a.flag = 0
    seq = nread.sequence

    a.query_sequence = seq
    a.cigarstring = nread.cigar_str

    cigar = nread.cigar_str
    if nread.q_st > 0:
        cigar = str(nread.q_st)+"S" + cigar

    left = len(nread.sequence) - nread.q_en
    if left >  0:
        cigar = cigar + str(left)+"S"



    a.cigarstring = cigar

    cigseqlen = 0
    for cigaroprator, cigarlen in a.cigar:
        if  cigaroprator == 0 or cigaroprator == 1 or cigaroprator == 4:
            cigseqlen += cigarlen

    print(cigar,str(nread.q_en))
    print(len(nread.sequence),cigseqlen)
    print(len(nread.move))
    print(len(nread.signal))

    a.reference_id = reflist.index(nread.chrom)

    a.reference_start = nread.r_st
    a.mapping_quality = 20
    return a

import numpy as np
import numba
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize
import numba


def normDiff(a, b):
    ary, scale, add = optimize_difference(a, b)
    ary = np.clip(ary, -120, 120) + 120
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


import pysam
import alignment.RecalibAndRealign as RecalibAndRealgin
import os
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import alignment.AdjustUtils as autil
import matplotlib.pyplot as plt
import alignment.SIgnalUtils as sutils
import alignment.SigAlgin as SigAlgin
import alignment.RecalibAndRealign as RecalibAndRealign

def viterbiTest():

    pfile = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/separate_pq/BC1/IVT_seq_1_2202.pq"

    ref = "/share/reference/IVTmix.fa"
    pdfout = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/testrealgin.pdf"
    fmercurrent = "/mnt/share/project/newRNAporeTestData/9mer_levels_v1.txt"

    pdf = PdfPages(pdfout)

    fmerdict = autil.getCurrentDict(fmercurrent)
    df = pd.read_parquet(pfile)
    cnt = 0
    aligner = mp.Aligner(ref, preset="map-ont", k=14, best_n=1)

    for row in df.itertuples():
        # nread = NanoTuneRead(sortkey, rna, read.read_id, chrom,
        #                      strand, r_st, r_en, q_st, q_en, cigar_str, fastq,
        #                      trace, move, row_data, ref, UNIT=5)
        # recalbsignal, moveboundary, newalgin, theoryExtend, formatSignal, alginSignal, gapresolve, theory2, theoryRead, intervals, theory, diffsig, duration, meansignal = RecalibAndRealgin.recalibandrealign(
        #     nread, fmerdict)

        # columns = ['read_name', 'start', 'end', 'ref_name', 'cigar', 'seq', 'quality',
        #            'move', 'startIdx', 'endIndex', 'signal'

        plt.figure(figsize=(30, 4))
        readid = str(row.read_name)
        seq = row.seq
        print(readid,seq)
        signal = row.signal[::-1]
        siglen = len(signal)
        unit = 5
        moves = row.move
        sigboundary = list(map(lambda x:siglen-(x*unit),moves))
        sigboundary.reverse()

        movelen = (siglen // unit)
        print("movelen",movelen)
        moveoundary = list(map(lambda x: movelen - x, moves))
        print(moveoundary)
        moveoundary.reverse()

        cigar = row.cigar
        q_st,q_ed = autil.getQstQen(cigar)
        q_en = len(seq) - q_ed

        plt.plot(signal)
        for move in sigboundary:
            plt.axvline(move, color='gray', linestyle='--',linewidth=0.1,alpha=0.5)

        pdf.savefig()
        plt.close()

        idxstart = sigboundary[q_st]
        idxsend = sigboundary[q_en]
        print(idxstart,idxsend)
        trimsignal = signal[idxstart:idxsend]
        plt.figure(figsize=(30, 4))
        plt.plot(trimsignal)
        pdf.savefig()
        plt.close()

        moveoundary = moveoundary[q_st:q_en]
        moveoundary = moveoundary - moveoundary[0]
        print("mv", len(moveoundary) ,moveoundary)

        genome = aligner.seq(row.ref_name, row.start, row.end)
        # fmerDict,genome,factor,pre,toPlusVal=True
        pre = 4
        granular_level = 3
        theory1 = sutils.theoryMeanRNA(fmerdict,genome, granular_level,pre)
        plt.figure(figsize=(30, 4))
        plt.plot(theory1)
        pdf.savefig()
        plt.close()

        print(q_st,q_en,len(seq))
        seq = seq[q_st:q_en]
        fromSeq = sutils.theoryMeanRNA(fmerdict, seq, granular_level, pre)
        plt.figure(figsize=(30, 4))
        plt.plot(fromSeq)
        pdf.savefig()
        plt.close()




        #normalization
        trimsignal = trimsignal.astype(np.float64)
        binAveSignal =  autil.getBinAveSignal(moveoundary, trimsignal)
        factor=1
        theory = sutils.theoryMeanRNA(fmerdict, genome, factor, pre)
        alginSignal, a, b, null_index = autil.alignForRecalib(ref,  binAveSignal, theory, cigar)
        recalbsignal = (trimsignal * a) + b
        recalbsignal = recalbsignal.astype(np.float32)

        newcigar, tracebackPath = SigAlgin.viterbi(fromSeq, theory1, cigar, granular_level)
        print(len(seq), len(tracebackPath), tracebackPath)
        intervals, nullindex = autil.toInterval(genome, tracebackPath, moveoundary, granular_level)
        print('intervals', len(intervals), len(seq), intervals)
        newalgin = autil.getFormatSignal(intervals,recalbsignal, toBytes=False)

        plt.figure(figsize=(30, 4))
        plt.plot(newalgin)
        pdf.savefig()
        plt.close()
        null_index_interval = RecalibAndRealign.entendInterval(genome, nullindex, intervals)

        gapresolve0, intervals, takesignal = RecalibAndRealign.mindGap(genome, newalgin, theory, null_index_interval,
                                                     recalbsignal, intervals)

        gapresolve = autil.getFormatSignal(intervals, recalbsignal, toBytes=False)
        duration = list(map(lambda x: 0 if x is None else x[1] - x[0], intervals))


        plt.figure(figsize=(30, 4))
        plt.plot(gapresolve)
        pdf.savefig()
        plt.close()
        cnt+=1
        if cnt == 10:
            break


    pdf.close()



viterbiTest()