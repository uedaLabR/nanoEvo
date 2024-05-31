import adjust.AdjustUtils as au

ascii_order = '!\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
ascii_score_dict = {ascii_order[k]: k for k in range(len(ascii_order))}


class NanoEvoRead():

    def getBinkey(self):
        return self.binkey

    def getChromStrand(self):

        return self.chrom + "_" + str(self.strand)

    # read.read_id, chrom, strand, r_st, r_en, q_st, q_en, cigar_str, fastq, trace, move,row_data
    # sortkey, rna, read.read_id, chrom_g,
    # strand_g, r_st_g, r_en_g, q_st, q_en, cigar_str_g, fastq,
    # trace, move, row_data, rseq
    def __init__(self, sortkey,rna, read_id, chrom, strand, r_st, r_en, q_st, q_en, cigar_str, fastq, trace, move,
                 signal,lgenome,UNIT=10):

        self.sortkey = sortkey
        self.read_id = read_id
        self.chrom = chrom
        self.strand = strand
        self.r_st = r_st
        self.r_en = r_en
        self.q_st = q_st
        self.q_en = q_en
        self.cigar_str = cigar_str
        self.cigar_org = cigar_str
        self.refgenome = lgenome
        self.trace = None
        self.idprobbean = None

        tracelen = len(move)

        self.fastq = fastq
        fastq_list = self.fastq.split('\n')

        if rna:
            if trace is not None:
                self.trace = trace[::-1].astype(np.int16)
            move = np.array(move)
            self.move = move[::-1].astype(np.int16)
            #self.move = move.astype(np.int16)
            if signal is not None:

                adoptorlen = (len(signal) - (UNIT * tracelen))
                self.signal = signal[adoptorlen:].astype(np.float64)
                self.signal = signal[::-1]


                # print("signal", len(signal),adoptorlen,UNIT * tracelen)


            self.sequence = fastq_list[1].replace('U', 'T')
        else:
            if trace is not None:
                self.trace = trace.astype(np.int16)
            self.move = move.astype(np.int16)
            self.sequence = fastq_list[1]
            if signal is not None:
                adoptorlen = (len(signal) - (5 * tracelen))
                self.signal = signal[adoptorlen:].astype(np.float64)

        self.qscore = np.array([ascii_score_dict[symbol] for symbol in fastq_list[3]], dtype=np.int16)
        self.mean_qscore = sum(self.qscore) / len(self.qscore)
        self.readstart = 0
        self.readseq = ""
        self.traceboundary = None
        self.encodedTrace = None
        self.fromTs = False

        self.signalboundary = None
        self.cigar_org = cigar_str
        self.tracebackPath = None
        self.leftfirst = 0
        self.leftlast = 0
        self.settraceboundary(self.toIdx(self.move))
        self.cigar_N = None
        self.readseq = None
        self.traceseq = None
        self.encodedTrace = None
        self.aitvl = None

        self.formatSifnal = None
        self.diffsignal = None
        self.duration = None

    def toIdx(self,move):
        cnt = 0
        l = []
        for i in move:

            if i == 1:
                l.append(cnt)
            cnt += 1
        return l

    def setTs(self,tsName,r_st,r_en):

        self.tsName = tsName
        self.r_st_ts = r_st
        self.r_en_ts = r_en
        self.fromTs = True

    def setViterbiRange(self,leftfirst, leftlast):

        self.leftfirst = leftfirst
        self.leftlast = leftlast

    def setreadseq(self,seq):
        self.readseq = seq

    def settraceboundary(self,traceboundary):

        self.traceboundary = traceboundary
        self.traceintervals = boundaryToIntervals(traceboundary)
        self.aitvl = au.intervalToAbsolute(self.traceintervals)

    def setTraceseq(self,traceseq):

        self.traceseq = traceseq

    def setsignalboundary(self,signalboundary):

        self.signalboundary = boundaryToIntervals(signalboundary)


    def toTuple(read):

        #
        offset = 0
        if read.traceboundary is not None:
            offset = read.traceboundary[0]
            if offset < 0:
                offset = 0


        strand = read.strand
        if strand == -1:
            strand = 2

        return read.sortkey,read.read_id,read.chrom, strand, read.readstart,read.readseq,\
               read.r_st, read.r_en,read.q_st, read.q_en,read.cigar_str,\
               read.fastq, offset,read.traceintervals,\
               read.leftfirst,read.leftlast,  \
               read.traceseq, \
               read.encodedTrace,read.signal,read.signalboundary


    def toTupleSimple(read):

        #
        offset = 0
        if read.traceboundary is not None:
            offset = read.traceboundary[0]
            if offset < 0:
                offset = 0

        strand = read.strand
        if strand == -1:
            strand = 2

        formatsignal = None
        if read.traceintervals is not None:

            formatsignal = au.getFormatSignal(read.aitvl, read.normSignal) # make it 64 data point bin
            # print(formatsignal)

        return read.sortkey, read.read_id, read.chrom, strand, read.readstart, read.readseq, \
               read.r_st, read.r_en, read.q_st, read.q_en, read.cigar_str, \
               read.fastq, offset, read.traceintervals, \
               read.leftfirst, read.leftlast, \
               read.traceseq, \
               read.encodedTrace, formatsignal


    def toTupleForTrain(read):

        return read.sortkey, read.read_id, read.chrom, read.strand, read.r_st, read.r_en, \
            read.formatSignal , read.diffsignal, read.duration

def getMetaForTrain():

    meta = {'sortkey':'u8', 'read_id':'object','chrom':'object','strand':'u1',
            'r_st':'u4','r_en':'u4','formatSignal':'object','diffsignal':'object','duration':'object'
            }

    return meta


def getMeta():

    meta = {'sortkey':'u8', 'read_id':'object','chrom':'object','strand':'u1','readstart':'u4','readseq':'object',
            'r_st':'u4','r_en':'u4','q_st':'u4','q_en':'u4','cigar':'object',
            'fastq':'object','offset':'u1','traceintervals':'object',
            'leftfirst':'u4','leftlast':'u4','traceseq':'object',
            'trace':'object','signal':'object'
            }

    return meta

import numpy as np
def boundaryToIntervals(traceboundary):

    arr = np.array(traceboundary)
    diff = np.diff(arr)
    diff = np.clip(diff,0,65535)
    return diff


#batch holder for read
class ReadHolder():

    def __init__(self):
        self.holder = []

    def add(self,ntr:NanoTuneRead):
        self.holder.append(ntr)

    def getList(self):
        return self.holder

    def size(self):
        return len(self.holder)