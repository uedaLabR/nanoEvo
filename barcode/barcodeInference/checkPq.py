import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam


def find_nth_index(array, element, n):
    count = 0
    for i, val in enumerate(array):
        if val == element:
            count += 1
            if count == n:
                return i
    return -1  #


def addBamInfo(bamdict,bam):

    bamfile = pysam.AlignmentFile(bam, "rb", check_sq=False)
    cnt =0
    for read in bamfile:

        if cnt%10000==0:
            print(cnt)
            # if cnt == 50000:
            #     break
        cnt+=1
        read_name = read.query_name
        start = read.reference_start
        end = read.reference_end
        ref_name = bamfile.get_reference_name(read.reference_id)
        cigar = read.cigarstring
        seq = read.query_sequence
        qs =""
        quality = read.query_qualities
        if quality is not None and len(quality)>0:
            qs = "".join([chr(q+33) for q in quality])

        readlen = len(seq)
        if readlen < 500:
            continue
            # Do not take short read since they are likely to be spike in RNA

        left = 50
        cglist = read.cigar
        if len(cglist) > 0 and read.cigar[-1][0] == 4:  # softclip
            endpos = read.cigar[-1][1]
            left = readlen-endpos

        tags = read.get_tags()
        for tag in tags:
            if tag[0] == "mv":
                move = tag[1]
                # move = np.array(move)
                spacing = move[0]
                idx = find_nth_index(move, 1, left)
                startIdx = find_nth_index(move, 1, 5)
                endIndex = spacing * idx

                compactMove = [i for i, x in enumerate(move) if x == 1]
                info = (read_name,start,end,ref_name,cigar,seq,qs,compactMove,startIdx,endIndex)
                #
                bamdict[str(read_name)] = info


    return bamdict


def adjustIdx(index,signal):

    nmax = 0
    maxdiff = 0
    for n in range(0,2500,10):

        b4ave = np.mean(signal[index+n-100:index+n])
        afterave = np.mean(signal[index + n :index + n+100])
        diff = afterave-b4ave
        if diff > maxdiff:
            maxdiff = diff
            nmax = n
    # print(nmax)
    return index + nmax

def extractSig(signal,start, end):

    unit = 4096+1024
    sub_array = signal[start:end]
    arrayB = np.zeros( unit)
    length = len(sub_array)
    if length <=  unit:
        start_idx = (unit - length) // 2
        arrayB[start_idx:start_idx + length] = sub_array
    else:
        # right algin
        arrayB = sub_array[length-unit:length]

    return arrayB

import pandas as pd
def check(indata,pdff):


    files = glob.glob(indata+"/*.pq")
    flen = len(files)
    cnt = 0

    pdf = PdfPages(pdff)
    for file in files:


        df = pd.read_parquet(file)
        for row in df.itertuples():

            readid = str(row.read_name)
            signal = row.signal
            start = row.startIdx
            end = row.endIndex

            cnt += 1
            plt.figure(figsize=(15, 4))
            plt.plot(signal)
            plt.title(readid)
            plt.axvline(start, color='r', linestyle='--')
            plt.axvline(end, color='r', linestyle='--')
            pdf.savefig()
            plt.close()
            print(cnt)
            if cnt == 30:
                break
        break
    pdf.close()



indata  = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/addbam_pq"
pdff = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/checkPdf.pdf"

check(indata,pdff)