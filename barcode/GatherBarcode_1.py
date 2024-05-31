import pysam
import mappy as mp

def find_nth_index(array, element, n):
    count = 0
    for i, val in enumerate(array):
        if val == element:
            count += 1
            if count == n:
                return i
    return -1  #


maxlen = 270
minlen = 65
import pandas as pd
import numpy as np
def gatherBarcode(ref,bam,out):


    # aligner = mp.Aligner(ref, n_threads=10, min_dp_score=15, w=4, bw=1, k=1,
    #                      best_n=1, min_cnt=1, min_chain_score=1)  # l
    aligner = mp.Aligner(ref, "map-ont")  # l
    bamfile = pysam.AlignmentFile(bam, "rb",check_sq=False)
    header = bamfile.header.copy()

    barcodemapData = []
    cnt = 0
    for read in bamfile:

        seq = read.query_sequence
        readlen = len(seq)
        if readlen > maxlen:
            continue
        if readlen < minlen:
            continue
        hits = list(aligner.map(seq))
        if len(hits) == 1:
            hit = hits[0]
            chr = hit.ctg
            header.get_tid(hit.ctg)
            start = hit.r_st
            end = hit.r_en
            maplen = (end - start)
            if maplen < 65:
                continue
            movelen = 0
            tags = read.get_tags()
            for tag in tags:
                if tag[0] == "mv":
                    move = tag[1]
                    spacing = move[0]
                    # print("move",move[0],np.array(move))
                    left = len(seq)-hit.q_en
                    print("left",left)
                    idx = find_nth_index(move,1,left)
                    startIdx = find_nth_index(move,1,5)
                    endIndex = spacing*idx
                    print(endIndex)
            if left > 0:
                barcodemapData.append((read.query_name,chr,startIdx, endIndex))
                cnt+=1
                print((cnt,read.query_name,chr,startIdx,endIndex))


    df = pd.DataFrame(barcodemapData, columns=['read_name','barcode','startIdx','endIndex'])
    barcode_counts = df['barcode'].value_counts()
    print(barcode_counts)
    df.to_csv(out, index=False)





barcoderef = "/mnt/share/ueda/RNA004/nanoEvo/ref/barcode.fa"
bam = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bam_pass/FAX70081_pass_57a8288c_9318e4bb.bam"
# pod5dir ="/mnt/share/ueda/RNA004/U87/U87_WT/20231215_1505_MN32625_FAX70236_31c89747/pod5_pass"
out = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/out.csv"

gatherBarcode(barcoderef,bam,out)