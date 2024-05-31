import numpy as np
import pysam
import pandas as pd
from collections import defaultdict

def extractBCBan(bc,in_bam , bc_bam,sortandMap=True):

    read_counts = defaultdict(int)
    with pysam.AlignmentFile(in_bam, "rb") as infile:
        with pysam.AlignmentFile(bc_bam, "wb", template=infile) as outfile:

            for read in infile:


                bcinread = read.get_tag("XB")
                if bcinread in bc:
                    outfile.write(read)
                    if not read.is_unmapped:
                        # マッピングされているリファレンスの名前を取得し、カウントを増やす
                        ref_name = infile.get_reference_name(read.reference_id)
                        read_counts[ref_name] += 1

                    # else:
                    #    read_counts["unmap"] += 1

    if sortandMap:
        input_dir, input_filename = os.path.split(bc_bam)
        filename_without_extension = os.path.splitext(input_filename)[0]
        sorted_bam = os.path.join(input_dir, filename_without_extension + "_sorted.bam")
        pysam.sort("-o", sorted_bam, bc_bam)
        pysam.index(sorted_bam)
        os.remove(bc_bam)
    return read_counts

import csv
def merge_bams(input_bams, output_bam,result,stats):

    df = pd.read_csv(result)
    print(df)
    counts = df['BC'].value_counts()
    print(counts)

    BC_dict = dict(zip(df['readid'], df['BC']))
    mycounter = {}
    mycounter[-3] = 0
    mycounter[-2] = 0
    mycounter[-1] = 0
    cnt = 0
    with pysam.AlignmentFile(input_bams[0], "rb") as first_infile:
        with pysam.AlignmentFile(output_bam, "wb", template=first_infile) as outfile:

            #
            for input_bam in input_bams:
                with pysam.AlignmentFile(input_bam, "rb") as infile:

                    #
                    for read in infile:
                        #
                        if read.is_unmapped:

                            mycounter[-3] = mycounter[-3] + 1
                            read.set_tag('XB', -3)
                            outfile.write(read)
                            continue

                        cnt+=1
                        if cnt%10000==0:
                            print(cnt)
                            print(mycounter)

                        read_name = read.query_name
                        if read_name in BC_dict:
                            bc = BC_dict[read_name]
                            bc = bc+1
                            read.set_tag('XB', bc)
                            if bc in mycounter:
                                mycounter[bc] = mycounter[bc] +1
                            else:
                                mycounter[bc] = 1

                        else:

                            if len(read.query_sequence) > 500:
                                read.set_tag('XB', -1)
                                mycounter[-1] = mycounter[-1] + 1

                            else:
                                read.set_tag('XB', -2)
                                mycounter[-2] = mycounter[-2] + 1

                        #remove move since unused there after and heavy
                        read.set_tag('mv', None, value_type=None)

                        outfile.write(read)

    print(mycounter)

    with open(stats, 'w', newline='') as file:

        keys = sorted(mycounter.keys())
        for d in keys:
            file.write(str(d)+","+str(mycounter[d])+"\n")

    return mycounter.keys()

# bam1 = "/share/ueda/nanoEvo/pass_filter.bam"
# bam2 = "/share/ueda/nanoEvo/fail_filter.bam"
# result = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bc_result.csv"
# stats = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/stats.csv"
# outdir = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bams"



import os
def run(input_bams,result,stats,outdir):


    if not os.path.exists(outdir):
        os.mkdir(outdir)

    output_bam = outdir+"/merge.bam"
    print("marge bam")
    bcs = merge_bams(input_bams, output_bam,result,stats)
    countdict = {}
    for bc in bcs:

        if bc < 0:
            continue
        bc_bam = outdir + "/BC"+str(bc)+".bam"
        print("doing " + bc_bam)
        read_counts = extractBCBan([bc],output_bam , bc_bam)
        countdict["BC"+str(bc)] =  read_counts

    # bc_bam = outdir + "/unmap.bam"
    # print("doing " + bc_bam)
    # read_counts = extractBCBan([-3],output_bam , bc_bam,sortandMap=False)
    # countdict["unmap"] =  read_counts

    bc_bam = outdir + "/multiplex error.bam"
    print("doing " + bc_bam)
    read_counts = extractBCBan([-2,-1],output_bam , bc_bam)
    countdict["multiplex_error"] =  read_counts

    keys = sorted(countdict.keys())
    rowkys = set()
    for key in keys:
        read_counts = countdict[key]
        rowkys.update(read_counts.keys())

    rowkys_list = sorted(list(rowkys))
    collen = len(keys)
    rowlen = len(rowkys)
    data = np.zeros((rowlen,collen))

    colidx = 0
    for key in keys:
        read_counts = countdict[key]
        rowidx =0
        for rowkey in rowkys_list:

            if rowkey in read_counts:
                v = read_counts[rowkey]
                data[rowidx,colidx] = int(v)

            rowidx+=1

        colidx +=1

    df = pd.DataFrame(data, columns=keys, index=rowkys_list)
    df.to_csv(stats, index_label="Index")


bam1 = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtY.bam"
# bam2 = "/share/ueda/nanoEvo/fail_filter.bam"
result = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bc_result.csv"
stats = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtYbams/stats.csv"
outdir = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtYbams"
# input_bams = [bam1,bam2]
input_bams = [bam1]

run(input_bams,result,stats,outdir)


bam1 = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6A.bam"
# bam2 = "/share/ueda/nanoEvo/fail_filter.bam"
stats = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6Abam/stats.csv"
outdir = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6Abam"
# input_bams = [bam1,bam2]
input_bams = [bam1]

run(input_bams,result,stats,outdir)

bam1 = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6A_DRACH.bam"
# bam2 = "/share/ueda/nanoEvo/fail_filter.bam"
stats = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6A_DRACH/stats.csv"
outdir = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindatav0.7/ivtm6A_DRACH"
# input_bams = [bam1,bam2]
input_bams = [bam1]

