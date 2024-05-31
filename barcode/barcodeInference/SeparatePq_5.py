import glob
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pysam



import pandas as pd
import numpy as np

import pandas as pd




def separate(indata,outdata,result):

    df = pd.read_csv(result)
    BC_dict = dict(zip(df['readid'], df['BC']))

    files = glob.glob(indata + "/*.pq")
    flen = len(files)
    cnt = 0

    for file in files:

        ddict = {}
        cnt += 1
        print("doing.." + file + " " + str(cnt) + "/" + str(flen))
        df = pd.read_parquet(file)
        for row in df.itertuples(index=False, name=None):

            read_name = str(row[0])
            if read_name in BC_dict:

                bc = BC_dict[read_name]
                if bc in ddict:
                    dlist = ddict[bc]
                else:
                    dlist = []
                    ddict[bc] = dlist

                dlist.append(row)

        for bckey in ddict:

            outpath = outdata+"/BC"+str(bckey+1)
            if not os.path.exists(outpath):
                os.makedirs(outpath)

            outpath = outpath +"/" +os.path.basename(file)
            dlist = ddict[bckey]
            columns=['read_name', 'start', 'end', 'ref_name', 'cigar', 'seq', 'quality',
             'move', 'startIdx', 'endIndex', 'signal', 'barcodesignal']
            df = pd.DataFrame(dlist,columns=columns)
            df.to_parquet(outpath)



def list_subdirectories(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def mergePq(outdata):

    subds = list_subdirectories(outdata)
    for sd in subds:

        if sd is None:
            continue

        files = glob.glob(outdata+"/"+sd+"/*.pq")
        ddict = {}
        if len(files) > 20:
            for file in files:

                df = pd.read_parquet(file)
                unique_values = df['ref_name'].unique()
                sub_dfs = {val: df[df['ref_name'] == val] for val in unique_values}

                for val, sub_df in sub_dfs.items():

                    sub_df = sub_df.drop('barcodesignal', axis=1)
                    if val in ddict:
                        dff = ddict[val]
                        ndf = pd.concat([dff, sub_df])
                        ddict[val] = ndf
                    else:
                        ddict[val] = sub_df
                os.remove(file)

            keys = ddict.keys()
            for key in keys:

                if key is None:
                    continue
                fn = outdata+"/"+sd+"/"+key+".pq"
                df =  ddict[key]
                df.to_parquet(fn)


indata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/addbam_pq"
outdata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/separate_pq"
result = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bc_result.csv"

separate(indata,outdata,result)

mergePq(outdata)