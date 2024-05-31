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
    header = bamfile.header.copy()

    for read in bamfile:

        read_name = read.query_name
        start = read.reference_start
        end = read.reference_end
        ref_name = bamfile.get_reference_name(read.reference_id)
        cigar = read.cigarstring
        seq = read.query_sequence
        quality = read.query_qualities
        readlen = len(seq)
        if readlen < 500:
            continue
            # Do not take short read since they are likely to be spike in RNA

        left = 50
        if read.cigar[-1][0] == 4:  # softclip
            endpos = read.cigar[-1][1]
            left = readlen-endpos

        tags = read.get_tags()
        for tag in tags:
            if tag[0] == "mv":
                move = tag[1]
                spacing = move[0]
                idx = find_nth_index(move, 1, left)
                startIdx = find_nth_index(move, 1, 5)
                endIndex = spacing * idx

                info = (read_name,start,end,ref_name,cigar,seq,quality,move,startIdx,endIndex)
                #
                bamdict[read_name] = info


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


import pandas as pd
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
import barcode.CNNWavenet as cnnwavenet


def formatX(X,wlen):
   return np.reshape(X, (-1, wlen, 1))

def addAnotation(wt,indata,result):

    wlen = 4096+1024
    lr = 0.0008
    num_classes = 8
    model = cnnwavenet.build_network(shape=(None, wlen, 1), num_classes=num_classes)
    optim = keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    model.load_weights(wt)

    files = glob.glob(indata + "/*.pq")
    flen = len(files)
    filecnt = 0
    dd = {}
    for file in files:

        filecnt += 1
        print("doing.." + file + " " + str(filecnt) + "/" + str(flen))
        df = pd.read_parquet(file)
        names = df['read_name']
        bc_signal = np.concatenate(df['barcodesignal'].values)
        print(bc_signal.shape)
        bc_signal = np.clip(bc_signal, 0, 800)
        bc_signal = formatX(bc_signal, wlen)

        prediction = model.predict(bc_signal, batch_size=None, verbose=0, steps=None)

        cnt = -1

        for row in prediction:

            # incriment
            cnt += 1
            rdata = np.array(row)
            maxidxs = np.where(rdata == rdata.max())
            #unique hit with more than zero Intensity
            if len(maxidxs) == 1 and rdata.max() >= 0.75:
                maxidx = int(maxidxs[0])
                readid = names[cnt]
                dd[readid] = maxidx


    with open(result, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(['readid', 'BC'])
        for key, value in dd.items():
            writer.writerow([key, value])


wt = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/weight"
indata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/addbam_pq"
result = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/bc_result.csv"

addAnotation(wt,indata,result)