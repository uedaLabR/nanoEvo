import pod5 as p5
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# venv ptthon 3.9

def adjustIdx(index,signal):

    nmax = 0
    maxdiff = 0
    for n in range(0,800,5):

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

def pickupSignal(pod5dir,infile,outfile,outdata):

    pdf = PdfPages(outfile)

    read_dict = {}
    print("start reading")
    with open(infile, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            print(row[0],len(row[0]))
            read_dict[str(row[0])] = row

    print("end reading")

    files = glob.glob(pod5dir+"/*.pod5")
    unit = 5
    datadict = {}
    flen = len(files)
    cnt = 0
    for file in files:

        cnt+=1
        print("doing.." + file +" "+str(cnt)+"/"+str(flen))
        with p5.Reader(file) as reader:
            for read_record in reader.reads():
                id = str(read_record.read_id)
                if id in read_dict:

                    v = read_dict[id]
                    signal = read_record.signal
                    sample_rate = read_record.run_info.sample_rate
                    time = np.arange(len(signal)) / sample_rate
                    barcode = v[1]
                    startindex = int(v[2])
                    endindex = int(v[3])


                    # print("signal len, movelen", len(signal), startindex,endindex)

                    if barcode in datadict:
                        datalist = datadict[barcode]
                    else:
                        datalist = []
                        datadict[barcode] = datalist
                    endindex = adjustIdx(endindex,signal)
                    signal = np.clip(signal, 0, 800)
                    data = (v[0],v[1],startindex,endindex,signal,time)
                    datalist.append(data)

    #
    readids = []
    bsc =[]
    bc_signals=[]

    print("end")
    for barcode in datadict:
        print(barcode)
        datalist = datadict[barcode]
        umto = min(6920,len(datalist))
        for n in range(umto):
            v = datalist[n]
            readid, bc, startindex, endindex, signal, time = v
            readids.append(readid)
            bsc.append(bc)
            bc_signal = extractSig(signal,startindex, endindex)
            bc_signals.append(bc_signal)

            if n < 10:
                plt.figure()
                plt.plot(signal)
                plt.title(bc+"_"+readid)
                plt.axvline(startindex, color='r', linestyle='--')
                plt.axvline(endindex, color='r', linestyle='--')
                pdf.savefig()
                plt.close()

    pdf.close()

    data = {
        'readid': readids,
        'bc': bsc,
        'bc_signal':bc_signals
    }
    table = pa.Table.from_pydict(data)
    pq.write_table(table, outdata)

    # df = pd.DataFrame(alldatalist,columns=['readid', 'bc', 'startindex', 'endindex', 'signal', 'time'])
    # df.to_parquet(outdata)


pod5dir ="/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/pod5_pass"
infile = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/out.csv"
outfile = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/example.pdf"
outdata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/out.pq"

pickupSignal(pod5dir,infile,outfile,outdata)