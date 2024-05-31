import pod5 as p5
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq



def convert(pod5dir1,pod5dir2,outdata):


    files = glob.glob(pod5dir1+"/*.pod5")
    files2 = glob.glob(pod5dir2 + "/*.pod5")

    files.extend(files2)

    unit = 5
    datadict = {}
    flen = len(files)
    cnt = 0


    for file in files:

        readids = []
        signals = []
        outfilename = outdata+"/"+str(cnt)+"_init.pq"
        cnt+=1
        print("doing.." + file +" "+str(cnt)+"/"+str(flen))
        with p5.Reader(file) as reader:
            for read_record in reader.reads():
                id = str(read_record.read_id)
                signal = read_record.signal
                readids.append(id)
                signals.append(signal)


        data = {
            'readid': readids,
            'signals': signals
        }
        table = pa.Table.from_pydict(data)
        pq.write_table(table, outfilename)




    # df = pd.DataFrame(alldatalist,columns=['readid', 'bc', 'startindex', 'endindex', 'signal', 'time'])
    # df.to_parquet(outdata)


pod5dir1 ="/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/pod5_pass"
pod5dir2 ="/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/pod5_fail"

outdata = "/mnt/share/ueda/RNA004/nanoEvo/ivt_traindata/20240126_1557_MN32625_FAX70081_57a8288c/init_pq"

convert(pod5dir1,pod5dir2,outdata)