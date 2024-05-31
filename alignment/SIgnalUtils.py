import numpy as np

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