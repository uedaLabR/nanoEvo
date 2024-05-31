import numpy as np
from numba.typed import List
import pysam
import alignment.RealignUtils as util
import numba

@numba.jit(nopython=True)
def getMRange(n, m_range,bin_interval,intervallen):


    if n > len(m_range) - 1:
        m_in_alinment = m_range[len(m_range) - 1]
    else:
        m_in_alinment = m_range[n]
    start = m_in_alinment - bin_interval
    end = m_in_alinment + bin_interval
    if start < 0:
        start = 0
    if end > intervallen:
        end = intervallen
    return start,end


EXACT_MATCH = 100
UNMATCH_Penalty = 20
@numba.jit(nopython=True)
def getMatchScore(t, r):

    if t == r:
       return EXACT_MATCH

    diffPenalty = abs(t-r)
    return EXACT_MATCH -UNMATCH_Penalty- diffPenalty

# newcigar,intervals,nullindex = SigAlgin.viterbi(theory2,theoryRead,readseq,nread.cigar_str,moveboundary)
def viterbi(theory, theoryRead, cigar_str,granular_level):

    genomelen = len(theory)
    a = pysam.AlignedSegment()
    a.cigarstring = cigar_str
    m_range = List()
    for n in range(genomelen//granular_level):
        l = util.convertRefToReadPos(n,a.cigar) * granular_level
        for n in range(granular_level):
            m_range.append(l)


    return viterbiEach(theory, theoryRead, m_range)


STOP = 0
DIOGONAL_MOVE = 1
HORIZONTAL_MOVE = 2
VERTICAL_MOVE = 3
SKIP_MOVE_BASE = 10
@numba.jit(nopython=True)
def viterbiEach(theory, theoryRead,m_range):

    intervallen = len(theoryRead)
    genomelen = len(theory)

    scorematrix = np.zeros((genomelen, intervallen), dtype='float32')
    tracing_matrix = np.ones((genomelen, intervallen), dtype='uint8') # defult DIOGONAL MOVE
    # score matrix

    bin_interval = 20
    ScoreGAP = - 2
    max_score = -1
    max_index = (-1, -1)
    #
    for n in range(genomelen):

        start,end = getMRange(n, m_range,bin_interval,intervallen)


        for m in range(start,end):

            matchscore = getMatchScore(theory[n], theoryRead[m])
            if m == 0 or n == 0:

                scorematrix[n][m] = matchscore

            else:

                diagonal_score = scorematrix[n - 1, m - 1] + matchscore
                vertical_score = scorematrix[n - 1, m] + ScoreGAP
                horizontal_score = scorematrix[n, m - 1] + ScoreGAP
                # Taking the highest score
                scorematrix[n, m] = max(0, diagonal_score, vertical_score, horizontal_score)
                # Tracking where the cell's value is coming from
                if scorematrix[n, m] == 0:
                    tracing_matrix[n, m] = STOP

                elif scorematrix[n, m] == horizontal_score:
                    tracing_matrix[n, m] = HORIZONTAL_MOVE

                elif scorematrix[n, m] == vertical_score:
                    tracing_matrix[n, m] = VERTICAL_MOVE

                elif scorematrix[n, m] == diagonal_score:
                    tracing_matrix[n, m] = DIOGONAL_MOVE

                # Tracking the cell with the maximum score
                if scorematrix[n, m] >= max_score:
                    max_index = (n, m)
                    max_score = scorematrix[n, m]



    n,m = max_index
    tracebackPath = []
    tracebackPath.append((n, m))
    tracebackPathCig = []


    while n >= 0:

        if tracing_matrix[n][m] == DIOGONAL_MOVE:
            n = n - 1
            m = m - 1
            if n >= 0 and m >= 0:
                tracebackPath.append((n, m))
                tracebackPathCig.append("M")

        elif tracing_matrix[n][m] == HORIZONTAL_MOVE:

            m = m - 1
            if n >= 0 and m >= 0:
                tracebackPath.append((n, m))
                tracebackPathCig.append("I")

        elif tracing_matrix[n][m] == VERTICAL_MOVE:

            n = n - 1
            if n >= 0 and m >= 0:
                tracebackPath.append((n, m))
                tracebackPathCig.append("D")


        if m <= 0:
            break

    tracebackPath.reverse()
    tracebackPathCig.reverse()

    # for n,m in tracebackPath:
    #

    newcigar = compress_string(tracebackPathCig)
    # print(newcigar)
    return newcigar,tracebackPath

@numba.jit(nopython=True)
def compress_string(s):
    if not s:
        return ""

    result = []
    current_char = s[0]
    count = 1

    for char in s[1:]:
        if char == current_char:
            count += 1
        else:
            cg = str(count) + current_char
            if cg == "1D":
                cg = "1H"
            result.append(cg)
            current_char = char
            count = 1

    cg = str(count) + current_char
    if cg == "1D":
        cg = "1H"
    result.append(cg)

    return ''.join(result)