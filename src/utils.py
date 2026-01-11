import math 

def mean(data):
    n = len(data)
    d = len(data[1])
    mean = [0]*d

    for j in range(d):
        for i in range(n):
            mean[j] +=data[i][j] 
        mean[j] /= n
    return mean,n,d

def mean_centred_data(data, mean, n, d):

    mean_centred = []

    for i in range(n):
        row = []
        for j in range(d):
            row.append(data[i][j] - mean[j])
        mean_centred.append(row)
    return mean_centred

def calculate_cov_matrix(mean_centred, n, d):
    cov_matrix = [[0 for _ in range(d)] for _ in range(d)]

    for i in range(d):
        for j in range(d):
            for k in range(n):
                cov_matrix[i][j] += mean_centred[k][i] * mean_centred[k][j] 
            cov_matrix[i][j] /= (n-1)
    return cov_matrix

def mat_vec_mult(cov_matrix,v):
    resultant_matrix = [0]* len(v)
    for i in range(len(cov_matrix)):
        for j in range(len(cov_matrix)):
            resultant_matrix[i] += cov_matrix[i][j] * v[j]
    return resultant_matrix


def magnitude(resultant_matrix):
    mag = 0
    for i in range(len(resultant_matrix)):
        mag += resultant_matrix[i]* resultant_matrix[i]
    return math.sqrt(mag)