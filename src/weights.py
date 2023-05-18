def sign(value):
    if value >= 0:
        return 1
    else:
        return -1

def calc_weights(svmPredictions, resultsSize, majority=False):

    nrOfSVMs = len(svmPredictions)
    finalPredictions = [0] * resultsSize

    # calculating the weights
    for i in range(resultsSize):

        if majority:
            
            totalZero = 0
            totalOne  = 0

            for j in range(nrOfSVMs):
                goal = sign(svmPredictions[j][i])
                if goal==1:
                    totalOne+=1
                else:
                    totalZero+=1
            
            if totalOne >= totalZero:
                finalPredictions[i] = 1
            else:
                finalPredictions[i] = 0

        else :

            absValPred = [0] * nrOfSVMs             # distance to the hyperplane
            rank = [0] * nrOfSVMs                   # ranking of the most certain svm's
            pred = [0] * nrOfSVMs                   # prediction from each SVM
            sumTotal = 0                            # factor to calculate the weights

            for j in range(nrOfSVMs):
                absValPred[j] = abs(svmPredictions[j][i])
                pred[j] = sign(svmPredictions[j][i])
                sumTotal += (j+1)

            for j in range(nrOfSVMs):
                rank[j] = absValPred.index(max(absValPred))               # ranks from best to worst
                absValPred[rank[j]] = 0

            for j in range(nrOfSVMs):
                weigth = (nrOfSVMs - j) / sumTotal
                finalPredictions[i] += (weigth * pred[rank[j]])

            if finalPredictions[i] >= 0:
                finalPredictions[i] = 1
            else:
                finalPredictions[i] = 0

    return finalPredictions