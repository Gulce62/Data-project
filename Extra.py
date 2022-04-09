# b = [1, 5, 6, 2, 6, 4]
# e = [3, 7, 0, 2, 5, 6]
# a = np.array([])
# for i in b:
#     a = np.append(a, i)
#     print(a)
# c = a.argsort()[:3]
# print(c)
# d = np.array([])
# for i in c:
#     d = np.append(d, e[i])
# print(Counter(d).most_common(1)[0][0])
# print(mode(d))


# def printFeatures(dispf):
#     featuresinChromagram = dispf.loc[:,:11]
#     minOfChroma = featuresinChromagram.min().min()
#     maxOfChroma = featuresinChromagram.max().max()
#     meanOfChroma = featuresinChromagram.stack().mean()
#     chromaStdev = featuresinChromagram.stack().std()
#     print(f'Features of 12 Chromagrams: \
#     Min = {minOfChroma:.3f}, \
#     Max = {maxOfChroma:.3f}, \
#     Mean = {meanOfChroma:.3f}, \
#     Standard Deviation = {chromaStdev:.3f}')
#
#     featuresinMelspectrogram = dispf.loc[:,12:139]
#     minOfMel = featuresinMelspectrogram.min().min()
#     maxOfMel = featuresinMelspectrogram.max().max()
#     meanOfMel = featuresinMelspectrogram.stack().mean()
#     stdvOfMel = featuresinMelspectrogram.stack().std()
#     print(f'\nFeatures of 128 Melspectrograms: \
#     min = {minOfMel:.3f}, \
#     max = {maxOfMel:.3f}, \
#     mean = {meanOfMel:.3f}, \
#     Standard deviation = {stdvOfMel:.3f}')
#
#     featuresinMFCC = dispf.loc[:,140:179]
#     minOfMFCC = featuresinMFCC.min().min()
#     maxOfMFCC = featuresinMFCC.max().max()
#     meanOfMFCC = featuresinMFCC.stack().mean()
#     stdvOfMFCC = featuresinMFCC.stack().std()
#     print(f'\n40 MFCC features: \
#     min = {minOfMFCC:.2f},\
#     max = {maxOfMFCC:.2f},\
#     mean = {meanOfMFCC:.2f},\
#     deviation = {stdvOfMFCC:.3f}')

# classify_knn = KNN(k=3, distance_type=2)
# classify_knn.fit(X_train, y_train)
# y_prediction = classify_knn.all_prediction(X_train)
# print("Accuracy Percentage of the KNN classification with", distance_type[2],
#           "is", calculate_accuracy(y_train, y_prediction))


# def confusionMatrix(Matr1, Matr2):
#     emotionL = ["happy","sad","angry","neutral","calm","disgust","surprised","fearful"]
#     confusionMatr = np.zeros((len(emotionL),len(emotionL)))
#     Matr1 = np.array(Matr1)
#     Matr2 = np.array(Matr2)
#     for j in range(len(Matr1)):
#         for i in range(len(emotionL)):
#             if Matr1[j] == emotionL[i]:
#                 for k in range(len(emotionL)):
#                     if Matr2[j] == emotionL[k]:
#                         confusionMatr[i,k] += 1
#     return confusionMatr