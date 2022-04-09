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