import gensim
model = gensim.models.KeyedVectors.load('300features_1minwords_10context.model')

# with open('my_w2v_model.txt', 'w', encoding='utf-8') as f:
#     vector = []
#     for sentence in words:
#         for word in sentence:
#             line = list(model[word])
#             lines = [str(i) for i in line]
#             linestr = ' '.join(lines)
#             L = str(word) + ' ' + str(linestr)
#             vector.append(L)
#             print(L)
#     vect = '\n'.join(vector)
#     f.write(vect)