import re
import numpy as np
# import scipy
from scipy.spatial import distance

with open('cats.txt') as f:
    sentences = f.readlines()

sentences = [sentence.strip().lower() for sentence in sentences]
#filter(None, sentences)
text = ' '.join(sentences)
words = {}

#filter(None, sentence)

k = 0
for word in [word for word in re.split('[^a-z]', text) if word]:
    if not word in words.values():
        words[k] = word
        k += 1


matrix = np.array([])

for sentence in sentences:
    line_matrix = []
    words_in_sentence = re.split('[^a-z]', sentence)

    for index in words:
        count = len([w for w in words_in_sentence if w.strip() == words[index]])
        line_matrix.append(count)

    if len(matrix):
        matrix = np.vstack((matrix, np.array(line_matrix)))
    else:
        matrix = np.array(line_matrix)

cosine_values = [distance.cosine(matrix[0], row) for row in matrix]
# print cosine_values


distances = list()
for i in range(len(sentences)):
    dd = distance.cosine(matrix[0], matrix[i])
    distances.append((i, dd))

sorted_list = sorted(distances, key=lambda tup: tup[1])

print sorted_list[1][0], sorted_list[2][0]

