from sense2vec import Sense2Vec
from scipy import spatial
import numpy as np
s2v = Sense2Vec().from_disk('s2v_old')


king = s2v[s2v.get_best_sense("Marie Curie")]
woman = s2v[s2v.get_best_sense("woman")]
man = s2v[s2v.get_best_sense("man")]
queen = s2v[s2v.get_best_sense("Albert Einstein")]

print(1-spatial.distance.cosine(king, queen))
print(1-spatial.distance.cosine(woman, queen))
print(1-spatial.distance.cosine(man, queen))

npking = np.array(king)
npwoman = np.array(woman)
npman = np.array(man)
results = npking + npwoman + npman
print(1-spatial.distance.cosine((results/3), queen))

