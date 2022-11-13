
from MovieLens import MovieLens
from ContentKNNAlgorithm import ContentKNNAlgorithm
from Evaluator import Evaluator
from surprise import NormalPredictor

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Carregant valoració de pel·lícules...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputant el rànking de popularitat de les pel·lícules epr mesurar la popularitat del resultat...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

# Carregar sets de dades pels algoritmes de recomanació
(ml, evaluationData, rankings) = LoadMovieLensData()

# Construir un evaluador per evaluar aquests algoritmes.
evaluator = Evaluator(evaluationData, rankings)

contentKNN = ContentKNNAlgorithm()
evaluator.AddAlgorithm(contentKNN, "ContentKNN")

# Recomanador Random
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)


