from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm

class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print("Avaluant ", algorithm.GetName(), "...")
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Mostrar resultats per pantalla
        print("\n")
        
        if (doTopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                        name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                                      metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))
                
        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Valors petits signifiquen més precisió.")
        print("MAE:       Mean Absolute Error. Valors petits signifiquen més precisió.")
        if (doTopN):
            print("HR:        Hit Rate; amb quina freqüència som capaços de recomanar una puntuació 'deixada fora'. Més alt és millor.")
            print("cHR:       Cumulative Hit Rate; percentatge d'èxits, limitat a puntuacions per sobre d'un determinat llindar. Més alt és millor.")
            print("ARHR:      Average Reciprocal Hit Rank -Percentatge d'èxits que té en compte la classificació. Més alt és millor." )
            print("Cobertura:  Proporció d'usuaris per als quals hi ha recomanacions per sobre d'un determinat llindar. Més alt és millor.")
            print(""""Diversitat: 1-S, on S és la puntuació mitjana de similitud entre cada parell de recomanacions possible 
                  per a un usuari determinat. Més alt significa més divers""")

            print("Popularitat: Classificació mitjana de popularitat dels articles recomanats. Més alt vol dir més popular.")
        
    def SampleTopNRecs(self, ml, testSubject=295, k=10):
        
        for algo in self.algorithms:
            print("\nUsant recomanador ", algo.GetName())
            
            print("\nConstruint model de recomanació...")
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print("Computant recomanacions..")
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithm().test(testSet)
            
            recommendations = []
            
            print ("\nRecomanem:")
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
                

            
            
    
    