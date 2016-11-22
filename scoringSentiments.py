from webapp.models import Feature_Analysis
from django.db.models import Avg, Max, Min, Count
import django
import operator
django.setup()

def computeStdDev(resultSet, mean, count):
    
    sum = 0
    
    for X in resultSet:
        split = str(X).split(": ")
        X = float(split[1])
        
        sum += pow(X - mean,2)
        
    std = pow(sum/count,0.5)
    
    return std

def normalizeData(valToNorm, mean, std):
    "This is to normalize data"
    
    norm = (valToNorm - mean)/std
    
    return norm

def getScoreClass(score):
    
    if (score <= -0.9):
        score = 0
    if (score > -0.9 and score <= -0.7):
        score = 1
    elif (score > -0.7 and score <= -0.5):
        score = 2
    elif (score > -0.5 and score <= -0.3):
        score = 3
    elif (score > -0.3 and score <= -0.1):
        score = 4           
    elif (score > -0.1 and score <= 0.1):
        score = 5
    elif (score >= 0.1 and score < 0.3):
        score = 6
    elif (score >= 0.3 and score < 0.5):
        score = 7
    elif (score >= 0.5 and score < 0.7):
        score = 8           
    elif (score >= 0.7 and score < 0.9):
        score = 9     
    elif (score >= 0.9):
        score = 10     
    else: 
        score = 0
        
    score += 2
    if (score > 10):
        score = 10      
        
    return score      

def getScoresForFeatures(productId):
    "This is to compute the relative scores for the features of the given product"
    
    features = ['price', 'screen', 'storage', 'design', 'performance', 'keyboard', 'trackpad', 'battery', 'webcam']
    
    for feature in features:
    
        featureDict = {
            'price' : 0,
            'screen' : 0,
            'storage' : 0,
            'design' : 0,
            'performance' : 0,
            'keyboard' : 0,
            'trackpad' : 0,
            'battery' : 0,
            'webcam' : 0
            }
        
        gotAvgDb = 0
        gotAvgProduct = 0
        gotCount = 0
        score = 0
            
        countDB_feature = Feature_Analysis.objects.filter(feature = feature).aggregate(Count('score'))
        avgDB_feature = Feature_Analysis.objects.filter(feature = feature).aggregate(Avg('score'))
        avgProduct_feature = Feature_Analysis.objects.filter(feature = feature, product_id = productId).aggregate(Avg('score'))

        if countDB_feature['score__count'] is not None:
            countDB_feature = float(countDB_feature['score__count'])
            gotCount = 1
        
        if avgDB_feature['score__avg'] is not None:
            avgDB_feature = float(avgDB_feature['score__avg'])
            gotAvgDb = 1
            
        if avgProduct_feature['score__avg'] is not None:   
            avgProduct_feature = float(avgProduct_feature['score__avg'])
            gotAvgProduct = 1
     
        if (gotAvgDb == 1 and gotAvgProduct == 1 and gotCount == 1):
            resultSet = Feature_Analysis.objects.filter(feature = feature)
            std = computeStdDev(resultSet,avgDB_feature,countDB_feature)
            
            score = normalizeData(avgProduct_feature,avgDB_feature,std)
            score = getScoreClass(score)
            print(str(feature) + ": " + str(score))
            print("--------")
            featureDict[feature] = score
        else: 
            print(str(feature) + ": " + str(score)) 
            print("--------")    

    print(featureDict)
    print(sorted(featureDict.items(), key=operator.itemgetter(1)))
    print(list(featureDict.items()).sort(key=lambda t:t[1], reverse=True))

# Test            

for p in [23,22,21,20,19,18,16,15,14,13,10,7]:    
    getScoresForFeatures(p)    
