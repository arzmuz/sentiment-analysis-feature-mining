# Imports -------------------------------------------------------------------------

# General imports
import sys
import json
import pprint
import nltk
import re
import collections

# Lemmatizer
from nltk.stem import WordNetLemmatizer
from _operator import contains
from numpy.core.defchararray import lstrip
wordnet_lemmatizer = WordNetLemmatizer()

# Object relational mapping
from webapp.models import Product, Product_Review, Feature_Analysis, Sentiword

# ---------------------------------------------------------------------------------

price = 'price'
screen = 'screen'
storage = 'storage'
design = 'design'
performance = 'performance'
keyboard = 'keyboard'
trackpad = 'trackpad'
battery = 'battery'
webcam = 'webcam'


featureDict = {
                'price' : price,
                'money' : price,
                'cost' : price,
                'expense' : price,
                'dollar' : price,
                'buck' : price,
                'payment' : price,
                'pay' : price,
                'value' : price,
                'screen' : screen,
                'touchscreen' : screen,
                'display' : screen,
                'monitor' : screen,
                'resolution' : screen,
                'retina' : screen,
                'hd' : screen,
                'full hd' : screen,
                'storage' : storage,
                'hdd' : storage,
                'ssd' : storage,
                'solid state drive' : storage,
                'hard disk' : storage,
                'hard drive' : storage,
                'design' : design,
                'architecture' : design,
                'construction' : design,
                'built' : design,
                'durability' : design,
                'portability' : design,
                'weight' : design,
                'size' : design,
                'performance' : performance,
                'speed' : performance,
                'processor' : performance,
                'memory' : performance,
                'ram' : performance,
                'keyboard' : keyboard,
                'key' : keyboard,
                'button' : keyboard,
                'trackpad' : trackpad,
                'touchpad' : trackpad,
                'battery' : battery,
                'battery life' : battery,
                'webcam' : webcam,
                'camera' : webcam,
                'cam' : webcam
               }

# Spell Checking
def words(text):
    return re.findall('[a-z]+', str.lower(text))

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(open('training_set.txt', 'r').read()))
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

#
def mapPosTagToDB( nltkTag ):
    "This maps the NLTK pos tags to SentiWords pos tags"
    if nltkTag == "JJ" or nltkTag == "JJR" or nltkTag == "JJS":
        return "a";
    elif nltkTag == "RB" or nltkTag == "RBR" or nltkTag == "RBS":
        return "r";

def mapSynonymToFeature( syn ):
    "This maps synonyms to their respective features"
    return featureDict.get(syn)

lemmatizer = WordNetLemmatizer()

stop_words= ['ourselves', 'hers', 'between', 'yourself', 'again', 'there', 'about', 'once', 'during', 'out', 'having', 'with', 'they', 'own', 'an', 
             'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
              'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'me', 'were', 'her', 'than' 
              'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'when', 'at', 
              'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 
              'over', 'why', 'so', 'can', 'did', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'only', 'myself', 'which', 'those', 
              'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here']


featureArray = ['battery', 'battery life','price', 'money', 'cost', 'expense', 'dollar', 'buck', 'payment', 'pay', 'value', 'screen', 'touchscreen', 'display', 'monitor', 'resolution', 'retina', 'hd', 'full hd', 'portability', 'weight', 'size', 'design', 'architecture', 'construction', 'built', 'durability', 'performance', 'speed', 'processor', 'keyboard', 'key', 'button', 'trackpad', 'touchpad', 'webcam', 'camera', 'memory', 'RAM', 'storage', 'hdd', 'ssd', 'solid state drive', 'hard disk', 'hard drive']
negations = ["not", "n't"]


AllProductReviews = Product_Review.objects.all()
for productReview in AllProductReviews:
    #reviewContent = productReview.review_content
    reviewContent = "The performance is not that quick, so be prepared for that. However, the design is spectacular! The trackpad is very very comfortable to use, I love it. The colors also look good on the screen in my opinion."
    #reviewContent = "The battery is hello okay bye today tomorrow hello okay bye today tomorrow hello okay bye today tomorrow hello okay bye today tomorrow hello okay bye today tomorrow hello okay bye today tomorrow hello okay bye today tomorrow big."
    #reviewContent = "This laptop doesn't have the highest specs out there, but for the price it's quite capable"
    reviewContentBySentences = nltk.sent_tokenize(reviewContent)
    
    for sentence in reviewContentBySentences:
        if 'but' in sentence:
            modified_eachSentence = sentence.split('but')
             
            reviewContentBySentences.remove(sentence)
             
            for eachModSentence in modified_eachSentence:
                reviewContentBySentences.append(eachModSentence)
    
    for sentence in reviewContentBySentences:
            
        featureCountInSentence = 0
        totalDbSentiScore = 0.0
        original_sentence=[]
        lemma=[]
        final_sentence=[]
        sentenceTokenizedByWord = nltk.word_tokenize(sentence)
        
        for word in sentenceTokenizedByWord:
            sentenceTokenizedByWord[sentenceTokenizedByWord.index(word)] = correct(word)
            
        #print(sentenceTokenizedByWord)    
        
        original_sentence.append(sentenceTokenizedByWord)  
        final_token = [] 
        for t in sentenceTokenizedByWord:
            lowered=t.lower()
            lemma.append(lemmatizer.lemmatize(lowered))
            lemma_tagged = nltk.pos_tag(lemma)
           
        for word, tag in lemma_tagged:
            if word not in stop_words:
                final_token.append((word,tag))
        final_sentence.append(final_token)        
        
        print("Original sentence:")
        print(original_sentence)   
        print("Final sentence:")
        print(final_token)
        #print(final_sentence)
        print("________________________________________________________________________________")
            
        for feature in featureArray:
            for t in final_token:
                if feature in t:
                    featureCountInSentence += 1  
          
        for feature in featureArray: 
            
            canCommit = 1
            foundEmphasis = 0
            gotNegative = 0
            featureAnalysisObj = Feature_Analysis(review_id = productReview.id)
            featureAnalysisObj.sentence = sentence
            featureAnalysisObj.feature = mapSynonymToFeature(feature)
            featureAnalysisObj.product_id = productReview.product_id
            
            chunkDist=[]  
            for t in final_token:
                if feature in t:
                    
                    featureIndex = final_token.index(t)
                    print("Feature position/index: " + str(featureIndex))
                    
                    print('Feature: ' + feature.upper())
                    print('Found in sentence: ' + sentence)
        
        #             review_tagged = nltk.pos_tag(final_sentence)
                    chunkGram = r"""Chunk: {<RB|JJ*.?>*<JJ*.?>}"""
                    chunkParser = nltk.RegexpParser(chunkGram)
                    chunked = chunkParser.parse(final_token)
                    #print(chunked)
                    
                    chunks = re.findall(r'Chunk\s(.*?)/*\)', str(chunked)) 
        #             chunks = re.sub('/RB|/JJ|\.|\)', '', str(chunks))
        
                    chunks = re.sub('/JJR|/JJS|/JJ|/RBR|/RBS|/RB', '', str(chunks))
                    print('Description: ' + str(chunks))
                                        
                    chunks = str(chunks).strip('[\'\']')
                    chunks = str(chunks).strip('[\"\"]')
                    chunks = str(chunks).replace("',", ",")
                    chunks = str(chunks).replace("\",", ",")
                    chunks = str(chunks).replace(", \'", ",")
                    chunks = str(chunks).replace(", \"", ",")
                    
                    featureAnalysisObj.description = str(chunks)
                        
                    if (chunks):
                        '''
                        Code for the case when in a sentence there are two chunks=sentiment so there are probably two
                        sentiments for two different features. 
                        EXAMPLE: "battery is GOOD but screen is BAD"
                        '''
                        if "," in str(chunks):
                            splitChunks = str(chunks).split(',')
                            
                            for eachChunk in splitChunks:
                                eachChunk = str(eachChunk).lstrip()
                                #print(eachChunk.upper())
                                if " " in eachChunk:
                                    '''
                                    Case in which one (or more) of the chunk=sentiment is composed by more than one word. 
                                    EXAMPLE: "Screen is VERY GOOD but screen is BAD"
                                    '''
                                    splittedChunk=eachChunk.split(" ")                            
                                    distances=[]
                                    currentIndex = 0
                                    
                                    for i in final_token:
                                        previous=""
                                        
                                        for j in splittedChunk:
                                            if j in i:
                                                #print(final_token.index(i))
                                                if j==previous:
                                                    pass
                                                else:
                                                    if j==splittedChunk[0]:
                                                        firstChIndx=final_token.index(i,currentIndex)
                                                    featureChunkDistance = abs(featureIndex - final_token.index(i,currentIndex))
                                                    #distances.append(featureChunkDistance)
                                                    
                                                    if len(distances)!=0:
                                                        for k in chunkDist:
                                                            if (j!=k[1]) or ((j==k[1]) and (featureChunkDistance<k[0])):
                                                                distances.append((featureChunkDistance))
                                                                #distances.remove(k)
                                                    else:
                                                        distances.append((featureChunkDistance))
                                                    
                                                    previous=j
                                                    currentIndex = currentIndex + 1
                                            
                                                #splittedChunk.remove(j)
                                    print("First index: "+str(firstChIndx))
                                    print("Chunk length: " + str(len(distances)))
                                    #print(distances)
                                    avgIndexPos= float(firstChIndx)+(float(len(distances))/2- 0.5)
                                    print('distance from ' + chunks + ': ' + str(abs(float(featureIndex) - avgIndexPos)))
                                    chunkDist.append((abs(float(featureIndex) - avgIndexPos), eachChunk))
                                else:    
                                    
                                    currentIndex = 0
                                                        
                                    for i in final_token:
                                        if eachChunk in i:
                                            featureChunkDistance = abs(featureIndex - final_token.index(i,currentIndex))
                                            print('distance from ' + eachChunk + ': ' + str(featureChunkDistance))
                                            
                                            if len(chunkDist)!=0:
                                                for k in chunkDist:
                                                    if (eachChunk!=k[1]) or ((eachChunk==k[1]) and (featureChunkDistance<k[0])):
                                                        chunkDist.append((featureChunkDistance, eachChunk))
                                                        chunkDist.remove(k)
                                            else:
                                                chunkDist.append((featureChunkDistance, eachChunk))               
                                        currentIndex=currentIndex+1    
   
                            print(chunkDist)
                            minChunkDist = min(chunkDist)                
                            print(minChunkDist)
                            
                            if (minChunkDist[0] < 10):
                                featureAnalysisObj.description = minChunkDist[1]
                            else:
                                canCommit = 0
                            
                            if (featureCountInSentence == 1):           # If a sentence has 1 feature but multiple chunks
                                totalDbSentiScore = 0.0
                                #negationFound = 0
                                for each in chunkDist:
                                    for word,tag in final_token:
                                        if word == each[1]:
                                            newPosTag = tag
                                            newPosTag = mapPosTagToDB(newPosTag)
                                            
#                                         for neg in negations:
#                                             if neg in word:
#                                                 negationFound = 1
                                
                                    print(each[1])
                                    
                                    if (' ' in each[1]):
                                        splittedChunk = each[1].split(" ")
                                        
                                        negationFound = 0
                                        for each in splittedChunk:
                                            for word,tag in final_token:
                                                if word == each:
                                                    newPosTag = tag
                                                    newPosTag = mapPosTagToDB(newPosTag)
                                                    #print(j)
                                                    #print(newPosTag)
                                                    
                                                    for neg in negations:
                                                        if neg in word:
                                                            negationFound = 1
                                            
                                            if each == 'very':
                                                foundEmphasis = 1
                                            
                                            dbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=each, pos=newPosTag)
                                            print(dbSentiScore)
                                            dbSentiScore = str(dbSentiScore).strip("[]")
                                            print(dbSentiScore)
                                            
                                            if not str(dbSentiScore).strip('[]'):
                                                print('No score received from DB')                        
                                            else:
                                                dbSentiScoreFloat = float(dbSentiScore)
                                            
                                                if (foundEmphasis == 1 and dbSentiScoreFloat < 0.0):
                                                    gotNegative = 1
                                                    dbSentiScoreFloat = -dbSentiScoreFloat
                                                    
                                                totalDbSentiScore += dbSentiScoreFloat               
                                               
                                                if (foundEmphasis == 1 and gotNegative == 1):
                                                    print(totalDbSentiScore)
                                                    totalDbSentiScore = -totalDbSentiScore
                                    
                                        print(negationFound)    
                                        if negationFound == 1:
                                            totalDbSentiScore = -totalDbSentiScore
                                                        
                                    else:
                                        dbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=each[1], pos=newPosTag)
                                        print(dbSentiScore)
                                        dbSentiScore = str(dbSentiScore).strip("[]")
                                        print(dbSentiScore)
                                        
                                        if not str(dbSentiScore).strip('[]'):
                                            print('No score received from DB')                        
                                        else:
                                            totalDbSentiScore += float(dbSentiScore)
                                    
                            else:                                       # If a sentence has multiple features
                                if ' ' in minChunkDist[1]:
                                    totalDbSentiScore = 0.0
                                    #print('INSIDE IF')
                                    splittedChunk = minChunkDist[1].split(" ")
                                    for each in splittedChunk:
                                        for word,tag in final_token:
                                            if word == each:
                                                newPosTag = tag
                                                newPosTag = mapPosTagToDB(newPosTag)
                                                #print(j)
                                                #print(newPosTag)
                                        
                                        if each == 'very':
                                            foundEmphasis = 1
                                        
                                        dbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=each, pos=newPosTag)
                                        print(dbSentiScore)
                                        dbSentiScore = str(dbSentiScore).strip("[]")
                                        print(dbSentiScore)

                                        if not str(dbSentiScore).strip('[]'):
                                            print('No score received from DB')                        
                                        else:
                                            dbSentiScoreFloat = float(dbSentiScore)
                                            
                                            if (foundEmphasis == 1 and dbSentiScoreFloat < 0.0):
                                                gotNegative = 1
                                                dbSentiScoreFloat = -dbSentiScoreFloat
                                                
                                            totalDbSentiScore += dbSentiScoreFloat               
                                           
                                            if (foundEmphasis == 1 and gotNegative == 1):
                                                totalDbSentiScore = -totalDbSentiScore
                                    
                                    for neg in negations:
                                        if neg in splittedChunk:
                                            totalDbSentiScore = -totalDbSentiScore   
                            
                                else:
                                    #print('INSIDE ELSE')
                                    for word,tag in final_token:
                                            if word == minChunkDist[1]:
                                                newPosTag = tag
                                                newPosTag = mapPosTagToDB(newPosTag)
                                        
                                    totalDbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=minChunkDist[1], pos=newPosTag)
                                
                        else:
                            '''
                            Code for the case when in a sentence there is only one chunk=sentiment 
                            EXAMPLE: "The price is HIGH"
                            '''
                            #print(chunks)
                            if " " in chunks:
                                '''
                                Case in which the chunk=sentiment is composed by more than one word. 
                                    EXAMPLE: "The price is VERY HIGH"
                                '''
                                splittedChunk=chunks.split(" ") 
                                
                                distances=[]
                                for i in final_token:
                                    previous=""
                                    for j in splittedChunk:
                                        if j in i:
                                            #print(final_token.index(i))
                                            if j==previous:
                                                pass
                                            else:
                                                if j==splittedChunk[0]:
                                                    firstChIndx=final_token.index(i)
                                                featureChunkDistance = abs(featureIndex - final_token.index(i))
                                                distances.append(featureChunkDistance)
                                                previous=j
                                                
                                                for word,tag in final_token:
                                                    if word == j:
                                                        newPosTag = tag
                                                        newPosTag = mapPosTagToDB(newPosTag)
                                                        #print(j)
                                                        #print(newPosTag)
                                                
                                                if j == 'very':
                                                    foundEmphasis = 1
                                                
                                                dbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=j, pos=newPosTag)
                                                print(dbSentiScore)
                                                dbSentiScore = str(dbSentiScore).strip("[]")
                                                print(dbSentiScore)
                                                
                                                if not str(dbSentiScore).strip('[]'):
                                                    print('No score received from DB')                        
                                                else:
                                                    dbSentiScoreFloat = float(dbSentiScore)
                                                    
                                                    if (foundEmphasis == 1 and dbSentiScoreFloat < 0.0):
                                                        gotNegative = 1
                                                        dbSentiScoreFloat = -dbSentiScoreFloat
                                                        
                                                    totalDbSentiScore += dbSentiScoreFloat               
                                                   
                                                    if (foundEmphasis == 1 and gotNegative == 1):
                                                        totalDbSentiScore = -totalDbSentiScore
                                                                                                    
                                                    for neg in negations:
                                                        if neg in splittedChunk:
                                                            totalDbSentiScore = -totalDbSentiScore
                                                                    
                                            #splittedChunk.remove(j)
                                print("First index: "+str(firstChIndx))
                                print("Chunk length: " + str(len(distances)))
                                #print(distances)
                                avgIndexPos= float(firstChIndx)+(float(len(distances))/2- 0.5)
                                print('distance from ' + chunks + ': ' + str(abs(float(featureIndex) - avgIndexPos)))
                            
                                if (abs(float(featureIndex) - avgIndexPos) > 10):
                                    canCommit = 0
                            
                            else:                        
                                for i in final_token:
                                    if chunks in i:
                                        print(final_token.index(i))
                                        featureChunkDistance = abs(featureIndex - final_token.index(i))
                                        print('distance from ' + chunks + ': ' + str(featureChunkDistance))
                                
                                        if (featureChunkDistance > 10):
                                            canCommit = 0
                                
                                for word,tag in final_token:
                                    if word == chunks:
                                        newPosTag = tag
                                        newPosTag = mapPosTagToDB(newPosTag)
                                
                                totalDbSentiScore = Sentiword.objects.values_list('score', flat=True).filter(lemma=chunks, pos=newPosTag)
                    
                    print('Sentiment Score: ' + str(totalDbSentiScore))
                    print('________________________________________________________')       

                    if not str(totalDbSentiScore).strip('[]'):
                        print('No score received from DB')                        
                    else:
                        convTotalDbSentiScore = float(str(totalDbSentiScore).strip('[]'))
                        
                        featureAnalysisObj.score = convTotalDbSentiScore
                        
                        if (convTotalDbSentiScore != 0.0 and canCommit == 1):                
                            featureAnalysisObj.save()
                            print(':: Committed to DB ::')
        

