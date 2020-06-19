#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pretty_midi
import argparse
import librosa
import os
import sys
import math
import random
import numpy as np


# In[2]:


workdir = "C:\\Users\\toend\\Documents\\ITU\\Thesis"


# In[3]:


def reshape(piano_roll):
    h, w = piano_roll.shape
    slices = []
    for i in range(w):
        columnSlice = piano_roll[:,i]
        columnSlice = np.asarray(columnSlice)
        slices.append(columnSlice)
    return np.asarray(slices)


# In[4]:


def make0or1(piano_roll):
    #Need to convert all "volume" values (f.ex. 47) to just values of 1 to not confuse the network
    simplified_piano_roll = np.where(piano_roll==0, piano_roll, 1)
    return simplified_piano_roll


# In[5]:


def createPianoRoll(piano, frequency):
    endTime = math.ceil(piano.get_end_time())
    #print(endTime)
    notes = piano.notes
    #adding a pad to avoid index out of bounds
    width = (endTime*frequency)
    piano_roll = np.zeros((128, width))
    for note in notes:
        pitch = note.pitch
        #NORMALIZING
        velocity = note.velocity/100
        start = int(round(note.start * frequency))
        end = int(round(note.end * frequency))
        for i in range(start-1, end):
            piano_roll[pitch][i] = velocity
    return piano_roll


# In[6]:


def loadData(path, frequency):
    song = []
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file() and not ".ini" in entry.name:
                #print(entry.name)
                filename = entry.name
                midi_file = entry.path
                pm = pretty_midi.PrettyMIDI(midi_file)
                if(len(pm.instruments)!=0):
                    piano = pm.instruments[0]
                    piano_roll = createPianoRoll(piano, frequency)
                    piano_roll = reshape(piano_roll)
                    song.append(piano_roll)
    return np.array(song)


# In[7]:


def padPianoRollToCorrectShape(pianoRoll, frequency):
    result = np.zeros((30*frequency, 128))
    result[:pianoRoll.shape[0],:pianoRoll.shape[1]] = pianoRoll
    return result


# In[8]:


def getAllData(frequency):
    subFolders = ["MIDIdata\\cuts\\02cuts",
                  "MIDIdata\\cuts\\04cuts",
                  "MIDIdata\\cuts\\06cuts",
                  "MIDIdata\\cuts\\08cuts",
                  "MIDIdata\\cuts\\09cuts",
                  "MIDIdata\\cuts\\11cuts",
                  "MIDIdata\\cuts\\13cuts",
                  "MIDIdata\\cuts\\14cuts",
                  "MIDIdata\\cuts\\15cuts",
                  "MIDIdata\\cuts\\17cuts",
                  "MIDIdata\\cuts\\18cuts"]
    
    for subFolder in subFolders:
        path = os.path.join(workdir, subFolder)
        X = loadData(path, frequency)
        Y = []
        for pianoRoll in X:
            pianoRoll = padPianoRollToCorrectShape(pianoRoll, frequency)
            pianoRoll = make0or1(pianoRoll) #SIMPLIFYING THE DATA
            Y.append(pianoRoll)
        yield np.array(Y)


# In[9]:


def getDataSets(frequency):
    subFolders = ["MIDIdata\\TRAINING\\ArousedNegative", "MIDIdata\\TRAINING\\ArousedPositive", 
                  "MIDIdata\\TRAINING\\CalmNegative", "MIDIdata\\TRAINING\\CalmPositive"]
    
    for subFolder in subFolders:
        path = os.path.join(workdir, subFolder)
        X = loadData(path, frequency)
        Y = []
        for pianoRoll in X:
            pianoRoll = padPianoRollToCorrectShape(pianoRoll, frequency)
            pianoRoll = make0or1(pianoRoll) #SIMPLIFYING THE DATA
            Y.append(pianoRoll)
        yield np.array(Y)


# In[10]:


def compressPianoRoll(pianoRoll):
    compressedRoll = np.empty((1, pianoRoll.shape[1]))
    for i in range(0, len(pianoRoll), 2):
        compressedRoll = np.vstack([compressedRoll, pianoRoll[i]])
    return np.delete(compressedRoll, 0, axis=0)


# In[11]:


def compressDataSet(X):
    compressedSet = np.empty((1, 1500, X.shape[2]))
    for i in range(len(X)):
        piece = compressPianoRoll(X[i])
        piece = np.reshape(piece, (1, 1500, 128))
        compressedSet = np.append(compressedSet, piece, axis=0)
    return np.delete(compressedSet, 0, axis=0)


# In[12]:


def matchOneToMany(X, Y):
    matchedX = np.empty((1, X.shape[1], X.shape[2]))
    matchedY = np.empty((1, Y.shape[1], Y.shape[2]))
    for i in range(len(X)):
        Xroll = X[i]
        Xroll = np.reshape(Xroll, (1, X.shape[1], 128))
        for k in range(len(Y)):
            Yroll = Y[k]
            Yroll = np.reshape(Yroll, (1, Y.shape[1], 128))
            matchedX = np.append(matchedX, Xroll, axis=0)
            matchedY = np.append(matchedY, Yroll, axis=0)
    matchedX = np.delete(matchedX, 0, axis=0)
    matchedY = np.delete(matchedY, 0, axis=0)
    Z = list(zip(matchedX, matchedY))
    return Z


# In[13]:


#USED TO CHECK AMOUNT OF NODES PLAYED
def sumCheck(A, B):
    sumOfA = A.sum()
    sumOfB = B.sum()
    difference = sumOfA - sumOfB
    return abs(difference)


# In[14]:


#USED TO CHECK POSITION-WISE EQUALITY 
#(what nodes are played when, ie. overall structure)
def equalityCheck(A, B):
    number_of_equal_elements = np.sum(A==B)
    total_elements = np.multiply(*A.shape)
    percentage = number_of_equal_elements/total_elements
    return percentage


# In[15]:


#USED TO CHECK THE DIFFERENCE IN THE NODES BEING PLAYED 
#(Completely disregards musical theory, which would be very useful to add)
def euclideanDistance(A, B):
    dist = np.linalg.norm(A-B)
    return dist


# In[16]:


def matchClosestOneToOne(X1, X2):
    X = np.zeros((1, 3000, 128))
    Y = np.zeros((1, 3000, 128))
    X1length = X1.shape[0]
    for i in range(X1length):
        X2length = X2.shape[0]
        #maxDistance = 0
        maxScore = 0
        index = 0
        for k in range(X2length):
            #euclideanDistance = euclideanDistance(X1[i], X2[k])
            equalityScore = equalityCheck(X1[i], X2[k])
            if equalityScore == 1:
                pass
            elif equalityScore > maxScore:
                maxScore = equalityScore
                index = k
        toMatch = X1[i]       
        toMatch = np.expand_dims(toMatch, axis=0)
        X = np.append(X, toMatch, axis=0)
        match = X2[index]
        match = np.expand_dims(match, axis=0)
        Y = np.append(Y, match, axis=0)
        X2 = np.delete(X2, index, 0)
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)
    Z = list(zip(X, Y))  
    return Z


# In[17]:


def matchFurthestOneToOne(X1, X2):
    X = np.zeros((1, 3000, 128))
    Y = np.zeros((1, 3000, 128))
    X1length = X1.shape[0]
    for i in range(X1length):
        X2length = X2.shape[0]
        #maxDistance = 1000
        maxScore = 1
        index = 0
        for k in range(X2length):
            #euclideanDistance = euclideanDistance(X1[i], X2[k])
            equalityScore = equalityCheck(X1[i], X2[k])
            if equalityScore < maxScore:
                maxScore = equalityScore
                index = k
        toMatch = X1[i]       
        toMatch = np.expand_dims(toMatch, axis=0)
        X = np.append(X, toMatch, axis=0)
        match = X2[index]
        match = np.expand_dims(match, axis=0)
        Y = np.append(Y, match, axis=0)
        X2 = np.delete(X2, index, 0)
    X = np.delete(X, 0, 0)
    Y = np.delete(Y, 0, 0)
    Z = list(zip(X, Y))  
    return Z


# In[18]:


def matchOneToAnyNumberOfMatches(LongestArray, ShortestArray, NumberOfMatches):
    lengthOfLongest = len(LongestArray)
    lengthOfShortest = len(ShortestArray)
    
    matchedX = np.empty((lengthOfLongest*NumberOfMatches, LongestArray.shape[1], LongestArray.shape[2]))
    matchedY = np.empty((lengthOfLongest*NumberOfMatches, ShortestArray.shape[1], ShortestArray.shape[2]))
    
    amountFromTop = int(0.7*NumberOfMatches)
    amountFromMedium = int(0.2*NumberOfMatches)
    amountFromBottom = int(0.1*NumberOfMatches)
        
    for i in range(lengthOfLongest):
        Xroll = LongestArray[i]
        mapper = {}
        for k in range(lengthOfShortest):
            Yroll = np.array(ShortestArray[k])
            #distance = equalityCheck(Xroll, Yroll)
            #if(distance != 1): #Check if 1 to avoid matching with itself if placed in both folders
            #    mapper[k] = distance
            distance = sumCheck(Xroll, Yroll)
            if(distance != 0): #Check if 0 to avoid matching with itself if placed in both folders
                mapper[k] = distance
        sortedMap = {k: v for k, v in sorted(mapper.items(), key=lambda item: item[1])}
        
        listOfIndexes = []
        for index in sortedMap:
            listOfIndexes.append(index)
        
        lengthOfMatches = len(listOfIndexes)
        
        endOfTopBracket = int(0.1*lengthOfMatches)
        endOfmediumBracket = int(0.8*lengthOfMatches) 
        endOfbottomBracket = lengthOfMatches - 1
        
        matchedRolls = [] 
        
        topBracketSamples = random.sample(range(0, endOfTopBracket), amountFromTop)
        mediumBracketSamples = random.sample(range(endOfTopBracket, endOfmediumBracket), amountFromMedium)
        bottomBracketSamples = random.sample(range(endOfmediumBracket, endOfbottomBracket), amountFromBottom)
        
        for sampleIndex in topBracketSamples:
            matchedRolls.append(listOfIndexes[sampleIndex])
            
        for sampleIndex in mediumBracketSamples:
            matchedRolls.append(listOfIndexes[sampleIndex])
        
        for sampleIndex in bottomBracketSamples:
            matchedRolls.append(listOfIndexes[sampleIndex])
                                    
        for p in range(NumberOfMatches):
            index = i*NumberOfMatches + p                
            matchedX[index] = Xroll
            index_of_match = matchedRolls[p]
            matchedY[index] = ShortestArray[index_of_match]
             
    Z = list(zip(matchedX, matchedY))
    return Z


# In[20]:


def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


# In[ ]:




