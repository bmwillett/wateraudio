# imports 
###########################################

import numpy as np
import matplotlib.pyplot as plt
import random

import librosa
import librosa.display
import sounddevice as sd
from scipy.io import wavfile

from IPython.display import Audio, clear_output, display, Markdown
import time
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
import statistics

from scipy.stats import mode

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, ReLU, DepthwiseConv2D
from tensorflow.keras import regularizers

import os
os.chdir('/Users/bwillett/Documents/GitHub/wateraudio')


# global variables
###########################################

labels=['cold','hot']
num_labels=len(labels)
label2ind={labels[i]:i for i in range(num_labels)}

default_sr=22050   

all_clips={}

# classes
###########################################

class clip:
    def __init__(self,data,label,name,kind,sr=default_sr):
        self.data=data
        self.size=data.shape[0]
        self.sr=sr
        self.duration=self.size/self.sr
        self.label=label        
        self.name=name
        self.kind=kind
        
    # returns random sample from clip of 'duration' seconds
    def get_sample(self,duration):
        sample_size=int(self.sr*duration)
        
        if sample_size>self.size:
            return None
        
        pos=rand.randrange(self.size-sample_size)
            
        return self.data[pos:pos+sample_size]
    
def no_filter(x):
    return True

#  class that iterates over array of clips, returning consecutive samples of length 'duration' in seconds
#     returns tuple: sample, sr, label
class sample_generator: 
    def __init__(self, clips, duration, max_iters=float("inf"),random=False,clip_filter=no_filter,sample_filter=no_filter):
        self.clips = clips
        self.duration=duration
        self.max_iters=max_iters
        self.random=random
        self.sample_filter=sample_filter
        
        self.iters=0
        self.cur_clip=self.clips[0]
        self.cur_clip_ind=0
        self.cur_clip_pos=0
        
        # restrict to clips long enought to extract a sample of desired length
        self.sample_sizes=[]
        self.good_clips=[]
        for clip in self.clips:
            sample_size=int(clip.sr*self.duration)
            if sample_size<=clip.size and clip_filter(clip):
                self.good_clips.append(clip)
                self.sample_sizes.append(sample_size)

        self.clips=self.good_clips       
        
    def __iter__(self): 
        self.cur_clip_ind=0
        self.cur_clip=self.clips[0]
        self.cur_clip_pos=0
        self.iters=0
        return self
  
    def __next__(self):
        if self.iters>=self.max_iters:
            raise StopIteration
            
        if self.random:
            cur_clip=random.choice(self.clips)
            sample_size=int(cur_clip.sr*self.duration)
            start=random.randrange(cur_clip.size-sample_size)
            sample=cur_clip.data[start:start+sample_size]
            if not self.sample_filter(sample):
                return self.__next__()
            self.iters+=1
            return sample,cur_clip.sr,cur_clip.label
            
        while self.cur_clip_pos+self.sample_sizes[self.cur_clip_ind]>=self.cur_clip.size:
            self.cur_clip_ind+=1
            if self.cur_clip_ind==len(self.clips):
                raise StopIteration
            self.cur_clip=self.clips[self.cur_clip_ind]
            self.cur_clip_pos=0
        self.cur_clip_pos+=self.sample_sizes[self.cur_clip_ind]
        
        sample=self.cur_clip.data[self.cur_clip_pos-self.sample_sizes[self.cur_clip_ind]:self.cur_clip_pos]
        if not self.sample_filter(sample):
            return self.__next__()
        
        self.iters+=1
        return sample,self.cur_clip.sr,self.cur_clip.label

# class used to return predictions for audio samples
class predictor:
    def __init__(self,clf,getdata,params,sample_size,sr=default_sr,ohe=False):
        self.clf=clf
        self.getdata=getdata
        self.sample_size=sample_size
        self.params=params
        self.sample_duration=sample_size/sr
        self.ohe=ohe
        
    def test_samples(self,samples,return_mode=False):
        good_samples=[]
        for sample in samples:
            if sample.shape[0]==self.sample_size:
                good_samples.append(sample)
                
        if len(good_samples)==0:
            print("no samples of correct length")
            return
        samples=good_samples
        
        X_pred=[]
        for sample in samples:
            X_pred.append(self.getdata(sample,self.params))
            
        X_pred=np.array(X_pred)
        preds=self.clf.predict(X_pred)
    
        if self.ohe:
            preds=np.argmax(preds,axis=1)
        
        if return_mode:
            best_pred=int(np.array(mode(preds))[0][0])
            return labels[best_pred]
        else:
            return [labels[int(pred)] for pred in preds]
             
    # pick n_sample random samples from clip
    def record_and_test(self,duration=3,num_samples=7,sr=default_sr,progress=True):
        assert duration>self.sample_duration
        
        recording=get_new_audio(duration,progress=progress)
        wavfile.write('./pred/pred.wav',sr,recording)
        topred,_=librosa.load('./pred/pred.wav',mono=True)
        topred=0.1*topred/np.std(topred)
        
        clips=[clip(topred,None,None,None)]

        samples=[]
        for sample,_,_ in sample_generator(clips, self.sample_duration, max_iters=num_samples,random=True):
            samples.append(sample)

        return self.test_samples(samples,return_mode=True)
    
    def continuous_record_and_test(self,duration=10):
        start=time.time()
        while time.time()-start<duration:
            pred=self.record_and_test(duration=1.1*self.sample_duration,num_samples=1,progress=False)
            status="<font color='blue'>COLD</font>" if pred=='cold' else "<font color='red'>HOT</font>"
            clear_output(wait=True)
            display(Markdown("water is currently: "+status))
            time.sleep(0.25)
        clear_output(wait=True)
            
# methods
###########################################

# read data from train and test folders
def load_files():
    print("loading files...")
    global all_clips
    all_clips={}
    for label in labels:
        all_clips[label]=[]
        for kind in ['train','test']:
            path='./'+kind+'/'+label+'/'
            for filename in os.listdir(path):
                if filename[0]=='.':
                    continue
                try:
                    new_data,new_sr=librosa.load(path+filename,mono=True)
                    new_data=0.1*new_data/np.std(new_data)
                    all_clips[label].append(clip(new_data,label,filename,kind,sr=new_sr))
                except:
                    print("error opening ",filename)
                    pass

    print("loaded:")
    for label in labels:
        total_length=sum([c.duration for c in all_clips[label]])
        ntrain=len([0 for c in all_clips[label] if c.kind=='train'])
        ntest=len([0 for c in all_clips[label] if c.kind=='test'])
        print("for label={}, {} training clips and {} test clips with total length {} seconds".format(label,ntrain,ntest,int(total_length)))
    return all_clips
    
    
# records audio from microphone, returns numpy array

def get_new_audio(duration,sr=default_sr,progress=True):
    recording = np.squeeze(sd.rec(int(duration * sr), samplerate=sr, channels=1))
    if progress:
        for i in tqdm(range(50),desc='recording...',bar_format='{desc} {bar} {elapsed}'):
            time.sleep(duration/50)
    sd.wait()
    if np.isnan(np.sum(recording)):
        print("error: nan in recording")
    recording=0.1*recording/np.std(recording)
    return recording
    

# retrieves random sample of duration 'duration' (in seconds) from random clip with label 'label' 

def generate_sample(duration,label,kind='any',sample_filter=no_filter):
    inds=[i for i,c in enumerate(all_clips[label]) if duration<=c.duration and kind in [c.kind,'any']]
    if len(inds)==0:
        print("no clip of duration {} seconds with label {}".format(duration,label))
        return 
    
    cur_clip=all_clips[label][random.choice(inds)]
    
    sample_size=int(cur_clip.sr*duration)
    start=random.randrange(cur_clip.size-sample_size)
    sample= cur_clip.data[start:start+sample_size]
    
    if not sample_filter(sample):
        return generate_sample(duration,label,kind=kind,sample_filter=sample_filter)
            
    return sample,cur_clip.sr
    
# records audio from microphone, returns numpy array

def get_new_audio(duration,sr=default_sr,progress=True):
    recording = np.squeeze(sd.rec(int(duration * sr), samplerate=sr, channels=1))
    if progress:
        for i in tqdm(range(50),desc='recording...',bar_format='{desc} {bar} {elapsed}'):
            time.sleep(duration/50)
    sd.wait()
    if np.isnan(np.sum(recording)):
        print("error: nan in recording")
    recording=0.1*recording/np.std(recording)
    return recording