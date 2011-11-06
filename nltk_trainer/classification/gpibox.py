from nltk.classify import ClassifierI
import math
import collections
import random

def sgn (x):
	return (x>0) - (x<0)
	
def zero():
	return 0

class GPIClassifier(ClassifierI):
	def __init__(self, w, target_names):
		self.w = w
		self.target_names = target_names
	
	def labels(self):
		return self.target_names
	
	def classify_numerical(self, featureset):
		s = 0.0
		for token in featureset.iterkeys():
			s += self.w[token]
			
		return 1.0/(1.0 + math.exp(-s))
		
	def classify(self, featureset):
		s = self.classify_numerical(featureset)
		return self.target_names[int(round(s))]
	
	@classmethod
	def train(cls, labeled_featuresets, aggressiveness, passivity):
		
		target_names =  set([])
		random.shuffle(labeled_featuresets)
		
		for (fs, label) in labeled_featuresets:
			target_names.add(label)
			if len(target_names) >= 2:
				break
				
		target_names = sorted(target_names)
		
		w = collections.defaultdict(zero)
		
		scl = cls(w, target_names)
		
		for (featureset, label) in labeled_featuresets:
			o = scl.classify_numerical(featureset)
			t = target_names.index(label)
			error = t - o
			abserror = math.fabs(error)
			if abserror > passivity:
				l = len(featureset)
				z = l + 0.5 / aggressiveness
				change = sgn(error)* (abserror - passivity) / z
		
				for token in featureset.iterkeys():
					scl.w[token] += change
		
		return scl
