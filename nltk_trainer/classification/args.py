from nltk.classify import DecisionTreeClassifier, MaxentClassifier, NaiveBayesClassifier
from nltk_trainer.classification.multi import AvgProbClassifier
from .gpibox import GPIClassifier
import nltk.data
from nltk.classify.weka import WekaClassifier

classifier_choices = ['NaiveBayes', 'DecisionTree', 'Maxent', 'GPIBox'] + MaxentClassifier.ALGORITHMS

try:
	from .sci import ScikitsClassifier
	classifier_choices.append('Scikits')
except ImportError:
	pass
	
try:
	from nltk.classify.weka import WekaClassifier
	classifier_choices.append('Weka')
except ImportError:
	pass

def add_maxent_args(parser):
	maxent_group = parser.add_argument_group('Maxent Classifier',
		'These options only apply when a Maxent classifier is chosen.')
	maxent_group.add_argument('--max_iter', default=10, type=int,
		help='maximum number of training iterations, defaults to %(default)d')
	maxent_group.add_argument('--min_ll', default=0, type=float,
		help='stop classification when average log-likelihood is less than this, default is %(default)d')
	maxent_group.add_argument('--min_lldelta', default=0.1, type=float,
		help='''stop classification when the change in average log-likelihood is less than this.
	default is %(default)f''')

def add_decision_tree_args(parser):
	decisiontree_group = parser.add_argument_group('Decision Tree Classifier',
		'These options only apply when the DecisionTree classifier is chosen')
	decisiontree_group.add_argument('--entropy_cutoff', default=0.05, type=float,
		help='default is 0.05')
	decisiontree_group.add_argument('--depth_cutoff', default=100, type=int,
		help='default is 100')
	decisiontree_group.add_argument('--support_cutoff', default=10, type=int,
		help='default is 10')

def add_gpibox_args(parser):
	gpi_group = parser.add_argument_group('Google Priority Inbox imitation Classifier',
		'These options only apply when the GPIBox classifier is chosen')
	gpi_group.add_argument('--aggressiveness', default=0.05, type=float, help='default is 0.05')
	gpi_group.add_argument('--passivity', default=0.2, type=float, help='default is 0.05')

def make_classifier_builder(args):
	if isinstance(args.classifier, basestring):
		algos = [args.classifier]
	else:
		algos = args.classifier
	
	for algo in algos:
		if algo not in classifier_choices:
			raise ValueError('classifier %s is not supported' % algo)
	
	classifier_train_args = []
	
	for algo in algos:
		classifier_train_kwargs = {}
		
		if algo == 'DecisionTree':
			classifier_train = DecisionTreeClassifier.train
			classifier_train_kwargs['binary'] = False
			classifier_train_kwargs['entropy_cutoff'] = args.entropy_cutoff
			classifier_train_kwargs['depth_cutoff'] = args.depth_cutoff
			classifier_train_kwargs['support_cutoff'] = args.support_cutoff
			classifier_train_kwargs['verbose'] = args.trace
		elif algo == 'NaiveBayes':
			classifier_train = NaiveBayesClassifier.train
		elif algo == 'Scikits':
			classifier_train = ScikitsClassifier.train
		elif algo == 'GPIBox':
			if args.senior and 'GPIBox' in args.senior:
				senior = nltk.data.load("classifiers/" + args.senior)
				classifier_train_kwargs['senior'] = senior.w
			classifier_train_kwargs['aggressiveness'] = args.aggressiveness
			classifier_train_kwargs['passivity'] = args.passivity
			classifier_train = GPIClassifier.train
		elif algo == 'Weka':
			classifier_train_kwargs['classifier'] = 'C4.5'
			classifier_train_kwargs['model_filename'] = '/tmp/wekarun.model'
			def call_train_weka(train_feats, **train_kwargs):
				return WekaClassifier.train(train_kwargs['model_filename'], train_feats, 'C4.5')
			classifier_train = call_train_weka
		else:
			if algo != 'Maxent':
				classifier_train_kwargs['algorithm'] = algo
			
			classifier_train = MaxentClassifier.train
			classifier_train_kwargs['max_iter'] = args.max_iter
			classifier_train_kwargs['min_ll'] = args.min_ll
			classifier_train_kwargs['min_lldelta'] = args.min_lldelta
			classifier_train_kwargs['trace'] = args.trace
		
		classifier_train_args.append((algo, classifier_train, classifier_train_kwargs))
	
	def trainf(train_feats):
		classifiers = []
		
		for algo, classifier_train, train_kwargs in classifier_train_args:
			if args.trace:
				print 'training %s classifier' % algo
			
			classifiers.append(classifier_train(train_feats, **train_kwargs))
		
		if len(classifiers) == 1:
			return classifiers[0]
		else:
			return AvgProbClassifier(classifiers)
	
	return trainf
	#return lambda(train_feats): classifier_train(train_feats, **classifier_train_kwargs)
