Analyzing Tagger Coverage
-------------------------

The ``analyze_tagger_coverage.py`` script will run a part-of-speech tagger on a corpus to determine how many times each tag is found. Example output can be found in `Analyzing Tagged Corpora and NLTK Part of Speech Taggers <http://streamhacker.com/2011/03/23/analyzing-tagged-corpora-nltk-part-speech-taggers/>`_.

Here's an example using the NLTK default tagger on the treebank corpus::
	``python analyze_tagger_coverage.py treebank``

To get detailed metrics on each tag, you can use the ``--metrics`` option. This requires using a tagged corpus in order to compare actual tags against tags found by the tagger. See `NLTK Default Tagger Treebank Tag Coverage <http://streamhacker.com/2011/01/24/nltk-default-tagger-treebank-tag-coverage/>`_ and `NLTK Default Tagger CoNLL2000 Tag Coverage <http://streamhacker.com/2011/01/25/nltk-default-tagger-conll2000-tag-coverage/>`_ for examples and statistics.

To analyze the coverage of a different tagger, use the ``--tagger`` option with a path to the pickled tagger::
	``python analyze_tagger_coverage.py treebank --tagger /path/to/tagger.pickle``

To analyze coverage on a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.tagged.TaggedCorpusReader-class.html>`_::
	``python analyze_tagger_coverage.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

For a complete list of usage options::
	``python analyze_tagger_coverage.py --help``
