import nltk, string, collections, math, re
import networkx as nx
import itertools
import numpy as np

def extract_words(text):
    punct = string.punctuation
    stopWords = set(nltk.corpus.stopwords.words('english'))

    goodtags = ['JJ','JJR','JJS','NN','NNP','NNS','NNPS']
    words = nltk.word_tokenize(text)
    wordlist = nltk.pos_tag(words)

    candidates = []
    for tags in wordlist:
        if tags[0].lower() not in stopWords and tags[1] in goodtags and not all(char in punct for char in tags[0]):
            candidates.append(tags[0].lower())
            
    for word in candidates:
        if word in punct or len(word) < 2:
            candidates.remove(word)

    return candidates

def score_keyphrases(text):
    words = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            words.append(word.lower())
    candidates = extract_words(text)
    graph = nx.Graph()
    graph.add_nodes_from(candidates)
    for w1, w2 in pairwise(candidates):
        graph.add_edge(w1, w2)

    scores = nx.pagerank(graph)
    ranks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    keywords = []
    for i in range(20):
        keywords.append(ranks[i][0])

    return keywords
##    keyphrases = {}
##    j = 0
##    for count, word in enumerate(words):
##        if count < j:
##            continue
##        if word in keywords:
##            kp_words = list(itertools.takewhile(lambda x: x in keywords, words[count:count+10]))
##            avgrank = sum(ranks[k][1] for k in range(len(kp_words)))/len(kp_words)
##            keyphrases[' '.join(kp_words)] = avgrank
##            j = count + len(kp_words)
##
##    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def sentence_similarity(sent1, sent2, stopwords):
    sent1 = [word.lower() for word in sent1]
    sent2 = [word.lower() for word in sent2]

    allwords = list(set(sent1 + sent2))

    vector1 = [0] * len(allwords)
    vector2 = [0] * len(allwords)

    for w in sent1:
        if w in stopwords:
            continue
        vector1[allwords.index(w)] += 1

    for w in sent2:
        if w in stopwords:
            continue
        vector2[allwords.index(w)] += 1

    return 1-nltk.cluster.util.cosine_distance(vector1, vector2)
    
    

def similarity_matrix(text):
    sentences = nltk.sent_tokenize(text)
    similaritymatrix = np.zeros((len(sentences), len(sentences)))

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similaritymatrix[i][j] = sentence_similarity(sentences[i], sentences[j], nltk.corpus.stopwords.words('english'))

    return similaritymatrix

def generate_summary(text, num_sentences):
    sentences = nltk.sent_tokenize(text)
    text_summary = []
    sentence_similarity_matrix = similarity_matrix(text)

    graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for i in range(num_sentences):
        index = ranked_sentences[i][0]
        text_summary.append(sentences[index] + ". ")

    return text_summary

##def candidate_features(candidates, text, excerpt, title):
##    candidate_scores = collections.OrderedDict()
##    
##    ctr = collections.Counter()
##    for sent in nltk.sent_tokenize(text):
##        for word in nltk.word_tokenize(sent):
##            ctr.update(word.lower())
##
##    for candidate in candidates:
##        pattern = re.complile(r'\b' + re.escape(candidate) + r'(\b|[,.;?!]|\s)', re.IGNORECASE)
##
##        cand_occur = len(pattern.findall(text))
##
##        if cand_occur == 0:
##            print(candidate + ' not found!')
##            continue
##
##        candidate_words = candidate.split()
##        max_word_length = max(len(word) for word in candidate_words)
##        term_length = len(candidate_words)
##        wordcountsum = sum(ctr[w] for w in candidate_words)
##        if term_length == 1:
##            lexical_cohesion = 0
##        else:
##            lexical_cohesion = term_length * (1 + math.log(cand_occur, 10)) * cand_occur/wordcountsum
##
##        if pattern.search(title):
##            in_title = 1
##        else:
##            in_title = 0
##
##        if pattern.search(excerpt):
##            in_excerpt = 1
##        else:
##            in_excerpt = 0
##
##        textlength = len(text)
##        first_match = pattern.search(text)
##        first_occur = first_match.start()/textlength
##        if cand_occur == 1:
##            spread = 0
##            last_occur = first_occur
##        else:
##            last_match = text.rfind(pattern)
##            last_occur = last_match.start()/textlength
##            spread = last_occur - first_occur
##
##        candidate_scores[candidate] = {'term count': cand_occur,
##                                       'term length': term_length,
##                                       'max word length': max_word_length
##                                       'spread': spread,
##                                       'lexical cohesion': lexical_cohesion,
##                                       'in excerpt': in_excerpt,
##                                       'in title': in_title,
##                                       'first occurence': first_occur,
##                                       'last occurence': last_occur}
##    return candidate_scores
##                                       
