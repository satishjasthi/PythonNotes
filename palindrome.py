from __future__ import print_function, division
from structshape import structshape
import random
import bisect
import string
import time
import bisect

def count_letter_occur(word,letter):
	asci_let = ord(letter.lower())
	count = 0
	word = word.lower()
	for each in word:
		if ord(each) == asci_let:	
			count += 1
	return count

def count_letter_occur_dict(word,letter):
	d = {}
	for each in word.lower():
		if each not in d:
			d[each] = 1
		else:
			d[each] += 1
	return d


def count_letter_occur_get(word,letter):
	d = {}
	for each in word.lower():
		if d.get(each) == None:
			d[each] = 1
		else:
			d[each] += 1
		
		
	return d

def reverse_lookup(dictionary,value):
	for key in dictionary:
		if dictionary[key] == value:
			return key
	raise LookupError()

def reverse_lookup_all_values(dic,value):
	values_list = []
	for key in dic:
		if dic[key] == value:
			values_list.append(key)
	return values_list

def invert_dict(d):
     inverse = dict()
     for key in d:
             val = d[key]
             print ("value,key: {}".format(val,key))
             if val not in inverse:
                     inverse[val] = [key]
                     print ('inverse[val],[key]:',inverse[val],[key])
             else:
                     inverse[val].append(key)
                     print('inverse[val].append(key)',inverse[val].append(key)) 
     return inverse

known = {0:0,1:1}

def fibonacci(n):
	if n in known:
		print(known)
		return known[n]
	res = fibonacci(n-1) + fibonacci(n - 2)
	known[n] = res
	return res
def create_a_wordlist(filename):
	wordlist = []
	with open(filename) as f:
		for each in f:
			wordlist.append(each.strip())
	return wordlist
	

def wordlist_to_dict(wordlist):
	d = {}
	for each in wordlist:
		d[each] = None
	return d

def in_bisect(wordlist,target):
	t1 = time.time()
	i = bisect.bisect_left(wordlist,target)
	result = target == wordlist[i]
	t2 = time.time()
	T = t2-t1
	return(result,T)

def in_bisect_dict(dict_wordlist,target):
	t1 = time.time()
	result = target in dict_wordlist
	t2 = time.time()
	T = t2-t1
	return(result,T) 

def inverse_dict(dictionary):
	inverse = {}
	for key in dictionary:
		value = dictionary[key]
		inverse.setdefault(value,[]).append(key)
	return inverse

v = {0:'n+1'}

def ackermann(m,n):
	if m == 0:
		return n + 1
	if m>0 and n==0:
		return A(m-1,1)
	if m>0 and n>0:
		return A(m-1,A(m,n-1))



cache = {}

def ackermann(m, n):
    """Computes the Ackermann function A(m, n)

    See http://en.wikipedia.org/wiki/Ackermann_function

    n, m: non-negative integers
    """
    if m == 0:
        return n+1
    if n == 0:
        return ackermann(m-1, 1)
    try:
	print('Try cache[m, n] : {}'.format(cache[m, n]))
        return cache[m, n]
    except KeyError:
        cache[m, n] = ackermann(m-1, ackermann(m, n-1))
	print('except cache[m, n]: {}'.format(cache[m, n]))
        return cache[m, n]



def histogram(t):
	'Creates a histogram of values of list'
	d = {}
	for each in t:
		d.setdefault(each,[]).append(1)
		
	for key in d:
		d[key] = len(d[key])
	return d

def has_duplicate(t):
	d = histogram(t)
	for key in d:
		if d[key] > 1:
			return True
	return False

def rotate_words(word,integer):
	letters = list(word.lower())
	for index,each in enumerate(letters):
		letters[index] = chr(ord(letters[index]) + integer)
	return(''.join(letters))

def rotate_words_dict(filename,integer):
	d = wordlist_to_dict(create_a_wordlist(filename))
	d_new = {}
	print (d)
	for key in d:
		d_new[rotate_words(key,integer)] = random.randint(1,1000)
	return d_new
		
def convert_homophonesFile_to_dict(filename):
	'Converts the homophones to a dictionary with word as key and pronouncitation as value'
	d = {}
	with open(filename) as fin:
		for each in fin :
			d[each.strip().split('  ')[0]] = each.strip().split('  ')[1]
		return d

def find_homophones(d):
	dnew = {}
	for key in d:
		string1 = str(key[1::])
		string2 = str(key[0] + key[2::])
		if d.get(string1) == d[key] and d.get(string2) == d[key]:
			dnew[key] = d[key]
	return dnew



#d = convert_homophonesFile_to_dict('homophones.txt')

def sumall(*args):
	return sum(tuple(args))


def histogram(string):
	'Maps each letter of word to its frequency'
	d = {}
	for each in string:
		d[each] = d.get(each,0) + 1
	return d


def most_frequent(string):
	'Returns the letters of string with decreasing order of their respective frequencies'
	d = histogram(string)
	t,result = [], []
	for letter,freq in d.items():
		t.append([freq,letter])
	t = sorted(t,reverse = True)
	for freq,letter in t:
		result.append(letter)
	return result
d = histogram('Mississippia')


def create_a_wordlist(filename):
	'creates a wordlist from a txt file,where we have one word/line'
	wordlist = []
	with open(filename) as f:
		for each in f:
			wordlist.append(each.strip())
	return wordlist

def wordlist_to_dict(wordlist):
	'Converts given word list a dict with keys as words and random number as values'
	d = {}
	for each in wordlist:
		d[each] = random.randint(1,10000)
	return d

def signature(s):
    """Returns the signature of this string.

    Signature is a string that contains all of the letters in order.

    s: string
    """
    # TODO: rewrite using sorted()
    t = list(s)
    t = sorted(t)
    t = ''.join(t)
    return t


def all_anagrams(filename):
    """Finds all anagrams in a list of words.

    filename: string filename of the word list

    Returns: a map from each word to a list of its anagrams.
    """
    d = {}
    for line in open(filename):
        word = line.strip().lower()
        t = signature(word)

        # TODO: rewrite using defaultdict
        if t not in d:
            d.setdefault(t,[]).append(word)
        else:
            d[t].append(word)
    return d


def print_anagram_sets(d):
    """Prints the anagram sets in d.

    d: map from words to list of their anagrams
    """
    for v in d.values():
        if len(v) > 1:
            print(len(v), v)

d = all_anagrams('words355k.txt')



def print_anagram_sets_in_order(d):
    """Prints the anagram sets in d in decreasing order of size.

    d: map from words to list of their anagrams
    """
    # make a list of (length, word pairs)
    t = []
    for v in d.values():
        if len(v) > 1:
            t.append((len(v), v))

    # sort in ascending order of length
    t.sort()

    # print the sorted list
    for x in t:
        print(x)

#print(print_anagram_sets_in_order(d))

def filter_length(d, n):
    """Select only the words in d that have n letters.

    d: map from word to list of anagrams
    n: integer number of letters

    returns: new map from word to list of anagrams
    """
    res = {}
    for word, anagrams in d.items():
        if len(word) == n:
            res[word] = anagrams
    return res


def signature_of_word(s):
	'Returns the alphabetically sorted word form given word '
	's: string'
	t = list(s)
	t = ''.join(sorted(t)) 
		
	return t

def anagrams_list(filename):
	"""Finds all the anangrams of a word from a wordlist
	   filename: name of the file containing wordlist
	   returns a map of each word to a list of its anagrams """
	d = {}
	for each in open(filename):
		each = each.strip().lower()
		signature = signature_of_word(each)
		if signature not in d:
			d.setdefault(signature,[]).append(each)
		else: 
			d.setdefault(signature,[]).append(each)
	return d

def sort_anagrams_list(anagrams_list):
	'Sorts the given anagram list in decreasing order'
	result = []
	for each in anagrams_list:
		if len(anagrams_list[each]) > 1:
			result.append( (len(anagrams_list[each]),each,anagrams_list[each]))
	result.sort()
	return result
		

def eigth_letter_words(filename):
	'''Returns all seven letter words from a file containing a list of words'''
	result = ()
	for each in open(filename):
		each = each.strip().lower()
		if len(each) == 8:result = result + (each,)
	return result


def histogram(filename):
	'returns the frequency of each eight letter word'
	words = eigth_letter_words(filename)
	d = {}
	result = []
	for each in words:
		sign = signature(each)
		if signature not in d: d.setdefault(sign,[]).append(each)
		else: d.setdefault(sign,[]).append(each)
	for each in d:
		if len(d[each]) > 1:
			l = tuple([len(d[each]),each,d[each]])
			result.append(l)
			result.sort()
	return result

def metathesis_pairs(d):
    """Print all pairs of words that differ by swapping two letters.

    d: map from word to list of anagrams
    """
    for anagrams in d.values():
        for word1 in anagrams:
            for word2 in anagrams:
                if word1 < word2 and word_distance(word1, word2) == 2:
                    print(word1, word2)


def word_distance(word1, word2):
    """Computes the number of differences between two words.

    word1, word2: strings

    Returns: integer
    """
    assert len(word1) == len(word2)

    count = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            count += 1

    return count
d = all_anagrams('words355k.txt')

#print ( metathesis_pairs(d) )


def sentence2word(filename):
	'returns wordlist from sentences'
	result = []
	with open(filename) as fin:
		for line in fin:
			line = line.translate(None,string.punctuation)
			line = line.strip()
			result.append(line)
	result = ''.join(result)
	result = result.split(' ')
	return result

def histogram_of_words(wordlist):
	'returns frequency of each word in word list'
	d = {}
	for word in wordlist:
		if word not in d:
			d[word] = d.get(word,0) + 1
		else:	
			d[word] = d.get(word,0) + 1
	return d

def sort_wordlist_based_on_freq(word_dict):
	'creates a sorted list of words based on their frequency'
	result = []
	for key in word_dict:
		result.append((word_dict[key],key))
	result.sort()
	return result

words = sentence2word('theoceanoftheosophy.txt')

wordhistogram = histogram_of_words(words)

sortedwordhistogram = sort_wordlist_based_on_freq(wordhistogram)

#print(sort_wordlist_based_on_freq(wordhistogram))


def total_num_words(filename):
	'count total number of words in a text file'
	words = sentence2word(filename)
	return len(words)

#print(total_num_words('theoceanoftheosophy.txt'))



def twenty_most(filename):
	'returns the 20 most frequent words in file'
	words = sentence2word(filename)
	wordhistogram = histogram_of_words(words)
	sortedwordhistogram = sort_wordlist_based_on_freq(wordhistogram)
	
	return(sortedwordhistogram[-20:],structshape(sortedwordhistogram))

#print(twenty_most('theoceanoftheosophy.txt'))

def histogram(wordlist):
	'returns histogram of wordlist'
	d = {}
	for each in wordlist:
		d[each] = d.setdefault(each,0) + 1
	return d


def choose_from_list(histogram):
	'returns the probability of each word in a given histogram'
	for key in histogram:
		histogram[key] = histogram[key]/sum(histogram.values())
	keys = histogram.keys()
	word = random.choice(keys)
	return(word,histogram[word])	

def process_file(filename):
	'converts a txt file to a dict of words'
	hist = {}
	for line in open(filename):
		line = process_line(line,hist)
	return hist

def process_line(line,hist = {}):
	'''converts a line of text to a dict of words by removing
	punctuation
	line : each line from text file
	hist : word histogram '''
	
	line = line.replace('-','').strip().lower()
	for word in line.split():
		word = word.strip(string.punctuation + string.whitespace)
		hist[word] = hist.get(word,0) + 1

hist = process_file('emma.txt')
#print( hist )

def most_common(hist):
	'returns most frequent words from word histogram'
	r = []
	for key,value in hist.items():
		r.append((value,key))
	r.sort(reverse = True)
	return(r[:10])

#print( most_common(hist) )

p = most_common(hist)

#print('The most common words are : \n')
#for freq,word in p[:10]:
#	print(freq,word,sep = '\t')



w1 = process_file('words.txt')
w2 = process_file('emma.txt')

wordlist = set(w2.keys()).difference(set(w1.keys()))

#print(wordlist)

def cumulative_histogram(filename):
	'returns the cumulative frequency of words from a text file'
	j = {}
	total = 0
	word_dict_hist = process_file(filename)
	for key in word_dict_hist:
		j[key] = j.setdefault(key,0) + 1 + total
		total = j[key]
	#j = sorted(j,key = lambda keys:j[keys] ,reverse = True)
	return j

#print( cumulative_histogram('emma.txt') )

ch = cumulative_histogram('emma.txt')

def random_bisect(filename):
	'finds the index of randomly generated cumulative sum'
	hist = cumulative_histogram(filename)


#---------------------------------------------------------------
	

def process_file_tuple(filename):
	'converts a text file into a tuple of words'
	t = []
	for line in open(filename):
		line = line.strip().lower()
		for word in line.split():
			word = word.strip(string.punctuation + string.whitespace)
			t.append(word)			
	return tuple(t)



word_tuples =  process_file_tuple('emma.txt')

#print('Completed word_tuples')

def word_pair_tuples(word_tuples):
	'returns a word-pair tuples'
	i = 0
	p = []
	while i <len(word_tuples) - 1:
		p.append((word_tuples[i],word_tuples[i+1]))
		i += 1
	return tuple(p)

wp = word_pair_tuples(word_tuples)
print (wp)

def markov_dic(wp):
	'creates a markov mapping using word pair tuples'
	m = {}
	for index,each in enumerate(wp):
		key = ' '.join(each)
		value = ' '.join(wp[index + 1])
		m.setdefault(key,[]).append(value)
		if index == len(wp) - 2: break		
	return m

md = markov_dic(wp)

def create_a_random_sentence(md):
	'creates a random sentence from markov dict'
	count = 0
	sentence = []
	for each in md:
		maxm = len(md[each])
		index = random.randint(0,maxm-1)
		sentence2 = md[each][index]
		output = each + ' ' + sentence2
		sentence.append(output)
		count += 1
		if count == 50:break
	for each in sentence:
		print(each,end = ' ' )
	

print(create_a_random_sentence(md))	











