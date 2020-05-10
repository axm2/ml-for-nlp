#!/usr/bin/env python
# coding: utf-8

# ### Imports and logging
# 
# First, we start with our imports and get logging established:

# In[1]:


# imports needed and set up logging
import bz2
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


data_file="news.crawl.bz2"

with bz2.open ('news.crawl.bz2', 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break


# ### Read files into a list
# Now that we've had a sneak peak of our dataset, we can read it into a list so that we can pass this on to the Word2Vec model. Notice in the code below, that I am directly reading the 
# compressed file. I'm also doing a mild pre-processing of the reviews using `gensim.utils.simple_preprocess (line)`. This does some basic pre-processing such as tokenization, lowercasing, etc and returns back a list of tokens (words). Documentation of this pre-processing method can be found on the official [Gensim documentation site](https://radimrehurek.com/gensim/utils.html). 
# 
# 

# In[3]:



def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with bz2.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")    


# ## Training the Word2Vec model
# 
# Training the model is fairly straightforward. You just instantiate Word2Vec and pass the reviews that we read in the previous step (the `documents`). So, we are essentially passing on a list of lists. Where each list within the main list contains a set of tokens from a user review. Word2Vec uses all these tokens to internally create a vocabulary. And by vocabulary, I mean a set of unique words.
# 
# After building the vocabulary, we just need to call `train(...)` to start training the Word2Vec model. Training on the [OpinRank](http://kavita-ganesan.com/entity-ranking-data/) dataset takes about 10 minutes so please be patient while running your code on this dataset.
# 
# Behind the scenes we are actually training a simple neural network with a single hidden layer. But, we are actually not going to use the neural network after training. Instead, the goal is to learn the weights of the hidden layer. These weights are essentially the word vectors that we’re trying to learn. 

# In[4]:


model = gensim.models.Word2Vec (documents, size=150, window=5, min_count=2, workers=8, iter=10)


# Q1: Report similarity scores for the following pairs: (dirty,clean),(big,dirty),(big,large),(big,small)

# In[5]:


model.wv.similarity('dirty','clean')


# In[6]:


model.wv.similarity('big','dirty')


# In[7]:


model.wv.similarity('big','large')


# In[8]:


model.wv.similarity('big','small')


# Q2: Report 5 most similar items and the scores to 'polite', 'orange'

# In[9]:


w1 = ["polite"]
model.wv.most_similar(w1,topn=5)


# In[10]:


w1 = ["orange"]
model.wv.most_similar(w1,topn=5)


# Q3: Now change the parameters of your model as follows: window=2, size=50. Answer the 2 questions above for this new model.

# In[11]:


model2 = gensim.models.Word2Vec (documents, size=50, window=2, min_count=2, workers=8, iter=10)


# In[12]:


model2.wv.similarity('dirty','clean')


# In[13]:


model2.wv.similarity('big','dirty')


# In[14]:


model2.wv.similarity('big','large')


# In[15]:


model2.wv.similarity('big','small')


# In[16]:


w1 = ["polite"]
model2.wv.most_similar(w1,topn=5)


# In[17]:


w1 = ["orange"]
model2.wv.most_similar(w1,topn=5)

