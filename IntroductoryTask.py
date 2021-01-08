#!/usr/bin/env python
# coding: utf-8

# In[34]:


#making a list of all reviews
def combine_reviews(mypath):
    #calls method to make a list of file addresses for each review
    review_locations = find_locations(mypath)
    #calls method to make list of all reviews
    return make_list_of_reviews(mypath, review_locations)


# In[35]:


# make a list of file addresses for each review
def find_locations(mypath):
    from os import listdir
    from os.path import isfile, join
    review_locations = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return review_locations


# In[36]:


#given a list of all review file addresses, makes a list of all reviews
def make_list_of_reviews(mypath, review_locations):
    reviews = []
    for file in review_locations:
        f = open(mypath + "/" + file,encoding='utf-8')
        current_review = f.read()
        #make lower case
        current_review = current_review.lower()
        #remove punctuation
        from string import punctuation
        current_review = ''.join([c for c in current_review if c not in punctuation])
        reviews.append(current_review)
        f.close()    
    return reviews


# In[37]:


positive_train_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/train/pos")


# In[38]:


positive_train_reviews[0:3]


# In[ ]:


negative_train_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/train/neg")
negative_train_reviews


# In[40]:


negative_train_reviews[0:3]


# In[41]:


all_train_reviews = positive_train_reviews+negative_train_reviews


# In[42]:


all_train_reviews[0:3]


# In[43]:


print("Number of positive training reviews: ",len(positive_train_reviews))
print("Number of negative training reviews: ", len(negative_train_reviews))
print("Total number of training reviews: ", len(all_train_reviews))


# In[44]:


#encoding reviews
def encode_reviews(positive_reviews, negative_reviews):
    #combine negative and positive reviews into one lsit
    all_reviews = positive_reviews + negative_reviews
    sorted_words = sort_words_by_FreqDist(all_reviews)
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
    reviews_int = encode_words_as_integers(all_reviews, vocab_to_int)
    return reviews_int


# In[45]:


def sort_words_by_FreqDist(all_reviews):
    import nltk
    all_text2 = ''.join(all_reviews)
    words = all_text2.split()
    #make an freqDist object to count all word frequencies
    fd = nltk.FreqDist(words)
    #use freqDist object to make a dictionary of words ordered by word frequencies
    #len(words) ensures all words are ordered
    sorted_words = fd.most_common(len(words))
    return sorted_words    


# In[46]:


def encode_words_as_integers(all_reviews, vocab_to_int):
    reviews_int = []
    for review in all_reviews:
        #for every word, find corresponding integer in vocab_to_int encoding
        r = [vocab_to_int[w] for w in review.split() if w in vocab_to_int]
        reviews_int.append(r)
    return reviews_int


# In[47]:


def encode_labels(num_positive, num_negative):
    encoded_labels = []
    positive = 1
    negative = 0
    #recall first 12500 reviews in allreviews were positive
    #second 12500 reviews were negative
    for x in range(num_positive):
        encoded_labels.append(positive)
    for x in range(num_negative):
        encoded_labels.append(negative)
    return encoded_labels


# In[48]:


#recall first 12500 reviews in allreviews were positive
# second 12500 reviews were negative
train_labels = encode_labels(12500, 12500)


# In[ ]:


positive_train_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/train/pos")
negative_train_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/train/neg")
encoded_train_reviews = encode_reviews(positive_train_reviews, negative_train_reviews)
encoded_train_reviews


# In[50]:


encoded_train_reviews[0:2]


# In[ ]:


positive_test_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/test/pos")
negative_test_reviews = combine_reviews("/Users/pugal/LeaseResearch/aclImdb/test/neg")
encoded_test_reviews = encode_reviews(positive_test_reviews, negative_test_reviews)


# In[52]:


encoded_test_reviews[0:2]


# In[53]:


print("Number of positive testing reviews: ",len(positive_test_reviews))
print("Number of negative testing reviews: ", len(negative_test_reviews))
#25000 reviews, but website says 25001
print("Total number of testing reviews: ", positive_test_reviews + negative_test_reviews)


# In[54]:


#making histogram of number of words in each training review
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
encoded_train_reviews_len = [len(x) for x in encoded_train_reviews]
pd.Series(encoded_train_reviews_len).hist()
plt.show()
pd.Series(encoded_train_reviews_len).describe()    


# In[55]:


encoded_train_reviews_len[0:3]


# In[56]:


print("min isn't 0 so don't need to remove short reviews")


# In[57]:


#Step 9a: filter reviews
def filter_reviews(reviews_int, small_cutoff, large_cutoff, reviews_len):
    return [reviews_int[i] for i, l in enumerate(reviews_len) if l< large_cutoff and l > small_cutoff ]


# In[58]:


#Step 9b: filter labels
def filter_labels(labels, small_cutoff, large_cutoff, reviews_len):
    return [labels[i] for i, l in enumerate(reviews_len) if l< large_cutoff and l > small_cutoff ]


# In[59]:


#only have labels for reviews between 0 and 500 words
filtered_train_labels = filter_labels(train_labels, 0, 500, encoded_train_reviews_len)
print(filtered_train_labels[0:3])
print(len(filtered_train_labels))


# In[60]:


#only have reviews between 0 and 500 words
filtered_train_reviews = filter_reviews(encoded_train_reviews, 0, 500, encoded_train_reviews_len)
get_ipython().run_line_magic('matplotlib', 'inline')
filtered_train_reviews_len = [len(x) for x in filtered_train_reviews]
pd.Series(filtered_train_reviews_len).hist()
plt.show()
pd.Series(filtered_train_reviews_len).describe()  


# In[61]:


#Step 10 Padding/Truncating the remaining data
def padding_truncating(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    import numpy as np
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features


# In[62]:


final_encoded_training_data = padding_truncating(filtered_train_reviews,200)
final_encoded_training_data[0:3]


# In[63]:


# let's visualise the encoded test reviews
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
encoded_test_reviews_len = [len(x) for x in encoded_test_reviews]
pd.Series(encoded_test_reviews_len).hist()
plt.show()
pd.Series(encoded_test_reviews_len).describe()
#use similar cutoffs for encoded test reviews


# In[64]:


#only have test reviews between 0 and 500 words
filtered_test_reviews = filter_reviews(encoded_test_reviews, 0, 500, encoded_test_reviews_len)
get_ipython().run_line_magic('matplotlib', 'inline')
filtered_test_reviews_len = [len(x) for x in filtered_test_reviews]
pd.Series(filtered_test_reviews_len).hist()
plt.show()
pd.Series(filtered_test_reviews_len).describe()  


# In[65]:


final_encoded_testing_data = padding_truncating(filtered_test_reviews,200)
final_encoded_testing_data[0:3]


# In[66]:


#only have labels for reviews between 0 and 500 words
test_labels = encode_labels(12500, 12500)
filtered_test_labels = filter_labels(test_labels, 0, 500, encoded_test_reviews_len)
print(filtered_test_labels[0:3])
print(len(filtered_test_labels))


# In[67]:


#setting aside 20% of the training reviews to make a seperate list of 'valid' reviews
split_frac = 0.8
len_feat = len(final_encoded_training_data)
train_x = final_encoded_training_data[0:int(split_frac*len_feat)]
train_y = filtered_train_labels[0:int(split_frac*len_feat)]
valid_x = final_encoded_training_data[int(split_frac*len_feat):]
valid_y = filtered_train_labels[int(split_frac*len_feat):]


# In[68]:


#Step 12 Dataloaders and Batching, adapted from article
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
# create Tensor datasets
#torch.from_numpy requires an array, so np.array used to transform list of labels into an array of labels
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(np.array(train_y)))
test_data = TensorDataset(torch.from_numpy(final_encoded_testing_data), torch.from_numpy(np.array(filtered_test_labels)))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(np.array(valid_y)))
# dataloaders
batch_size = 50
# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


# In[69]:


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()


# In[70]:


#Step 14: Define LSTM class
import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        #if (train_on_gpu):
         #   hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
          #        weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        #else:
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())


# In[71]:


def count_vocab(an_array):
    return sum(len(row) for row in an_array)


# In[72]:


# Instantiate the network
# Instantiate the model w/ hyperparams
vocab_size = count_vocab(train_x)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)


# In[73]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
#if(train_on_gpu):
    #net.cuda()

    
net.train()
# train for some number of epochs
for e in range(epochs):
    #how many times has the graph been backtraced?
    #temp = 0    
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        #if(train_on_gpu):
         #   inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        #h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                #if(train_on_gpu):
                    #inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# In[74]:


# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
#if(train_on_gpu):
    #net.cuda()

    
net.train()
# train for some number of epochs
for e in range(epochs):
    #how many times has the graph been backtraced?
    temp = 0    
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        #if(train_on_gpu):
         #   inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        #h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        if temp == 1:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
            temp = 1
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                #if(train_on_gpu):
                    #inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

