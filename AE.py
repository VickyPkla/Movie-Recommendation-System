import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
from torch.autograd import Variable

movies = pd.read_csv(r'ml-1m\movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv(r'ml-1m\users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv(r'ml-1m\ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

#Preparing the data
training_set = pd.read_csv(r'ml-100k\u1.base', delimiter='\t') #The data is separated by a tab. delimiter is the similar to that of sep.
training_set = np.array(training_set, dtype='int') #Converting dataframe to array
test_set = pd.read_csv(r'ml-100k\u1.test', delimiter='\t') #The data is separated by a tab. delimiter is the similar to that of sep.
test_set = np.array(test_set, dtype='int') #Converting dataframe to array

#Getting the number of users and movies
#The users are spread across the trainging and testing file. Hence we are using both to get the last ID of user to find the total users. Same with movies
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))  
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))


#Converting out data into 2D array such that the rows = users and columns = movies id
# arr[i][j] = rating given by ith user to the jth movie
def convert(data) :
    new_data = [] 
    for id_user in range(1,nb_users+1):
        id_movies = data[:,1][data[:,0] == id_user] #The second bracket is the condition used to make sure that we get only those movies which are rated by id_user
        id_ratings = data[:,2][data[:,0] == id_user] #The second bracket is the condition used to make sure that we get only those movies which are rated by id_user
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)

# Converting our data into a torch tensor as it is more efficient to worj with than numpy
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Developing the Autoencoder

class SAE(nn.Module):  # nn.Module is the parent class for our SAE.
    def __init__(self, ): # self, because after the , the variables of the parent class is used automatically
        super(SAE, self).__init__() # Super function is used to get access to the data members and member functions of the parent class
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x) # Since this is the last layer, we don't use the activation function
        return x
    
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5) # Decay is used to reduce the learning rate after every few epochs to ensure convergence

# Training our Stacked Autoencoder
nb_epochs = 200
for epoch in range(1,nb_epochs + 1):
    train_loss = 0
    # s = 0. # s is the number of users that rated atleast one movie
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0) # Pytorch doesn't work with 1D vectors. Therefore we are adding a new dimension at position 0 (Batch), just like in the case of CNN
        target = input.clone()
        #optimize the memory
        if torch.sum(target.data > 0) > 0: #Target.data returns a boolean of same size as that of target where each element is eith True(1) or False(0). Therefore, its sum gives us the numebr of users with atleast one rating givrn
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0 # Making the output of the movies 0 that are not rated
            loss = criterion(output, target)
            # mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # Check again
            loss.backward()
            train_loss = np.sqrt(loss.item())
            # s+=1
            optimizer.step()
    print(f"Epoch {epoch} - Loss = {train_loss}")

# Testing
# In the test set we have the ratings of movies that were still unrated in the training set
# We are using the train set to predict the ratings of these movies and compare with the test set
test_loss = 0
# s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0) # We use training set because it takes the ratings of the movies that the user has watched and based on that it will predict the ratings of the movies that the user hasn't watched
    #optimize the memory
    if torch.sum(target.data > 0) > 0: #Target.data returns a boolean of same size as that of target where each element is eith True(1) or False(0). Therefore, its sum gives us the numebr of users with atleast one rating givrn
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0 # Making the output of the movies 0 that are not rated
        loss = criterion(output, target)
        # mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10) # Check again
        test_loss = np.sqrt(loss.item())
        # s+=1
print(f"Test loss = {test_loss}")