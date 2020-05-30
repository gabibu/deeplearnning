import numpy as np
from .losses import *

class LinearClassifier(object):

    def __init__(self, X, y):
        """
        Class constructor. Use this method to initiate the parameters of
        your model (W)
        *** Subclasses will override this. ***

        Inputs:
        - X: array of data
        - y: 1-dimensional array of length N with binary labels

        This function has no return value

        """
        pass        

    def predict(self, X):

        """
        Use the weight of the classifier to predict a label. 
        *** Subclasses will override this. ***

        Input: 2D array of size (num_instances, num_features).
        Output: 1D array of class predictions (num_instances, 1). 
        """
        pass

    def calc_accuracy(self, X, y):

        """
        Calculate the accuracy on a dataset as the percentage of instances 
        that are classified correctly. 

        Inputs:
        - W: array of weights
        - X: array of data
        - y: 1-dimensional array of length N with binary labels
        Returns:
        - accuracy as a single float
        """
        accuracy = 0.0
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        accuracy = np.sum(self.predict(X) == y)/y.shape
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return accuracy

    def train(self, X, y, learning_rate=1e-3, num_iters=100, batch_size=200, verbose=False):
        #########################################################################
        # TODO:                                                                 #
        # Sample batch_size elements from the training data and their           #
        # corresponding labels to use in every iteration.                       #
        # Store the data in X_batch and their corresponding labels in           #
        # y_batch                                                               #
        #                                                                       #
        # Hint: Use np.random.choice to generate indices. Sampling with         #
        # replacement is faster than sampling without replacement.              #
        #                                                                       #
        # Next, calculate the loss and gradient and update the weights using    #
        # the learning rate. Use the loss_history array to save the loss on     #
        # iteration to visualize the loss.                                      #
        #########################################################################

        loss_history = []
        loss = 0.0
        for i in range(num_iters):
            ###########################################################################
            #                          START OF YOUR CODE                             #
            ###########################################################################
            batch_indexes = np.random.choice(X.shape[0], batch_size,replace = True)
            X_batch = X[batch_indexes]
            y_batch = y[batch_indexes]
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            self.W += - learning_rate * grad

            ###########################################################################
            #                           END OF YOUR CODE                              #
            ###########################################################################

            if verbose and i % 100 == 0:
                print ('iteration %d / %d: loss %f' % (i, num_iters, loss))

        return loss_history

    def loss(self, X, y):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass

class LinearPerceptron(LinearClassifier):
    # Classifier that uses Perceptron loss

    def __init__(self, X, y):

        ###########################################################################
        # TODO:                                                                   #
        # Initiate the parameters of your model.                                  #
        ###########################################################################

        #self.index_to_class = {index: class_val for index, class_val in  enumerate(np.unique(y))}

        self.W = np.random.rand(X.shape[1], len(np.unique(y))) * 0.0001
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

    def predict(self, X):
        y_pred = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        y_pred =  np.argmax(np.dot(X, self.W), axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch):
        return perceptron_loss_vectorized(self.W, X_batch, y_batch)


class LinearPerceptronRegularization(LinearPerceptron):
    # Classifier that uses Perceptron loss

    def __init__(self, X, y, regularization_rate):

        ###########################################################################
        # TODO:                                                                   #
        # Initiate the parameters of your model.                                  #
        ###########################################################################

        #self.index_to_class = {index: class_val for index, class_val in  enumerate(np.unique(y))}

        LinearPerceptron.__init__(self, X, y)
        self.regularization_rate = regularization_rate
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

    def loss(self, X_batch, y_batch):
        return perceptron_loss_vectorized_with_regularization(self.W, X_batch, y_batch,
                                                              self.regularization_rate)


class LogisticRegression(LinearClassifier):
    # Classifer that uses sigmoid and binary cross entropy loss

    def __init__(self, X, y):
        self.W = None
        ###########################################################################
        # TODO:                                                                   #
        # Initiate the parameters of your model.                                  #
        ###########################################################################
        self.W  = np.random.randn(X.shape[1], 1) * 0.0001
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################




    def predict(self, X):
        y_pred = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method.                                                  #
        ###########################################################################
        scores = X.dot(self.W)
        probs = sigmoid(scores)

        y_pred = np.where(probs > 0.5, 1 , 0).flatten()
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch):
        return binary_cross_entropy(self.W, X_batch, y_batch)
