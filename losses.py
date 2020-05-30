import numpy as np
from random import randrange

def hinge_loss(t, y):
    return np.max(0, 1- t * y)

def perceptron_loss_naive(W, X, y):
  """
  Structured perceptron loss function, naive implementation (with loops)
  Inputs:
  - W: array of weights
  - X: array of data
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Compute the perceptron loss as learned in class. Start by iterating over  #
  # over all instances and calculate the score and true score for each.       #
  # Now, for each class determine if the prediction is correct and update the #
  # loss over all mistakes.                                                   #
  # Compute the gradient of the loss function and store it as dW.             #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed.                                                   #
  #############################################################################

  for index in range(X.shape[0]):
      scores = X[index].dot(W)
      label_score = scores[y[index]]

      for cls in range(W.shape[1]):
          if cls == y[index]:
              continue

          current_score  = scores[cls]
          margin = 1 - (label_score - current_score)
          loss += max(0, margin)

          if margin > 0:
              dW[:, cls] = dW[:, cls] + X[index]
              dW[:, y[index]] =  dW[:, y[index]] - X[index]

  loss /= X.shape[0]
  dW /= X.shape[0]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def perceptron_loss_vectorized(W, X, y):
  """
  Vectorized version of perceptron_loss_naive. instead of loops, should use 
  numpy vectorization.

  Inputs and outputs are the same as perceptron_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the perceptron loss, storing the       #
  # result in loss and the gradient in dW                                     #
  #############################################################################
  scores = X.dot(W)
  correct_score = scores[np.arange(len(scores)), y]
  correct_score = correct_score.reshape(X.shape[0], -1)
  scores += 1 - correct_score
  scores[np.arange(len(scores)), y] = 0
  scores = np.fmax(scores, 0)
  loss = np.sum(np.fmax(scores, 0))

  mask = np.zeros(scores.shape)
  mask[scores > 0] = 1
  mask[np.arange(X.shape[0]), y] = -np.sum(mask, axis=1)
  dW = X.T.dot(mask)

  loss /= X.shape[0]
  dW /= X.shape[0]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def sigmoid(x):
  """
  Numerically stable Sigmoid function.

  Input: any unnormalized log probabilities vector
  Output: normalized probabilities
  """
  #############################################################################
  # TODO:                                                                     #
  # Implement the function                                                    #
  #############################################################################
  x = 1 / (1 + np.exp(-x))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return x

def binary_cross_entropy(W, X, y):
  """
  Structured BCE loss function. Implement this function using vectorized code.
  Inputs:
  - W: array of weights
  - X: array of data
  - y: 1-dimensional array of length N with binary labels (0,1). 
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement the function and store result in loss and the gradint in dW     #
  # Note: in class you defined BCE that takes values from the range (-1,1).   #
  # and the sigmoid function generally outputs values in the range (0,1).     #
  # Make the proper adjustments for your code to work.                        #
  #############################################################################
  # z = W.T.dot(X)
  # yz = y * z

  #todo:
  # y[y == 0] = -1
  #todo:
  y[y==0] = -1
  w1 = W.flatten()
  z = np.dot(X, w1)
  tz = z * y
  loss = np.sum(np.log(1 + np.exp(-tz)))

  s1 = sigmoid(-y * z)
  s2 = (s1 * y)
  s3 = s2.reshape(X.shape[0], 1) * X
  dW =  - np.sum(s3, axis=0).reshape(W.shape[0], 1)




  # #todo: my
  #
  # import math
  # n_samples, n_features = X.shape
  # classes_count = 1 #W.shape[1]
  #
  # def _inner_log_logistic_sigmoid(x):
  #   """Log of the logistic sigmoid function log(1 / (1 + e ** -x))"""
  #   if x > 0:
  #       return -math.log(1. + math.exp(-x))
  #   else:
  #       return x - math.log(1. + math.exp(x))
  #
  # def _log_logistic_sigmoid(X, out):
  #     n_samples, n_features = X.shape
  #     for i in range(n_samples):
  #         for j in range(n_features):
  #             out[i, j] = _inner_log_logistic_sigmoid(X[i, j])
  #     return out
  #
  # def log_logistic(X, out=None):
  #     is_1d = X.ndim == 1
  #     X = np.atleast_2d(X)
  #
  #     if out is None:
  #         out = np.empty_like(X)
  #
  #     _log_logistic_sigmoid(X, out)
  #
  #     if is_1d:
  #         return np.squeeze(out)
  #     return out
  #
  # w = W.flatten()
  # z = np.dot(X, w)
  # yz = y * z
  # # Loss
  # loss = -np.sum(log_logistic(yz))
  # # Gradient
  # z = sigmoid(yz)
  # z0 = (z - 1) * y
  # grad = np.empty_like(w)
  # grad[:n_features] = X.T.dot(z0)
  # dW = grad.reshape(n_features, classes_count)
  # print(dW.shape)
  # # todo:

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW

def grad_check(f, x, analytic_grad, num_checks=10, h=1e-5):

  for i in range(num_checks):
      ix = tuple([randrange(m) for m in x.shape])
      oldval = x[ix]
      x[ix] = oldval + h # increment by h
      fxph = f(x) # evaluate f(x + h)
      x[ix] = oldval - h # increment by h
      fxmh = f(x) # evaluate f(x - h)
      x[ix] = oldval # reset
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = analytic_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print ('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))


def perceptron_loss_vectorized_with_regularization(W, X, y, regularization_rate):

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W)
  correct_score = scores[np.arange(len(scores)), y]
  correct_score = correct_score.reshape(X.shape[0], -1)
  scores += 1 - correct_score
  scores[np.arange(len(scores)), y] = 0
  scores = np.fmax(scores, 0)
  loss = np.sum(np.fmax(scores, 0)) + regularization_rate * np.sum(np.power(W, 2))

  mask = np.zeros(scores.shape)
  mask[scores > 0] = 1
  mask[np.arange(X.shape[0]), y] = -np.sum(mask, axis=1)
  dW = X.T.dot(mask)

  loss /= X.shape[0]
  dW /= X.shape[0]

  return loss, dW
