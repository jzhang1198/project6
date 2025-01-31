"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

import pytest
from regression import logreg, utils

import numpy as np

def test_updates():
	"""
	Unit test for logistic regressor to ensure that losses and gradients are computed properly.
	"""

	X_train,X_test,y_train,y_test = utils.loadDataset(split_percent=.8)

	num_feats = X_train.shape[1] #define hyperparameters
	learning_rate = 0.001
	tol = 0.000001
	max_iter = 10000
	batch_size = 400
	regressor = logreg.LogisticRegression(num_feats, learning_rate, tol, max_iter, batch_size) #instantiate a logistic regressor

	old_loss = regressor.loss_function(np.hstack([X_train, np.ones((X_train.shape[0], 1))]), y_train)
	assert old_loss > 0 and old_loss < 100 #check that losses are reasonable and computed properly

	grad = regressor.calculate_gradient(np.hstack([X_train, np.ones((X_train.shape[0], 1))]), y_train)
	regressor.W = regressor.W - regressor.lr *  grad
	new_loss = regressor.loss_function(np.hstack([X_train, np.ones((X_train.shape[0], 1))]), y_train)
	assert new_loss < old_loss #check that the gradient is computed properly

	regressor.train_model(X_train, y_train, X_test, y_test)
	assert regressor.loss_history_val[-1] != 0 and regressor.loss_history_train[-1] != 0 #check that losses are not unreasonably low


def test_predict():
	"""
	Unit test for logistic regressor to ensure that weights
	"""

	X_train,X_test,y_train,y_test = utils.loadDataset(split_percent=.8)

	num_feats = X_train.shape[1] #define hyperparameters
	learning_rate = 0.001
	tol = 0.000001
	max_iter = 2
	batch_size = 2
	regressor = logreg.LogisticRegression(num_feats, learning_rate, tol, max_iter, batch_size) #instantiate a logistic regressor

	old_w = regressor.W
	old_grad = regressor.calculate_gradient(np.hstack([X_train, np.ones((X_train.shape[0], 1))]),y_train)
	regressor.train_model(X_train, y_train, X_test, y_test)
	new_w = regressor.W
	assert set(old_w - new_w) != {0} #check that self.W is being updated

	batch_size = 400
	max_iter = 10000
	regressor = logreg.LogisticRegression(num_feats, learning_rate, tol, max_iter, batch_size) #instantiate a logistic regressor
	regressor.train_model(X_train, y_train, X_test, y_test) #train model

	X,y = utils.loadDataset()
	X = np.hstack([X,np.ones((X.shape[0],1))])
	y_preds = np.around(regressor.make_prediction(X),decimals=0)
	assert set(y_preds) == {0,1} #check that predicted labels are 0 or 1

	accuracy = len([i for i in (y_preds - y) if i == 0]) / len(y)
	assert accuracy > 0.7 #check that accuracy is reasonable
