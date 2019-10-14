function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

possibleC = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
possibleSigma = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Setting the min allowed prediction error to be very high
minPredictionError = 9999999999;

for i = 1:size(possibleC)
  for j = 1:size(possibleSigma)
    fprintf('Training with c:%f, sigma:%f \n', possibleC(i), possibleSigma(j));
    model = svmTrain(X, y, possibleC(i), @(x1, x2) gaussianKernel(x1, x2, possibleSigma(j)));
    
    fprintf('Predicting...\n');
    predictions = svmPredict(model, Xval);
    
    predictionError = mean(double(predictions ~= yval));
    fprintf('Prediction error: %f\n', predictionError);
    
    if predictionError < minPredictionError
      fprintf('Found a better model with lesser prediction error.. Updating the values of C and sigma to %f and %f\n', possibleC(i), possibleSigma(j));
      C = possibleC(i);
      sigma = possibleSigma(j);
      
      minPredictionError = predictionError;
    endif
    
    
  endfor
endfor





% =========================================================================

end
