function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

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
%x1 = [1 2 1];
%x2 = [0 4 -1];
errorVec = zeros(128,3);
count=1;
while(C <= 2)
  sigma = 0.001;
  while(sigma <= 2)
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    errorVec(count,1)= mean(double(predictions ~= yval));
    errorVec(count,2)= C;
    errorVec(count,3)=sigma;
    count = count + 1;
    sigma = sigma * 3.0;
  end;
  C = C * 3.0;
end;
[val, index] = min(errorVec(1:count-1,1));
C = errorVec(index,2);
sigma = errorVec(index,3);
% =========================================================================

end
