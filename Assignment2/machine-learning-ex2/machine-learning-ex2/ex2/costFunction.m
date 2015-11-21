function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%   normX = zeros(size(X));
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% Calculate mean for each feature in X_norm
% mu = mean(X);
% % Calculate standard deviation for each feature in X_norm
% sigma = std(X);

% Update each feature in X_norm as (feature - mean)/StandardDeviation
% for i=1:size(X,2)
%     normX(:,1) = (X(:,i) - mu(i))/(sigma(i));
% end

hypothesisFuncn = sigmoid(X*theta);

J = -1/m * sum((y .* log(hypothesisFuncn)) + ((1-y) .* log(1-hypothesisFuncn)));

for i=1:size(grad)
    grad(i) = (hypothesisFuncn-y)' * X(:,i);
end

grad = 1/m * grad;

% =============================================================

end
