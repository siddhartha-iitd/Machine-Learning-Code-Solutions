function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesisFuncn = sigmoid(X*theta);
thetaForRegularization = theta(2:end).^2;

J = 1/m * (sum((-y .* log(hypothesisFuncn)) - ((1-y) .* log(1-hypothesisFuncn)))) + (lambda/(2*m))*(sum(thetaForRegularization));

for i=1:size(grad)
    grad(i) = (hypothesisFuncn-y)' * X(:,i);
    if i>1
        grad(i) = grad(i) + lambda*theta(i);
    end
end

grad = 1/m * grad;

% =============================================================

end
