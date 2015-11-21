function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    errorVector = (X*theta)-y;                      % errorVector is mx1 vector.
    theta1 = (alpha/m) * sum(errorVector.*X(:,1));  % X(:,1) represents X0 and .* does an element-by-element multiplication
    theta2 = (alpha/m) * sum(errorVector.*X(:,2));  % X(:,2) represents X1 and .* does an element-by-element multiplication
    theta(1) = theta(1) - theta1;                   % Update theta1 and theta2 simultaneously
    theta(2) = theta(2) - theta2;    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
