function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
youtputconv = eye(num_labels);

accumulatedDelta1=0;
accumulatedDelta2=0;







for i=1:m
  A1 = X(i,:);
  Z2 = A1 * Theta1';
  A2 = sigmoid(Z2);

  A2 = [ones(1, 1) A2];
  Z3 = A2 * Theta2';
  A3 = sigmoid(Z3);
  outputyVec = youtputconv(y(i,1),:);
  outputFunc =  (-outputyVec .* log(A3) - ((1-outputyVec) .* log(1 - A3)));
  J =  J + (sum(outputFunc));

  %Back propagation begin
  delta3 = A3 - outputyVec;
  %disp(size(delta3));
  %disp(size(Z2));
  %Theta2 - 10*26, delta_out - 1 *10
  accumulatedDelta2 = accumulatedDelta2 + A2' * delta3;
  delta2 =  (delta3 * Theta2(:,2:end))  .* sigmoidGradient(Z2);
  accumulatedDelta1 =  accumulatedDelta1 + A1' * delta2;
  %disp(size(accumulatedDelta2));
  %disp(size(accumulatedDelta1));

endfor
regTheta1 = Theta1(:,2:(input_layer_size+1));
regTheta2 = Theta2(:,2:(hidden_layer_size+1));
regTheta1 = regTheta1 .^ 2 ;
regTheta2 = regTheta2 .^ 2 ;
regularizedSum = sum(sum(regTheta1,2),1) + sum(sum(regTheta2,2),1);
regularizedCost = ((lambda/(2*m)) * regularizedSum);

Theta1_grad = ( (1/m) .* accumulatedDelta1' );
Theta2_grad = ( (1/m) .* accumulatedDelta2' );

%regularized gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) .+ ( (lambda/m) .* Theta1(:,2:end));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) .+ ( (lambda/m) .* Theta2(:,2:end));

J = ( (J/m) + regularizedCost);



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
