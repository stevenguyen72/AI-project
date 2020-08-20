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

% create y binary display
temp_y = diag(ones(1,num_labels));
y_binary = zeros(length(y),num_labels);
for i = 1:length(y)
    y_binary(i,:) = temp_y(:,y(i));
%     if y(i) == 10
%         y_binary(i,:) = temp_y(:,1);
%     else
%         y_binary(i,:) = temp_y(:,y(i)+1);  
%     end
end


% calculate J(theta)
X = [ones(m,1),X];
temp_hox = X * Theta1';
a2 = [ones(m,1),sigmoid(temp_hox)];
% temp_hox2 = [ones(m,1),a2];
hox_a3 = sigmoid(a2 * Theta2');

temp_J = (-y_binary .* log(hox_a3)) - ((1 - y_binary) .* log(1-hox_a3)); 
J = (1/m) * (sum(sum(temp_J)))+lambda/(2*m)*...
(sum(sum((Theta1(:,2:end).^2)))+sum(sum((Theta2(:,2:end).^2))));

% % setup backpropagation algorithm using for loop
% delta_1 = 0;
% delta_2 = 0;
% for i = 1:m
%     a_1 = X(i,:)';
%     a_2 = [1;sigmoid(Theta1 * a_1)];
%     a_3 = sigmoid(Theta2 * a_2);
%     epsilon_3 = a_3 - y_binary(i,:)';
%     epsilon_2 = (Theta2' * epsilon_3) .* (a_2 .*(1 - a_2));
%     epsilon_2 = epsilon_2(2:end);
%     delta_2 = delta_2 + epsilon_3 * a_2';
%     delta_1 = delta_1 + epsilon_2 * a_1';
% end
% Theta1_grad = 1/m * delta_1;
% Theta2_grad = 1/m * delta_2;
    
% setup backpropagation algorithm using vectorize
% setup epsilon
epsilon_3 = hox_a3 - y_binary;
epsilon_2 = (epsilon_3 * Theta2) .* (a2 .* (1-a2));
epsilon_2 = epsilon_2(:,2:end);

% setup delta
delta_2 = epsilon_3' * a2;
delta_1 = epsilon_2' * X;

% setup D
Theta2_grad = 1/m * delta_2;
Theta1_grad = 1/m * delta_1;

% setup regulazation for grad
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2(:,2:end);
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1(:,2:end);



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




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
