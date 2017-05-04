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

Citer = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaiter = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
x1 = [1 2 1]; x2 = [0 4 -1];

predmatrix =  zeros((length(Citer) * length(sigmaiter)), 3);
predcnt =0;
for i=1:length(Citer)
Ctrain = Citer(i);
for j=1:length(sigmaiter)
predcnt = predcnt+1;
sigmatain = sigmaiter(j);
fprintf('\ntraining for C %f \t\t and sigma \t\t %f',Ctrain, sigmatain);

model= svmTrain(X, y, Ctrain, @(x1, x2) gaussianKernel(x1, x2, sigmatain));
pred = svmPredict(model, Xval);
prederror = mean(double(pred ~= yval));
predmatrix(predcnt,:) = [Ctrain,sigmatain,prederror];
fprintf('\npredictions for %f\t',prederror);
endfor
endfor
[minpred,index] = min(predmatrix(:,3));
fprintf('\nMinimum prediction is %f\t\t',minpred);
fprintf('\ncorresponding C and sigma values are %f\t%f\t%f',predmatrix(index,:));

C = predmatrix(index,1);
sigma = predmatrix(index,2);

% =========================================================================

end
