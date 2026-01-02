function mdl = trainSLP(X, Y)
% Simple least-squares SLP training
X = [-ones(1, size(X, 2)); X];
R = X * X';
P = Y * X';
mdl.w = P / R;
end