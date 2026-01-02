function out = testSLP(mdl, X)
% Apply SLP model
X = [-ones(1, size(X, 2)); X];
out = sign(mdl.w*X);
end