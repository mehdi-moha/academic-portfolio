clc;
clear;
close all;
%% Generate nonlinear dataset
N = 100;
[classA, classB] = generateRings(N);

figure;
scatter(classA(1, :), classA(2, :), 120, 'g', 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(classB(1, :), classB(2, :), 120, 'r', 'filled', 'MarkerEdgeColor', 'k');
axis equal; grid on;
title('Nonlinear Dataset');
%% Split data into train and test
ratio = 0.7;
nA = round(size(classA, 2)*ratio);

trainA = classA(:, 1:nA);
testA = classA(:, nA+1:end);

trainB = classB(:, 1:nA);
testB = classB(:, nA+1:end);

Xtrain = [trainA, trainB];
Ytrain = [ones(1, size(trainA, 2)), -ones(1, size(trainB, 2))];

Xtest = [testA, testB];
Ytest = [ones(1, size(testA, 2)), -ones(1, size(testB, 2))];
%% AdaBoost settings
T = 230;
weights = ones(1, size(Xtrain, 2)) / size(Xtrain, 2);
models = cell(1, T);
alpha  = zeros(1,T);
%% Boosting loop
for t = 1:T

    % Weighted sampling
    cumW = cumsum(weights);
    r = rand(1, size(Xtrain, 2));
    idx = arrayfun(@(x) find(cumW >= x, 1), r);

    Xb = Xtrain(:, idx);
    Yb = Ytrain(idx);

    % Train weak learner (SLP)
    models{t} = trainSLP(Xb, Yb);

    % Test on training set
    pred = testSLP(models{t}, Xtrain);
    err = (pred ~= Ytrain);

    eps_t = sum(weights.*err);
    eps_t = max(min(eps_t, 1-1e-10), 1e-10); % avoid division issues

    % Compute alpha
    alpha(t) = 0.5 * log((1 - eps_t)/eps_t);

    % Update weights
    weights = weights .* exp(alpha(t)*err);
    weights = weights / sum(weights);
end
%% Final prediction
scores = zeros(T, size(Xtest, 2));

for t = 1:T
    scores(t, :) = alpha(t) * testSLP(models{t}, Xtest);
end

final = sign(sum(scores, 1));
acc = mean(final == Ytest) * 100;
%% Plot results
figure;
scatter(Xtest(1, final == 1), Xtest(2, final == 1), 120, 'g', 'filled', 'MarkerEdgeColor', 'k');
hold on;
scatter(Xtest(1, final == -1), Xtest(2, final == -1), 120, 'r', 'filled', 'MarkerEdgeColor', 'k');
axis equal; grid on;
title(['Accuracy = ', num2str(acc), ' %']);
