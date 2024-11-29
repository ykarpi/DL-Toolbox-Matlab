% reading the data from the .csv file
housing = readtable('BostonHousing.csv');

% Check for missing values in the dataset
missing_values = any(ismissing(housing), 1);

% Display the variables with missing values
variables_with_missing = housing.Properties.VariableNames(missing_values);

% Display the number of missing values for each variable
num_missing = sum(ismissing(housing(:, variables_with_missing)));
disp('Variables with missing values:');
disp(variables_with_missing);
disp('Number of missing values for each variable:');
disp(num_missing)


rm_column = housing.rm; % Extract the 'rm' column (which has missing values)
rm_imputed = fillmissing(rm_column, 'knn'); % Assign missing values in the 'rm' column
housing.rm = rm_imputed; % Replace the original 'rm' column with the updated values


% RUN THE CHECK FOR MISSING VALUES AGAIN
missing_values = any(ismissing(housing), 1);
vars_with_missing = housing.Properties.VariableNames(missing_values);
num_missing = sum(ismissing(housing(:, vars_with_missing)));
disp('Variables with missing values:');

disp(vars_with_missing);
disp('Number of missing values for each variable:');

disp(num_missing)

% Compute the correlation matrix
correlation_matrix = corr(housing{:,:});

figure;
imagesc(correlation_matrix);
colorbar;
colormap('turbo'); 
xticks(1:size(correlation_matrix, 1));
yticks(1:size(correlation_matrix, 1)); 
xticklabels(housing.Properties.VariableNames); 
yticklabels(housing.Properties.VariableNames);  
title('Correlation Matrix of Housing Dataset');
xlabel('Features');
ylabel('Features');


numerical_variables = housing{:, :};
num_numerical_variables = size(housing{:,:}, 2); % in case of Boston dataset - 14 variables


num_bins = 10;
% Creating histograms for each variable
figure;
for i = 1:num_numerical_variables
    subplot(ceil(sqrt(num_numerical_variables)), ceil(sqrt(num_numerical_variables)), i);
    histogram(numerical_variables(:, i), num_bins);
    title(housing.Properties.VariableNames{i});
    xlabel('Value');
    ylabel('Frequency');
end


% Features
inputNames = {'crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat'};
% Target variables
outputNames = {'medv'};

housingAttributes = [inputNames,outputNames];

X = housing{:,inputNames};
Y = housing{:,outputNames};

% Option 1
X_norm_1 = normalize(X);
X = X_norm_1;
% Option 2 - rescales in range [0, 1] 
X_norm_2 = normalize(X, "range");
%X = X_norm_2;


% Define the ratio for splitting the data
trainRatio = 0.7;  % 70% of data for training
valRatio = 0.15;    % 15% of data for validation
testRatio = 0.15;   % 15% of data for testing


% Calculate the number of samples for each set
numSamples = size(X, 1);

numTrain = floor(trainRatio * numSamples);
numVal = floor(valRatio * numSamples);
numTest = numSamples - numTrain - numVal;


% add a seed to minize randomness
rng(10)

% Create a random partition for the data
c = cvpartition(numSamples, 'HoldOut', 1 - trainRatio);
trainIdx = training(c);  % Indices for training set
tempIdx = find(~trainIdx);  % Indices for validation and testing set
c = cvpartition(length(tempIdx), 'HoldOut', valRatio / (valRatio + testRatio));
valIdx = tempIdx(training(c));  % Indices for validation set
testIdx = tempIdx(test(c));  % Indices for testing set

% Split the data based on the indices
X_train = X(trainIdx, :);
Y_train = Y(trainIdx);
X_val = X(valIdx, :);
Y_val = Y(valIdx);
X_test = X(testIdx, :);
Y_test = Y(testIdx);


% Display sizes of the sub-sets
disp(['Training set size: ', 'X_train = ', num2str(size(X_train)), ', ', 'Y_train = ', num2str(size(Y_train))]);
disp(['Validation set size: ', 'X_val = ', num2str(size(X_val)), ', ', 'Y_val = ', num2str(size(Y_val))]);
disp(['Testing set size: ', 'X_test = ', num2str(size(X_test)), ', ', 'Y_test = ', num2str(size(Y_test))]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REGRESSION TREES
rng(42);

tree_model = fitrtree(X_train, Y_train); % fit the model
view(tree_model,Mode="graph"); % view the decision process as a graph

% Predicting on the validation set
Y_pred_reg_trees = predict(tree_model, X_val);

% Calculate the mean squared error (MSE)
MSE = mean((Y_val - Y_pred_reg_trees).^2);
disp(['Mean Squared Error (MSE) on Validation Set: ', num2str(MSE)])

% Add pruning
rng(42);

% Pruning
alpha = 0.5; % must be between 0 and 1

tree_model_pruned = prune(tree_model, Alpha=alpha);
view(tree_model_pruned,Mode="graph");

% Predicting on the validation set
Y_pred_reg_trees_pruned = predict(tree_model_pruned, X_val);

% Calculate the mean squared error (MSE)
MSE = mean((Y_val - Y_pred_reg_trees_pruned).^2);
disp(['Mean Squared Error (MSE) on Validation Set: ', num2str(MSE)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RANDOM FOREST REGRESSION
rng(42);

% Define the parameters for the random forest
numTrees = 130;  % Number of trees in the forest

% Training the random forest regression model
rf_model = TreeBagger(numTrees, X_train, Y_train, Method="regression");

% Predict on validation set
Y_val_pred_rf = predict(rf_model, X_val);

% Calculate the mean squared error (MSE) on validation set
MSE_val_rf = mean((Y_val - Y_val_pred_rf).^2);

% Display the mean squared error on validation set
disp(['Mean Squared Error (MSE) on Validation Set (Random Forest): ', num2str(MSE_val_rf)]);

% Plot actual vs. predicted values
figure;
scatter(Y_val, Y_val_pred_rf, 'filled');
hold on;
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--'); % Plotting the line y = x for reference
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values (Validation Set) - Random Forest');
legend('Predicted vs. Actual', 'y = x', 'Location', 'northwest');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LSBOOST
rng(42);

iterations = 500;
lr = 0.01;

lsboost_1 = fitensemble(X_train,Y_train,'LSBoost', iterations, 'Tree', 'LearnRate', lr);
L = loss(lsboost_1,X_val, Y_val,'mode','ensemble');
fprintf('Mean-square validation error = %f\n',L);

%adding other parameters to improve performance:
t = RegressionTree.template('MinLeaf',5);
lsboost_2 = fitensemble(X_train,Y_train,'LSBoost', iterations, t, 'LearnRate', lr);
L = loss(lsboost_2,X_val, Y_val,'mode','ensemble');
fprintf('Mean-square validation error = %f\n',L);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LINEAR REGRESSION

rng(42);

% Perform linear regression
linear_regression_model = fitlm(X_train, Y_train);

% Predict on validation set
Y_val_pred_linear_regression = predict(linear_regression_model, X_val);

% Calculate the mean squared error (MSE) on validation set
MSE_val_linear_regression = mean((Y_val - Y_val_pred_linear_regression).^2);

% Display the mean squared error on validation set
fprintf('Mean-square validation error = %f\n',MSE_val_linear_regression);

% Plot actual vs. predicted values
figure;
scatter(Y_val, Y_val_pred_linear_regression, 'filled');
hold on;
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--'); % Plotting the line y = x for reference
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values (Validation Set) - Linear Regression');
legend('Predicted vs. Actual', 'y = x', 'Location', 'northwest');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUPPORT VECTOR REGRESSION
% Train the SVM regression model
%Linear
svm_model_linear = fitrsvm(X_train, Y_train, 'KernelFunction', 'linear');

% Make predictions on the test set
Y_pred_svm_linear = predict(svm_model_linear, X_val);

% Calculate the mean squared error (MSE)
MSE = mean((Y_val - Y_pred_svm_linear).^2);
disp(['Mean Squared Error (MSE) on Validation Set: ', num2str(MSE)]);

% Plot actual vs. predicted values
figure;
scatter(Y_val, Y_pred_svm_linear, 'filled');
hold on;
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--');
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values (Validation Set) - SVM linear');
legend('Predicted vs. Actual', 'y = x', 'Location', 'northwest');
grid on;

%Gaussin
svm_model_rbf = fitrsvm(X_train, Y_train, 'KernelFunction', 'gaussian');

% Make predictions on the test set
Y_pred_svm_rbf = predict(svm_model_rbf, X_val);

% Calculate the mean squared error (MSE)
MSE = mean((Y_val - Y_pred_svm_rbf).^2);
disp(['Mean Squared Error (MSE) on Validation Set: ', num2str(MSE)]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%KNN
rng(42);

% Train the K-NN model
k = 2; % Number of nearest neighbors
knn_model = fitcknn(X_train, Y_train, 'NumNeighbors', k);

% Make predictions on the test set
Y_pred_knn = predict(knn_model, X_val);

% Calculate the mean squared error (MSE)
MSE = mean((Y_val - Y_pred_knn).^2);

disp(['Mean Squared Error (MSE) on Validation Set: ', num2str(MSE)]);

% Plot actual vs. predicted values
figure;
scatter(Y_val, Y_pred_knn, 'filled');
hold on;
plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], 'r--');
hold off;
xlabel('Actual Values');
ylabel('Predicted Values');
title('Actual vs. Predicted Values (Validation Set) - K-NN');
legend('Predicted vs. Actual', 'y = x', 'Location', 'northwest');
grid on;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D CNN

rng(42);

% Prepare the data
X_train_expanded = reshape(X_train, [13 1 355]);
Y_train_expanded = reshape(Y_train, [1 355]);

X_val_expanded = reshape(X_val, [13 1 76]);
Y_val_expanded = reshape(Y_val, [1 76]);

X_test_expanded = reshape(X_test, [13 1 75]);
Y_test_expanded = reshape(Y_test, [1 75]);

% Define the architecture of our 1D CNN model
layers1D_CNN = [
    sequenceInputLayer([13 1])
    convolution1dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')
    convolution1dLayer(3, 32, 'Padding', 'same') 
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2, 'Padding', 'same')
    flattenLayer
    fullyConnectedLayer(50)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer 
];

lgraph1DCNN = layerGraph(layers1D_CNN);
plot(lgraph1DCNN)


% Specifying training options
options_1D_CNN = trainingOptions('adam', ... 
    'MaxEpochs', 300, ... 
    'MiniBatchSize', 64, ... 
    'InitialLearnRate', 0.01, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_val_expanded, Y_val_expanded}, ... 
    'ValidationFrequency', 10, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');


% Train the 1D CNN model
net_1D_CNN = trainNetwork(X_train_expanded, Y_train_expanded, layers1D_CNN, options_1D_CNN);

% Make predictions on the validation data:
Y_val_pred_1D_CNN = predict(net_1D_CNN, X_val_expanded);
Y_val_pred_1D_CNN = double(Y_val_pred_1D_CNN);

mse = immse(Y_val_expanded, Y_val_pred_1D_CNN);

fprintf('Mean Squared Error (MSE) on validation data: %.4f\n', mse);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NEURAL NETWORKS

% Define the architecture of the neural network
features = size(X_train, 2); % 13 in case of Boston dataset

% 2 fully connected layers with relu activation
layers_NN_1 = [
    featureInputLayer(features)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

% 2 FC layers with sigmoid activation
layers_NN_2 = [
    featureInputLayer(features)
    fullyConnectedLayer(50)
    sigmoidLayer
    fullyConnectedLayer(30)
    sigmoidLayer
    fullyConnectedLayer(1)
    regressionLayer
];

% 2 FC layers with relu activation and dropout layers
layers_NN_3 = [
    featureInputLayer(features)
    fullyConnectedLayer(50)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];


% Specify training options

% SGD optimizer
options_NN_sgd_1 = trainingOptions('sgdm', ... 
    'MaxEpochs', 120, ... 
    'MiniBatchSize', 64, ... 
    'InitialLearnRate', 0.00001, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', 10, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');
% One of the big issues for trying to train the model with SGD solver was overcoming the problem of getting NaN loss at the 2nd epoch. 
% Reducing learning rate for SGD helped to avoid this kind of situations. 
% additional information: https://stackoverflow.com/questions/37232782/nan-loss-when-training-regression-network
% It seems like neural networks with SGD are prone to this kind of problems particularly in the regression tasks since the output is unbounded.
% The likely cause of NaN loss is the exploding gradient problem.


% ADAM optimizer
options_NN_adam_1 = trainingOptions('adam', ... 
    'MaxEpochs', 100, ... 
    'MiniBatchSize', 32, ... 
    'InitialLearnRate', 0.01, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', 10, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

options_NN_adam_2 = trainingOptions('adam', ... 
    'MaxEpochs', 1000, ... 
    'MiniBatchSize', 64, ... 
    'InitialLearnRate', 0.001, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', 20, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

options_NN_adam_3 = trainingOptions('adam', ... 
    'MaxEpochs', 200, ... 
    'MiniBatchSize', 32, ... 
    'InitialLearnRate', 0.001, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_val, Y_val}, ... 
    'ValidationFrequency', 10, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

% SDG optimizer:
net_NN_sgd = trainNetwork(X_train, Y_train, layers_NN_1, options_NN_sgd_1);
% Make predictions on the validation data:
Y_val_pred_NN = predict(net_NN_sgd, X_val);
Y_val_pred_NN = double(Y_val_pred_NN);

mse = immse(Y_val, Y_val_pred_NN);

fprintf('Mean Squared Error (MSE) on validation data: %.4f\n', mse);


% Adam optimizer:
% original testing of NN with adam optimizer and relu activation
net_NN_adam_1 = trainNetwork(X_train, Y_train, layers_NN_1, options_NN_adam_1);
% Make predictions on the validation data:
Y_val_pred_NN = predict(net_NN_adam_1, X_val);
Y_val_pred_NN = double(Y_val_pred_NN);

mse = immse(Y_val, Y_val_pred_NN);
fprintf('Mean Squared Error (MSE) on validation data: %.4f\n', mse);


% Adam optimizer with sigmoid activation:
% using sigmoid as activation
net_NN_adam_2 = trainNetwork(X_train, Y_train, layers_NN_2, options_NN_adam_2);
% Make predictions on the validation data:
Y_val_pred_NN = predict(net_NN_adam_2, X_val);
Y_val_pred_NN = double(Y_val_pred_NN);

mse = immse(Y_val, Y_val_pred_NN);
fprintf('Mean Squared Error (MSE) on validation data: %.4f\n', mse);


%Adam optimizer + dropout layers
net_NN_adam_3 = trainNetwork(X_train, Y_train, layers_NN_3, options_NN_adam_3);
% Make predictions on the validation data:
Y_val_pred_NN = predict(net_NN_adam_3, X_val);
Y_val_pred_NN = double(Y_val_pred_NN);

mse = immse(Y_val, Y_val_pred_NN);
fprintf('Mean Squared Error (MSE) on validation data: %.4f\n', mse);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FINAL TESTING ON THE TEST SET USING TRAINED CLASSIFIERS

% Regression trees
Y_test_pred_reg_tree = predict(tree_model, X_test);
Y_test_pred_reg_tree = double(Y_test_pred_reg_tree);
mse_reg_tree = immse(Y_test, Y_test_pred_reg_tree);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_reg_tree);

% Regression trees (pruned)
Y_test_pred_reg_tree_pruned = predict(tree_model_pruned, X_test);
Y_test_pred_reg_tree_pruned = double(Y_test_pred_reg_tree_pruned);
mse_reg_tree_pruned = immse(Y_test, Y_test_pred_reg_tree_pruned);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_reg_tree_pruned);

% Random Forest regression
Y_test_pred_rf = predict(rf_model, X_test);
Y_test_pred_rf = double(Y_test_pred_rf);
mse_rf = immse(Y_test, Y_test_pred_rf);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_rf);

% LSBoost
Y_test_pred_lsboost = predict(lsboost_2, X_test);
Y_test_pred_lsboost = double(Y_test_pred_lsboost);
mse_lsboost = immse(Y_test, Y_test_pred_lsboost);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_lsboost);

% Linear regression
Y_test_pred_linear_reg = predict(linear_regression_model, X_test);
Y_test_pred_linear_reg = double(Y_test_pred_linear_reg);
mse_linear_reg = immse(Y_test, Y_test_pred_linear_reg);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_linear_reg);

% K-NN
Y_test_pred_knn = predict(knn_model, X_test);
Y_test_pred_knn = double(Y_test_pred_knn);
mse_knn = immse(Y_test, Y_test_pred_knn);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_knn);

% Support vector machine (linear) regression
Y_test_pred_svm_linear = predict(svm_model_linear, X_test);
Y_test_pred_svm_linear = double(Y_test_pred_svm_linear);
mse_SVM_linear = immse(Y_test, Y_test_pred_svm_linear);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_SVM_linear);

% Support vector machine (rbf) regression
Y_test_pred_svm_rbf = predict(svm_model_rbf, X_test);
Y_test_pred_svm_rbf = double(Y_test_pred_svm_rbf);
mse_SVM_rbf = immse(Y_test, Y_test_pred_svm_rbf);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_SVM_rbf);

% 1D CNN
Y_test_pred_1D_CNN = predict(net_1D_CNN, X_test_expanded);
Y_test_pred_1D_CNN = double(Y_test_pred_1D_CNN);
mse_1D_CNN = immse(Y_test_expanded, Y_test_pred_1D_CNN);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_1D_CNN);

% Neural Netowkrs (SGD solver)
Y_test_pred_NN_sgd = predict(net_NN_sgd, X_test);
Y_test_pred_NN_sgd = double(Y_test_pred_NN_sgd);
mse_nn_sgd = immse(Y_test, Y_test_pred_NN_sgd);
fprintf('Mean Squared Error (MSE) on test set: %.4f\n', mse_nn_sgd);

% Neural Networks (Adam solver)
Y_test_pred_NN_adam = predict(net_NN_adam_1, X_test);
Y_test_pred_NN_adam = double(Y_test_pred_NN_adam);
mse_nn_adam = immse(Y_test, Y_test_pred_NN_adam);
fprintf('Mean Squared Error (MSE) ontest set: %.4f\n', mse_nn_adam);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FPRINCIPAL COMPONENT ANALYSIS

% Performing normalization of data in range [0,1]
X_standardized = normalize(X, "range");

[~, score, ~, ~, explained] = pca(X_standardized);

% Create a scatter plot of the data in the space of the first two principal components
scatter(score(:,1),score(:,2))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
title('Scatter Plot of Data in the Space of the First Two Principal Components');

% Create a scatter plot of the data in the space of the first three principal components
scatter3(score(:,1),score(:,2),score(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')



% PCA with LR
% Getting MSE from linear regression as the basis for comparison to PCA-modified data

% Performing normalization of data in range [0,1]
X_standardized = normalize(X, "range");

% Split the data into training and testing sets
rng(10); 
cv = cvpartition(size(X_standardized, 1), 'HoldOut', 0.2); % 80% train, 20% test
X_train = X_standardized(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X_standardized(test(cv), :);
Y_test = Y(test(cv), :);

% Train linear regression model
linear_regression = fitlm(X_train, Y_train);
% Predict on the test set
Y_pred_lr = predict(linear_regression, X_test);
% Evaluate the model
mse_lr = immse(Y_test, Y_pred_lr); % Mean Squared Error
% Display evaluation metrics
fprintf('Mean Squared Error (MSE): %.4f\n', mse_lr);

% Performing normalization of data in range [0,1]
X_standardized = normalize(X, "range");

[coeff, score, latent] = pca(X_standardized);

num_components = 10;

% Keep only the selected principal components
X_pca = score(:, 1:num_components);

% Split the data into training and testing sets
rng(10); 
cv = cvpartition(size(X_pca, 1), 'HoldOut', 0.2); % 80% train, 20% test
X_train = X_pca(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X_pca(test(cv), :);
Y_test = Y(test(cv), :);

% Train linear regression model
linear_regression_pca = fitlm(X_train, Y_train);
% Predict on the test set
Y_pred_lr_pca = predict(linear_regression_pca, X_test);
% Evaluate the model
mse_lr_pca = immse(Y_test, Y_pred_lr_pca); % Mean Squared Error
% Display evaluation metrics
fprintf('Mean Squared Error (MSE) after PCA: %.4f\n', mse_lr_pca);



% PCA with NN
% Performing normalization of data in range [0,1]
X_standardized = normalize(X, "range");

[coeff, score, latent] = pca(X_standardized);

num_components = 12;

% Keep only the selected principal components
X_pca = score(:, 1:num_components);

% Split the data into training and testing sets
rng(10); 
cv = cvpartition(size(X_pca, 1), 'HoldOut', 0.2); % 80% train, 20% test
X_train = X_pca(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X_pca(test(cv), :);
Y_test = Y(test(cv), :);


% Define the architecture of the neural network
features = size(X_train, 2);

% 2 fully connected layers with relu activation
layers_NN_1 = [
    featureInputLayer(features)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(30)
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

% ADAM optimizer
options_NN_adam_1 = trainingOptions('adam', ... 
    'MaxEpochs', 100, ... 
    'MiniBatchSize', 32, ... 
    'InitialLearnRate', 0.01, ... 
    'Shuffle', 'every-epoch', ... 
    'ValidationData', {X_test, Y_test}, ... 
    'ValidationFrequency', 10, ... 
    'Verbose', false, ...
    'Plots', 'training-progress');

net_NN_adam_1_pca = trainNetwork(X_train, Y_train, layers_NN_1, options_NN_adam_1);

% Make predictions on the validation data:
Y_pred_NN_pca = predict(net_NN_adam_1_pca, X_test);
Y_pred_NN_pca = double(Y_pred_NN_pca);

mse_NN_pca = immse(Y_test, Y_pred_NN_pca);
fprintf('Mean Squared Error (MSE) on test data: %.4f\n', mse_NN_pca);
