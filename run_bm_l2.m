%% Prepare data
clear
addpath MNIST utils

% Select model to use
model_name = 'ADV-MNIST';
% model_name = 'LPD-MNIST';
% model_name = 'NOR-MNIST';

% Set attack radius, pick an image and pick a class to certify
Radius = 1;         % Attack radius
id = 1;             % Pick a image to certify
certify = 1;        % certify the class = mod(class_truth+certify,10) is safe

% Load model weights and correctly classified MNIST images
[W,b] = load_model(model_name);  % Load model weights
[X,y] = load_dataset(W,b);       % Load correctly classified MNIST images

% Solve BM-r using general purpose solver
[Weight, Bias, Cost, Image, Params] = get_arguments(W,b,X,y,id,certify);
[xopt,~,fopt,fval] = bm_l2(Weight, Bias, Cost, Image, Radius, Params);




