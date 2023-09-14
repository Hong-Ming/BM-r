function [W,b] = load_model(model_name)

% model = load(['MNIST/',model_name,'.mat']);
model = load([model_name,'.mat']); 
W = model.W; 
b = model.b;

end