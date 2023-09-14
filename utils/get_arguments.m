function [Weight, Bias, Cost, Image, Params] = get_arguments(W,b,X,y,id,certify)
class_true = y(id);                     % Truth class
class_certify = mod(class_true+certify,10);     % Desired attack class
Weight = W(1:end-1);
Bias   = b(1:end-1);
Cost   = W{end}(class_true+1,:)' - W{end}(class_certify+1,:)';
Image  = X(:,id);
ni = numel(Image); nu = numel(cat(1,Bias{:}));
Params.Offset = b{end}(class_true+1) - b{end}(class_certify+1);
Params.Debug = true;
Params.UB = [ones(ni,1) ; inf(nu,1)];         % Upper bound on u{k}
Params.LB = [zeros(ni,1);-inf(nu,1)];         % Lower bound on u{k}
end