function [num,out] = predict(W,b,image)
ReLU  = @(X) max(X,0);
nl = numel(W);
image = reshape(image,size(W{1},2),[]);
out = ReLU(W{1}*image+b{1});
for l = 2:nl-1
    out = ReLU(W{l}*out+b{l});
end
out = W{nl}*out+b{nl};
[~,num] = max(out,[],1); 
num = num - 1;

end