function [UB, LB] = get_bound_linf(W,b,Image,Radius,DoPrint)
if nargin < 5; DoPrint = true; end
Params.Offset   = 0;
Params.UB = ones(numel(Image),1);
Params.LB = zeros(numel(Image),1);
Params.Debug    = false;
Params.Display  = false;
workers = 10;
ell = numel(W);
for l = 1:ell
    ThisW = W{l}; Weight = W(1:l-1);
    Thisb = b{l}; Bias = b(1:l-1);
    Wpar = cell(workers,1);

    str = 1;
    for k = 1:workers
        if k <= mod(numel(Thisb),workers)
            len = ceil(numel(Thisb)/workers);
        else
            len = floor(numel(Thisb)/workers);
        end
        Wpar{k} = ThisW(str:str+len-1,:);  
        str = str + len;
    end

    ub = cell(workers,1); lb = cell(workers,1); w = zeros(workers,1);
    parfor pdx = 1:workers
        Cpar = Wpar{pdx}; batchs = size(Cpar,1);
        pub = zeros(batchs,1); plb = zeros(batchs,1); pw = 0; 
        for batch = 1:batchs
            Cost = Cpar(batch,:)';
            [~,pub(batch)] = lp_full_linf(Weight, Bias, -Cost, Image, Radius, Params);
            [~,plb(batch)] = lp_full_linf(Weight, Bias,  Cost, Image, Radius, Params);
            if pdx == workers && DoPrint
                pw = fprintf([repmat('\b',1,pw),'Layer: %d, neuron: %d/%d\n'],l,batch*workers,numel(Thisb)) - pw; 
            end
        end
        ub{pdx} = pub; lb{pdx} = plb; w(pdx) = pw;
    end
    if DoPrint
        fprintf([repmat('\b',1,w(end)),'Layer: %d, neuron: %d/%d\n'],l,numel(Thisb),numel(Thisb))
    end
    Params.UB = [Params.UB; -cell2mat(ub) + Thisb];
    Params.LB = [Params.LB;  cell2mat(lb) + Thisb];
end
UB = Params.UB;
LB = Params.LB;
end