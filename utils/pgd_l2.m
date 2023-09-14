function [xopt,fopt,fval] = pgd_l2(Weight, Bias, Cost, Image, Radius, Params)
% [XOPT, FOPT, FVAL] = PGD_L2(Weight, Bias, Cost, Image, Radius, Params)
% 
%  PGD_L2  Find a strong first-order attack for neural networks.
% 
%       fopt = min  <Cost, x{ell}> + Offset
%              s.t. x{k+1} = ReLU(W{k}x{k}+b{k}) for k = 1,...,ell-1,
%                   lb <= x{1} <= ub,
%                   ||x{1}-Image||_2 <= Radius.
%       
%       where
%           - ell is the number of layers in the neural network.
%           - x{1} is the optimization variable
%           - ReLU(x) = max(x,0) is the relu gate.
% 
% >  [XOPT] = PGD_L2(Weight, Bias, Cost, Image, Radius, Params) yields an local 
%       optimal solution to the above problem
% 
%       Inputs
%           Weight - a cell array contains the weights of a neural network.
%           Bias   - a cell array contains the biases of a neural network.
%           Cost   - a column vector in the objective.
%           Image  - original input image.
%           Radius - attack radius.
%           Params - a structure array with the following fields
%           (1) Params.Offset   - (default to 0) offset of the objective value.
%           (2) Params.UB       - (default to inf) upper bound of x{k} for all k.
%           (3) Params.LB       - (default to -inf) lower bound of x{k} for all k.
%           (4) Params.Debug    - (default to false) whether to enter debug mode
%           (5) Params.Display  - (default to true) whether to display info
%       Outputs
%           XOPT - the local optimal of x{1}.
% 
% >  [XOPT, FOPT] = PGD_L2(...) also returns the optimal objective value FOPT.
% 
% >  [XOPT, FOPT, FVAL] = PGD_L2(...) also store the history of objective
%       values and the number of iterations taken in the struct array FVAL. 
%       FVAL has the following fields:
%       (1) FVAL.pgd  - the history of objective value..
%       (2) FVAL.iter - the number of iterations taken.

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   15 Aug 2023

%--------------------------------------------------------------------------
%                      Input checks and clean
%--------------------------------------------------------------------------
[Weight, Bias, Cost, Image, Radius, Params, ell] = check_inputs(Weight, Bias, Cost, Image, Radius, Params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PGD SOLVER OPTIONS
StagnationLimit = 1e2;  % Terminate if no progress is made after this many iterations
MaxIterations   = 1e4;  % Terminate if reach this many iterations
PrintFreq       = 1000; % Change line in command window after this many iteration
lr              = 1e-3; % Learning rate for x
momentum        = 0.9;  % Momentum for SGD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
%                         Definitions functions
%--------------------------------------------------------------------------
ni  = size(Weight{1},2);
ub = Params.UB(1:ni);
lb = Params.LB(1:ni);

% Objective and gradient for pgd
function [f, dx] = myobj(x)
    % Forward propagation
    z = cell(ell,1);                   % Preactivations
    for i = 1:ell
        z{i} = Weight{i}*x + Bias{i};  % Store preactivation
        x = max(z{i},0);               % Activate
    end
    f = Cost'*x + Params.Offset; 
    
    if nargout < 2; return; end
    % Backward propagation
    dx = Cost;
    for i = ell:-1:1
        dx = Weight{i}'*((z{i}>0).*dx);
    end
end

% Project x onto 
% (1) ||x - Image||_2 <= Radius
% (2) lb <= x <= ub
function out = myproj(x)
    out = x;
    out = min(out,ub);
    out = max(out,lb);
    out = out - Image;
    out = out / max(norm(out)/Radius,1);
    out = Image + out;
end

% Debug Gradient
if Params.Debug; debug_grad(); end
%-------------------------------------------------------------------------- 
%                   RUN PROJECTED GRADIENT DESCENT
%-------------------------------------------------------------------------- 
if Params.Display
    w1 = fprintf(repmat('*',1,75));fprintf('*\n');
    w2 = fprintf('* Solver: PSGD + Nesterov');
    fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
    w2 = fprintf('* max iteration: %d, stagnation limit: %d',MaxIterations,StagnationLimit);
    fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
    w2 = fprintf('* momentum: %3.1e, learning rate: %3.1e',momentum,lr);
    fprintf(repmat(' ',1,w1-w2));fprintf('*\n');
    fprintf(repmat('*',1,w1));fprintf('*\n');
end

% Define variables for PGD
init_point = myproj(Image+randn(ni,1));
variable = init_point;
velocity = 0*init_point;
best = init_point;
fbest = inf; 
fval.pgd = inf(1,MaxIterations);
stagnation_count = 0;
wordcount = 0;

for iter = 1:MaxIterations
    % Update variable
    candidate = variable + momentum * velocity;
    [~, dx] = myobj(candidate);
    velocity = momentum * velocity - lr * dx;
    variable = myproj(variable  + velocity);
    fval.pgd(iter) = myobj(variable);
    if Params.Display
        fprintf(repmat('\b',1,wordcount));
        wordcount = fprintf('Iter: %4d, Obj(BM1): % 4.2e',iter, fval.pgd(iter));
    end
    
    % Termination conditions
    if fval.pgd(iter) < fbest
        fbest = fval.pgd(iter);
        best = variable;
        stagnation_count = 0;
    else
        stagnation_count = stagnation_count +1;
    end
    if stagnation_count > StagnationLimit
        if Params.Display
        	fprintf('\n  <<a href="">The algorithm terminates because it reach the stagnation limit</a>>\n')
        end
        break;
    end
    if Params.Display
        if mod(iter,PrintFreq)==0; wordcount = fprintf('\n') - 1; end
    end
    
    % Debug Projection
    if Params.Debug; debug_proj(); end
end
if iter == MaxIterations && Params.Display
   if mod(iter,PrintFreq)~=0; fprintf('\n'); end
   fprintf('  <<a href="">The algorithm terminates because it reach the maximum number of iterations</a>>\n')
end

% Define outputs
xopt = best; 
fopt = fbest;
fval.pgd = fval.pgd(1:iter);
fval.iter = iter;

if Params.Display
    fprintf('<strong>Final objectives</strong>\n')
    fprintf('Obj           : % 17.13f\n', fopt);
    fprintf('||x-Image||_2 : % 17.13f\n', norm(xopt - Image,2));
    fprintf('Radius        : % 17.13f\n', Radius);
    fprintf('\n') 
end

function debug_grad()
    seed = rng;
    DEBUG_TESTS = 100;     % How many random tests?
    FD_SIZE     = 1e-6;    % Perturbation size
    fprintf('\n<strong>(DEBUG MESSAGE) Gradient Check (%d trials)</strong>\n', DEBUG_TESTS);
    t  = FD_SIZE;
    maxerr_obj = 0;
    for trial = 1:DEBUG_TESTS
        x1 = randn(ni,1);
        d  = randn(ni,1);
        [~, df1]  =  myobj(x1);
        [f2,~  ]  =  myobj(x1+t*d);
        [f0,~  ]  =  myobj(x1-t*d);
        err_obj  =  (f2-f0)/(2*t) - df1'*d;
        maxerr_obj = max(norm(err_obj), maxerr_obj);
    end
    fprintf('Max objective gradient rel err (x):   %g\n', maxerr_obj');
    rng(seed)
end
function debug_proj()
    dif1 = norm(variable-Image,2)-Radius;
    [dif2,idx2] = max(variable-ub);
    [dif3,idx3] = max(lb-variable);
    if dif1 > 1e-6
        wordcount = 0;
        fprintf('\n<strong>(DEBUG MESSAGE) Constraint violation detected</strong>\n'); 
        fprintf('|x-Image|_2-Radius: %4.2e\n',dif1);
        fprintf('PRESS ANY KEY TO CONTINUE....\n');
        pause; 
    end
    if dif2 > 1e-6
        wordcount = 0;
        fprintf('\n<strong>(DEBUG MESSAGE) Constraint violation detected</strong>\n'); 
        fprintf('x(%i)-ub(%i): %4.2e\n',idx2,idx2,dif2);
        fprintf('PRESS ANY KEY TO CONTINUE....\n');
        pause; 
    end
    if dif3 > 1e-6
        wordcount = 0;
        fprintf('\n<strong>(DEBUG MESSAGE) Constraint violation detected</strong>\n'); 
        fprintf('lb(%i)-x(%i): %4.2e\n',idx3,idx3,dif3);
        fprintf('PRESS ANY KEY TO CONTINUE....\n');
        pause; 
    end
end
end

