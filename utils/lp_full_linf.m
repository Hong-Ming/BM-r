function [xopt,fopt,fval] = lp_full_linf(Weight, Bias, Cost, Image, Radius, Params)
% [XOPT, FOPT, FVAL] = LP_FULL_LINF(Weight, Bias, Cost, Image, Radius, Params)
% 
% LP_FULL_LINF  Use Knitro to solve the the following linear program proposed
%               by Salman 2020
% 
%          min  <Cost, x{ell}> + Offset
%          s.t. LB{1} <= x{1} <= UB{1},
%               ||x{1} - Image||_inf <= Radius,
%               x{k+1} >= W{k}*x{k} + b{k}, x{k+1} >= 0,
%               x{k+1} <= (UB{k}/(UB{k}-LB{k})) * (W{k}*x{k}+b{k}-LB{k})
% 
% >  [XOPT] = LP_FULL_LINF(Weight, Bias, Cost, Image, Radius, Params) yields the
%       optimal solution to the above linear program
%           
%       Inputs
%           Weight - a cell array contains the weights of a neural network.
%           Bias   - a cell array contains the biases of a neural network.
%           Cost   - a column vector in the objective.
%           Image  - original input image.
%           Radius - attack radius.
%           Params - a structure array with the following fields
%           (1) Params.Offset  - (default to 0) offset of the objective value.
%           (2) Params.UB      - (default to inf) upper bound of x{k} for all k.
%           (3) Params.LB      - (default to -inf) lower bound of x{k} for all k.
%           (4) Params.Debug   - (default to false) whether to enter debug mode
%           (5) Params.Display - (default to true) whether to display info
%       Outputs
%           XOPT - the local optimal of x.
% 
% >  [XOPT, FOPT] = LP_FULL_LINF(...) also returns the optimal objective value.
% 
% >  [UOPT, FOPT, FVAL] = LP_FULL_LINF(...) also store the history of objective
%       values in the struct array FVAL. 
%       FVAL has the following fields:
%       (1) FVAL.lp   - the history of objective values of the linear program.
%       (2) FVAL.iter - the number of iterations taken.

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   15 Aug 2023

%--------------------------------------------------------------------------
%                      Input checks and clean
%--------------------------------------------------------------------------
[Weight, Bias, Cost, Image, Radius, Params, ell, n] = check_inputs(Weight, Bias, Cost, Image, Radius, Params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KNITRO SOLVER OPTIONS
MaxIterations = 1e4;      % Max iteration. Other options are definied in knitro.opt

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
%                             Definitions
%--------------------------------------------------------------------------
% We partition an optimization variable in R^{nx*r} as variable = [x;vec(V)] 
% no is the number of neurons at the output layer
ni = numel(Image);       % ni is the number of neurons at the input layer
nu = sum(n);             % Number of neurons excluding the input
nx = nu + ni;            % Total number of neurons
if ell == 0
    no = numel(Cost);
else
    no = n(ell);
end
img = 1:ni;           % x(img) = [x{1}]
sel = ni+1:nx;        % x(sel) = [x{2}; ...; x{ell}]

% Smat*x = x(sel)
% Wmat*x + bvec = [W{1}x{1}+b{1};...;W{ell-1}x{ell-1}+b{ell-1}]
Imat = sparse(img,img,1,ni,nx,ni);
Smat = sparse(1:nu, sel, 1, nu, nx, nu);
Wmat = [blkdiag(Weight{:}), sparse(nu, no)];
bvec = cat(1,Bias{:});
LB = Params.LB; UB = Params.UB;
sigma = UB(sel)./(UB(sel)-LB(sel));

% Find out neurons that are always active or inactive
active = LB(sel) >= 0;
inactive = UB(sel) <= 0;
keep_sel = ~active & ~inactive;

% Define matrices in the following generic problem
%   min  <c,x>
%   s.t. A*x <= b,
%        B*x =  d
c = [sparse(nx-no,1); Cost];

A_lin = [-Imat; Imat; Wmat-Smat; -Smat; Smat-sigma.*Wmat];
b_lin = [-max(LB(img),Image-Radius); min(UB(img),Image+Radius); -bvec; zeros(nu,1); sigma.*(bvec-LB(sel))];
A_lin = A_lin([true(2*ni,1);keep_sel;keep_sel;keep_sel],:);
b_lin = b_lin([true(2*ni,1);keep_sel;keep_sel;keep_sel]);

B_lin = [Smat-Wmat; Smat];
d_lin = [bvec; zeros(nu,1)];
B_lin = B_lin([active;inactive],:);
d_lin = d_lin([active;inactive]);

% -------------------------------------------------------------------------
%       Output function for Knitro (for storing per iteration info)
% -------------------------------------------------------------------------
fval = struct();
function stop = outfun(~,optimValues,state)
stop = false;
% This is a hack, printing an empty string will force Knitro to display 
% current iteration info in command window
if Params.Display; fprintf(' \b'); end
switch state
    case 'init'
        fval.iter = 0;
        fval.lp   = inf(1,MaxIterations+1);
    case 'iter'
        fval.iter = fval.iter+1; 
        fval.lp(fval.iter) = optimValues.fval;
    case 'done'
        fval.lp   = fval.lp(1:fval.iter);      
end
end

%--------------------------------------------------------------------------
%                       SOLVE USING KNITRO
%--------------------------------------------------------------------------
% Initialization
if ell == 0
    x0 = Image;
else
    x0 = cell(ell+1,1);
    Params_pgd = Params;
    Params_pgd.Display = false;
    Params_pgd.Debug   = false;
    [x0{1}, pgdobj] = pgd_linf(Weight,Bias,Cost,Image,Radius,Params_pgd);
    for k = 1:ell
        x0{k+1} = max(Weight{k}*x0{k}+Bias{k},0); 
    end
    x0 = cell2mat(x0);
end

% Knitro options
if Params.Display
    opts = knitro_options('maxit', MaxIterations, 'outlev', 2);
else
    opts = knitro_options('maxit', MaxIterations, 'outlev', 0);
end

% Feed Jacobian pattern, Hessian pattern and Hessian(mv) and output function to knitro
extfeas.OutputFcn = @outfun;

outfun([],[],'init'); % this is a hack because knitro will never call outfun with status = init, we call it manually to initialize fval   
[xopt,fopt] = knitro_lp(c, A_lin, b_lin, B_lin, d_lin, [], [], x0, extfeas, opts, 'knitro.opt'); % call knitro    
outfun([],[],'done'); % this is a hack because knitro will never call outfun with status = done, we call it manually to trim fval

fopt = fopt + Params.Offset;
if Params.Display
    % Report final cost and violations
    fprintf('\n<strong>Objective value and lower bound:</strong>\n')
    fprintf('Obj (PGD)          : % 17.13f\n', pgdobj);
    fprintf('Obj (LP)           : % 17.13f\n', fopt);
    fprintf('||x{1}-Image||_inf : % 17.13f\n', norm(xopt(img)-Image,inf));
    fprintf('Radius             : % 17.13f\n', Radius);
    fprintf('\n')
end

end

