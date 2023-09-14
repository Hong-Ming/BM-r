function [xopt,fopt,fval] = lp_full_l2(Weight, Bias, Cost, Image, Radius, Params)
% [XOPT, FOPT, FVAL] = LP_FULL_L2(Weight, Bias, Cost, Image, Radius, Params)
% 
% LP_FULL_L2  Use Knitro to solve the the following linear program proposed
%             by Salman 2020
% 
%          min  <Cost, x{ell}> + Offset
%          s.t. LB{1} <= x{1} <= UB{1},
%               ||x{1} - Image||_2 <= Radius,
%               x{k+1} >= W{k}*x{k} + b{k}, x{k+1} >= 0,
%               x{k+1} <= (UB{k}/(UB{k}-LB{k})) * (W{k}*x{k}+b{k}-LB{k})
% 
% >  [XOPT] = LP_FULL_L2(Weight, Bias, Cost, Image, Radius, Params) yields the
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
% >  [XOPT, FOPT] = LP_FULL_L2(...) also returns the optimal objective value.
% 
% >  [UOPT, FOPT, FVAL] = LP_FULL_L2(...) also store the history of objective
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

% Imat*x = x(img)
% Smat*x = x(sel)
Imat = sparse(img,img,1,ni,nx,ni);
Smat = sparse(1:nu, sel, 1, nu, nx, nu);
Wmat = [blkdiag(Weight{:}), sparse(nu, no)];
bvec = cat(1,Bias{:});
LB = Params.LB; UB = Params.UB;
sigma = UB(sel)./(UB(sel)-LB(sel));

% Find out neurons that are always active or inactive
active = LB(sel) >= 0;
inactive = UB(sel) <= 0;
keep_img = [Image-Radius < LB(img); Image+Radius > UB(img)];
keep_sel = ~active & ~inactive;

% Define matrices in the following generic problem
%   min  <c,x>
%   s.t. A*x <= b,
%        B*x =  d
c = [sparse(nx-no,1); Cost];

A_lin = [-Imat; Imat; Wmat-Smat; -Smat; Smat-sigma.*Wmat];
b_lin = [-LB(img); UB(img); -bvec; zeros(nu,1); sigma.*(bvec-LB(sel))];
A_lin = A_lin([keep_img;keep_sel;keep_sel;keep_sel],:);
b_lin = b_lin([keep_img;keep_sel;keep_sel;keep_sel]);

B_lin = [Smat-Wmat; Smat];
d_lin = [bvec; zeros(nu,1)];
B_lin = B_lin([active;inactive],:);
d_lin = d_lin([active;inactive]);

% -------------------------------------------------------------------------
%                         Objective value 
% -------------------------------------------------------------------------
function [f, df] = myobj(variables)
    f = [];
    if nargout <= 1
        f = c'*variables+Params.Offset;
    end
    if nargout > 1
        df = c;
    end
end

% -------------------------------------------------------------------------
%                       Nonlinear constraints 
% -------------------------------------------------------------------------
function [g,h,dg,dh] = mycon(variables)
    % g : inequality constraint
    % h : equality constraint
    g = [];
    h = [];    
    
    err = variables(img)-Image;
    
    % Constraint functions
    if nargout <= 2
        g = sum(err(:).^2) - Radius^2;
    end
    % Compute gradient
    if nargout > 2
        dg = sparse(img,1,2*err,nx,1,ni);
        dh = [];
    end
end

% -------------------------------------------------------------------------
%                        Lagragian Hessian
% -------------------------------------------------------------------------
function [H] = myhess(~, lambda)
    % Only need to specify structural non-zero elements of the upper
    % triangle (including diagonal)
    lambda_g = lambda.ineqnonlin;
    
    H = diag(sparse(1,img,2*lambda_g,1,nx,ni));
end

% -------------------------------------------------------------------------
%              Lagragian Hessian matrix-vector product
% -------------------------------------------------------------------------
function [Hv] = myhess_mv(~, lambda, variables)
    lambda_g = lambda.ineqnonlin;
    
    Hv = sparse(img,1,2*lambda_g*variables(img),nx,1,ni);
end

% -------------------------------------------------------------------------
%              Get Jacobian and Hessian pattern for Knitro
% -------------------------------------------------------------------------
function [Jpattern, Hpattern] = get_pattern()
    x1 = rand(nx,1);
    [g1,h1]  =  mycon(x1);
    lam  = struct();
    lam.ineqnonlin = rand(numel(g1),1);
    lam.eqnonlin = rand(numel(h1),1);
    [~,~,dg1,dh1]  =  mycon(x1);
    H1 = myhess(x1,lam);
    H1 = H1 + H1';
    dg1(dg1~=0) = 1;
    dh1(dh1~=0) = 1;
    H1(H1~=0) = 1;
    Jpattern = sparse([dg1';dh1']);
    Hpattern = sparse(H1);    
end

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

% Debug Gradient
if Params.Debug; debug_grad(); end

%--------------------------------------------------------------------------
%                       SOLVE USING KNITRO
%--------------------------------------------------------------------------
% Initialization
if ell == 0
    x0 = Image;
    pgdobj = nan;
else
    x0 = cell(ell+1,1);
    Params_pgd = Params;
    Params_pgd.Display = false;
    Params_pgd.Debug   = false;
    [x0{1},pgdobj] = pgd_l2(Weight,Bias,Cost,Image,Radius,Params_pgd);
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
[Jpattern, Hpattern] = get_pattern();
extfeas.HessFcn      = @myhess;
extfeas.HessMult     = @myhess_mv;
extfeas.OutputFcn    = @outfun;
extfeas.JacobPattern = Jpattern;
extfeas.HessPattern  = Hpattern;

outfun([],[],'init'); % this is a hack because knitro will never call outfun with status = init, we call it manually to initialize fval       
[xopt,fopt,~,~,lambda] = knitro_nlp(@myobj, x0, A_lin, b_lin, B_lin, d_lin, [], [], @mycon, extfeas, opts, 'knitro.opt'); % call knitro
outfun([],[],'done'); % this is a hack because knitro will never call outfun with status = done, we call it manually to trim fval

% check KKT condition
if Params.Display
    % retrieve primal and dual variables
    x = xopt;
    y = [lambda.ineqlin;lambda.ineqnonlin;];
    z = -[lambda.eqlin;lambda.eqnonlin];
    A = [-A_lin;sparse(1,img,2*Image,1,nx,ni);];
    B = B_lin;

    ineqcon = [A_lin*x-b_lin;mycon(xopt)]; % ineqcon <= 0
    eqcon = B_lin*x-d_lin;
    fprintf('\n<strong>KKT Conditions Check</strong>\n');
    maxerr_x  = max(abs(c - A'*y - B'*z + myhess_mv([],lambda,x)));
    maxerr_y  = max(abs(ineqcon.*y));
    maxerr_z  = max(abs(eqcon));
    fprintf('max{|dL(x,y,z)/dx|}:  %7.5e\n', maxerr_x);
    fprintf('max{|dL(x,y,z)/dz|}:  %7.5e\n', maxerr_z);
    fprintf('max{|ineqcon.*y|}: %7.5e\n', maxerr_y);
    fprintf('numel(ineqcon > 0): %d\n', sum(ineqcon > 1e-6));
    fprintf('numel(y < 0): %d\n', sum(y < -1e-6));
    
    % Report final cost and violations
    fprintf('\n<strong>Objective value and lower bound:</strong>\n')
    fprintf('Obj (PGD)        : % 17.13f\n', pgdobj);
    fprintf('Obj (LP)         : % 17.13f\n', fopt);
    fprintf('||x{1}-Image||_2 : % 17.13f\n', norm(xopt(img)-Image,2));
    fprintf('Radius           : % 17.13f\n', Radius);
    fprintf('\n')
end

%--------------------------------------------------------------------------
%                             Debug Functions
%--------------------------------------------------------------------------
function debug_grad()
seed = rng;
DEBUG_TESTS = 100;     % How many random tests?
FD_SIZE     = 1e-6;    % Perturbation size
fprintf('\n<strong>(DEBUG MESSAGE) Gradient Check (%d trials)</strong>\n', DEBUG_TESTS);
numvar = nx;
maxerr_obj = 0; err_obj = [];
maxerr_ineq = 0; err_ineq = [];
maxerr_eq = 0; err_eq = [];
maxerr_hess = 0;
maxerr_hessmv = 0;
for trial = 1:DEBUG_TESTS
    x1  = randn(numvar,1);
    dir = randn(numvar,1);
    t   = FD_SIZE;

    % objective
    f0  =  myobj(x1-t*dir);
    f2  =  myobj(x1+t*dir);
    [~,df0]  =  myobj(x1-t*dir);
    [~,df1]  =  myobj(x1);
    [~,df2]  =  myobj(x1+t*dir);
    if ~isempty(df1); err_obj  =  ((f2 - f0)/(2*t) - df1'*dir) / norm(df1'*dir); end
    maxerr_obj = max(norm(err_obj), maxerr_obj);

    % constraints inequality
    [g0,~]  =  mycon(x1-t*dir);
    [g2,~]  =  mycon(x1+t*dir);
    [~,~,dg0,~]  =  mycon(x1-t*dir);
    [~,~,dg1,~]  =  mycon(x1);
    [~,~,dg2,~]  =  mycon(x1+t*dir);
    if ~isempty(dg1); err_ineq = ((g2 - g0)/(2*t) - dg1'*dir) / norm(dg1'*dir); end
    maxerr_ineq  = max(norm(err_ineq), maxerr_ineq);

    % constraints equality
    [~,h0]  =  mycon(x1-t*dir);
    [~,h2]  =  mycon(x1+t*dir);
    [~,~,~,dh0]  =  mycon(x1-t*dir);
    [~,~,~,dh1]  =  mycon(x1);
    [~,~,~,dh2]  =  mycon(x1+t*dir);
    
    if ~isempty(dh1); err_eq = ((h2 - h0)/(2*t) - dh1'*dir) / norm(dh1'*dir); end
    maxerr_eq  = max(norm(err_eq), maxerr_eq);

    % Hessian
    lam  = abs(randn(numel(g0)+numel(h0),1));
    lam_struct.ineqnonlin = lam(1:numel(g0));
    lam_struct.eqnonlin = lam(numel(g0)+1:end);
    H0 = myhess(x1,lam_struct);
    d2f1_d = (H0+H0'-diag(diag(H0)))*dir;
    dgh2 = [dg2,dh2]; dgh0 = [dg0,dh0];
    tmp = ((df2 + dgh2*lam) - (df0 + dgh0*lam))/(2*t);
    err_hess = (tmp - d2f1_d) / norm(d2f1_d);
    err_hessmv = (d2f1_d - myhess_mv(x1,lam_struct,dir)) / norm(d2f1_d);
    maxerr_hess = max(norm(err_hess), maxerr_hess);
    maxerr_hessmv = max(norm(err_hessmv), maxerr_hessmv);
end

fprintf('Max objective gradient rel err:   %g\n', maxerr_obj);
fprintf('Max ineq constr gradient rel err: %g\n', maxerr_ineq);
fprintf('Max eq constr gradient rel err:   %g\n', maxerr_eq);
fprintf('Max Lagrangian Hessian rel err:   %g\n', maxerr_hess);
fprintf('Max Hessian mvprod rel err:       %g\n', maxerr_hessmv); 
rng(seed)
end

end

