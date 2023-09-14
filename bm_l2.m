function [Uopt,lb,fopt,fval] = bm_l2(Weight, Bias, Cost, Image, Radius, Params)
% [UOPT, LB, FOPT, FVAL] = BM_L2(Weight, Bias, Cost, Image, Radius, Params)
% 
% BM_L2  Use Knitro to solve (BM-l2), which is a rank-r restricted SDP 
%        relaxation of NN verification problem:
%          min  <Cost, x{ell}> + Params.Offset
%          s.t. x{k+1} >= W{k}*x{k}+b{k}, x{k+1} >= 0,
%               LB{1} <= x{1} <= UB{1},
%               diag(X{k+1,k+1} - W{k}*X{k,k+1} - b{k}*x{k+1}') = 0,
%               tr(X{1,1}) - 2*x{1}'*Image + ||Image||^2 <= Radius^2,
%               [  1        x{1}'   ..    x{ell}'  ]
%               [ x{1}     X{1,1}   ..   X{1,ell}  ] is PSD and has maximal
%               [  .          .     ..       .     ] rank r.
%               [  .          .     ..       .     ]
%               [ x{ell}  X{1,ell}' ..  X{ell,ell} ]
%         for k = 1,...,ell-1
% 
% >  [UOPT] = BM_L2(Weight, Bias, Cost, Image, Radius, Params) yields an
%       global optimal solution to (BM-l2).
%           
%           [   1       x{1}'   ..    x{ell}'  ]   [   1       0    ][   1       0    ]'
%           [  x{1}    X{1,1}   ..   X{1,ell}  ]   [  u{1}    V{1}  ][  u{1}    V{1}  ]
%       Let [  .          .     ..       .     ] = [    .      .    ][    .      .    ] = UU' 
%           [  .          .     ..       .     ]   [    .      .    ][    .      .    ]
%           [ x{ell}  X{1,ell}' ..  X{ell,ell} ]   [ u{ell}  V{ell} ][ u{ell}  V{ell} ]
% 
%       (BM-l2) is definied as
%           min  <Cost, u{ell}> + Offset
%           s.t. u{k+1} >= W{k}*u{k}+b{k}, u{k+1} >= 0,
%                0 <= u{1} <= 1,
%                diag(u{k+1}*u{k+1}' + V{k+1}*V{k+1}') = ...
%                diag(W{k}*(u{k}+u{k+1}'+V{k}V{k+1}') - b{k}u{k+1}'),
%                ||u{1} - Image||^2 + ||V{1}||^2 <= Radius^2,
%           for k = 1,...,ell-1.
%       where
%           - ell is the number of layers in the neural network.
%           - u{1},...,u{ell}, V{1}, ...,V{ell} are optimization variables.
%       Inputs
%           Weight - a cell array contains the weights of a neural network.
%           Bias   - a cell array contains the biases of a neural network.
%           Cost   - a column vector in the objective.
%           Image  - original input image.
%           Radius - attack radius.
%           Params - a structure array with the following fields
%           (1) Params.Offset  - (default to 0) offset of the objective value.
%           (2) Params.UB      - (default to inf) upper bound of x{k}(:,1) for all k.
%           (3) Params.LB      - (default to -inf) lower bound of x{k}(:,1) for all k.
%           (4) Params.Debug   - (default to false) whether to enter debug mode
%           (5) Params.Display - (default to true) whether to display info
%       Outputs
%           UOPT - the local optimal of U.
%         
% >  [UOPT, LB] = BM_L2(...) also returns the lower bound on attack problem.
%           LB = <Cost,u{ell}> + Offset - epsilon_gap - epsilon_feas*||Uopt||^2
% 
% >  [UOPT, LB, FOPT] = BM_L2(...) also returns the optimal objective value 
%       in the struct array FOPT.
%       FOPT has the following fields:
%       (1) FOPT.pgd         - optimal objective value of PGD.
%       (2) FOPT.bm          - optimal objective value of (BM-l2).
%       (3) FOPT.esilon_gap  - duality gap of (BM-l2).
%       (4) FOPT.esilon_feas - minimum eigenvalue of S(y,z).
% 
% >  [UOPT, LB, FOPT, FVAL] = BM_L2(...) also store the history of objective
%       values and the number of iterations taken in the struct array FVAL. 
%       FVAL has the following fields:
%       (1) FVAL.iter - the number of iterations taken.
%       (2) FVAL.bm   - the history of objective values of rank-r BurerMonteiro reformulation.
%       (3) FVAL.r    - the search rank r.
% 
% See https://arxiv.org/abs/2211.17244 for more details.

% Author: Hong-Ming Chiu (hmchiu2@illinois.edu)
% Date:   15 Aug 2023

%--------------------------------------------------------------------------
%                      Input checks and clean
%--------------------------------------------------------------------------
[Weight, Bias, Cost, Image, Radius, Params, ell, n] = check_inputs(Weight, Bias, Cost, Image, Radius, Params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KNITRO SOLVER OPTIONS
MaxIterations = 1e3;       % Max iteration. Other options are definied in knitro.opt

% RIEMANNIAN STAIRCASE OPTIONS
StartSearchRank = ell+1;   % Initial search rank for burer-monteiro
MaxSearchRank   = ell+6;   % Maximum search rank for burer-monteiro

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%--------------------------------------------------------------------------
%                             Definitions
%--------------------------------------------------------------------------
% We partition an optimization variable in R^{nx*r} as variable = [x;vec(V)] 
ni  = size(Weight{1},2); % Number of neurons at the input layer
nu = sum(n);             % Number of neurons excluding the input
nx = nu + ni;            % Total number of neurons
img = 1:ni;              % x(img) = [x{1}]
sel = ni+1:nx;           % x(sel) = [x{2}; ...; x{ell}]

% Smat*x = x(sel)
% Wmat*x + bvec = [W{1}x{1}+b{1};...;W{ell-1}x{ell-1}+b{ell-1}]
Imat = sparse(1:ni, img, 1, ni, nx, ni);
Smat = sparse(1:nu, sel, 1, nu, nx, nu);
Wmat = [blkdiag(Weight{:}), sparse(nu, n(ell))];
bvec = cat(1,Bias{:});
WmatT = Wmat';
LB = [Params.LB(img); max(0,Params.LB(sel))];
UB = [Params.UB(img); max(0,Params.UB(sel))];

% Define matrices in the following generic problem
%   min  <c,u> + d
%   s.t. A*u + 0.5*Afun(u*u^T+V*V^T) >= b,
%        B*u + 0.5*Bfun(u*u^T+V*V^T)  = d
c = [sparse(nx-n(ell),1); Cost];

A_lin = [Imat; -Imat; Wmat-Smat; -Smat];
b_lin = [UB(img); -LB(img); -bvec; zeros(nu,1)];

A_nonlin = sparse(1,img,2*Image,1,nx,ni);
b_nonlin = norm(Image,2)^2-Radius^2;

A = [-A_lin; A_nonlin];
b = [-b_lin; b_nonlin]; 

B = -Smat.*bvec;
d = zeros(nu,1);

% -------------------------------------------------------------------------
%                         Objective value 
% -------------------------------------------------------------------------
function [f, df] = myobj(variables)
    f = [];
    if nargout <= 1
        f = c'*variables(1:nx)+Params.Offset;
    end
    if nargout > 1
        df = [c;sparse(nx*(r-1),1)];
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
    
    uV  = reshape(variables,nx,r);     % variable = [x;vec(V)]
    err = [uV(img,1)-Image, uV(img,2:end)];
    pre = [Wmat*uV(:,1)+bvec, Wmat*uV(:,2:end)]; 
    post = uV(sel,:);
    
    % Constraint functions
    if nargout <= 2
        g = sum(err(:).^2) - Radius^2;
        h = sum(post.*(post - pre), 2);
    end
    % Compute gradient
    if nargout > 2
        dg = 2*[sparse(err);sparse(nu,r)];
        dh = cell(1,r);
        for jj = 1:r
            dpre = -diag(sparse(post(:,jj))) * Wmat;
            dpost = sparse(1:nu, sel, 2*post(:,jj) - pre(:,jj), nu, nx, nu);
            dh{jj} = dpre + dpost;
        end
        dg = dg(:);
        dh = cat(2,dh{:})';
    end
end

% -------------------------------------------------------------------------
%                        Lagragian Hessian
% -------------------------------------------------------------------------
function [H] = myhess(~, lambda)
    % Only need to specify structural non-zero elements of the upper
    % triangle (including diagonal)
    lambda_g = lambda.ineqnonlin;
    lambda_h = lambda.eqnonlin;
    L = diag(sparse(lambda_h));
    
    H = sparse(1:nx,1:nx,2*[lambda_g*ones(ni,1); lambda_h],nx,nx,nx);
    H = H - [sparse(nx,ni), WmatT*L];
    H = kron(speye(r), H);
end

% -------------------------------------------------------------------------
%              Lagragian Hessian matrix-vector product
% -------------------------------------------------------------------------
function [Hv] = myhess_mv(~, lambda, variables)
    lambda_g = lambda.ineqnonlin;
    lambda_h = lambda.eqnonlin;
    uV = reshape(variables,nx,r);
    L = diag(sparse(lambda_h));
    
    Hv = -WmatT*(L*uV(sel,:));
    Hv(img,:) = Hv(img,:) + 2*lambda_g*uV(img,:);
    Hv(sel,:) = Hv(sel,:) + L*(2*uV(sel,:) - Wmat*uV);
    Hv = Hv(:);
end

% -------------------------------------------------------------------------
%              Get Jacobian and Hessian pattern for Knitro
% -------------------------------------------------------------------------
function [Jpattern, Hpattern] = get_pattern()
    x1 = rand(nx*r,1);
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
fval_store = struct();
fval = cell(0);
function stop = outfun(~,optimValues,state)
stop = false;
% This is a hack, printing an empty string will force Knitro to display 
% current iteration info in command window
if Params.Display; fprintf(' \b'); end
switch state
    case 'init'
        fval_store.iter = 0;
        fval_store.bm   = inf(1,MaxIterations+1);
    case 'iter'
        fval_store.iter = fval_store.iter+1; 
        fval_store.bm(fval_store.iter) = optimValues.fval;
    case 'done'
        fval_store.bm   = fval_store.bm(1:fval_store.iter);
        fval_store.r =r;
        fval{end+1} = fval_store;        
end
end

% -------------------------------------------------------------------------
%        These are the functions for constructing S(y,z)=[s0,s;s',0.5*S]
% -------------------------------------------------------------------------
function [S] = S_mat(lambda)
    % Form the entire matrix S, the bottom-right block of S(y,z)
    lambda_g = lambda.ineqnonlin;
    lambda_h = lambda.eqnonlin;
    L = diag(sparse(lambda_h));
    
    S = sparse(1:nx,1:nx,[lambda_g*ones(ni,1); lambda_h],nx,nx,nx);
    S = S - [sparse(nx,ni), Wmat'*L];
    S = S + S';
end
function [Sv] = S_mv(lambda, v)
    % Matrix-vector product between S and v
    lambda_g = lambda.ineqnonlin;
    lambda_h = lambda.eqnonlin;

    Sv = - WmatT*(lambda_h.*v(sel,:));
    Sv(img,:) = Sv(img,:) + 2*lambda_g.*v(img,:);
    Sv(sel,:) = Sv(sel,:) + lambda_h.*(2*v(sel,:) - Wmat*v);
%     Sv = Sv - WmatT*(lambda_h.*v(sel,:));
end
function Sv = Syz_mv(v)
    % Matrix-vector product between S(y,z) and v
    Sv = zeros(nx+1,1);
    Sv(1) = s0*v(1)+s'*v(2:end);
    Sv(2:end) = v(1)*s + 0.5*S_mv(lambda,v(2:end));
end

% Debug Gradient
if Params.Debug; r = 2; debug_grad(); end

%--------------------------------------------------------------------------
%                       SOLVE USING KNITRO
%--------------------------------------------------------------------------
% Initialization
Params_pgd = Params;
Params_pgd.Display  = false;
Params_pgd.Debug    = false;
u0 = cell(ell+1,1);
[u0{1}, fopt.pgd] = pgd_l2(Weight,Bias,Cost,Image,Radius,Params_pgd);
for k = 1:ell
    u0{k+1} = max(Weight{k}*u0{k}+Bias{k},0); 
end
V0 = 1e-10*randn(nx, StartSearchRank-1);
uV0 = [cell2mat(u0); V0(:)];

% Knitro options
if Params.Display
    opts = knitro_options('maxit', MaxIterations, 'outlev', 2);
else
    opts = knitro_options('maxit', MaxIterations, 'outlev', 0);
end

% Riemannian staircase
for r = StartSearchRank:MaxSearchRank
    % Inequality constraints for knitro : A_f*variable <= b_f
    A_f = [A_lin, sparse(numel(b_lin),nx*(r-1))];
    b_f = b_lin;
    
    % Feed Jacobian pattern, Hessian pattern and Hessian(mv) and output function to knitro
    [Jpattern, Hpattern] = get_pattern();
    extfeas.HessFcn      = @myhess;
    extfeas.HessMult     = @myhess_mv;
    extfeas.OutputFcn    = @outfun;
    extfeas.JacobPattern = Jpattern;
    extfeas.HessPattern  = Hpattern;
    
    outfun([],[],'init'); % This is a hack because knitro will never call outfun with status = init, we call it manually to initialize fval     
    [uVopt,fopt.bm,~,~,lambda] = knitro_nlp(@myobj, uV0, A_f, b_f, [], [], [], [], @mycon, extfeas, opts, 'knitro.opt'); % Call knitro    
    outfun([],[],'done'); % This is a hack because knitro will never call outfun with status = done, we call it manually to trim fval
    
    % Retrieve primal and dual variables
    uVopt = reshape(uVopt,nx,r);
    u = uVopt(:,1);
    V = uVopt(:,2:end);
    y = [lambda.ineqlin;lambda.ineqnonlin;];
    z = -[lambda.eqlin;lambda.eqnonlin];
    
    % Get slack matrix S(y,z) = [s0,s';s,0.5*S]
    % Since we are only interested in finding the smallest eigenvalue and the
    % corresponding eigenvector using the MATLAB internal function eigs(), we
    % do not have to form the entire S(y,z). All we need is an efficient 
    % implementation of the matrix-vector product S(y,z)*v.
    s  = 0.5*(c - A'*y - B'*z);
    s0 = -s'*u;
    [eigvec, eigval, eigflag] = eigs(@Syz_mv,nx+1,1,'smallestreal','IsFunctionSymmetric',true,'MaxIterations',1e3);
    if eigflag
        % Sometimes eigs() does not converge. In this case, we have to form the  
        % entire S(y,z) and use eig().
        Syz = full([s0,s';s,0.5*S_mat(lambda)]);
        [eigvec, eigval] = eig(Syz,'vector');
        [eigval, edx] = min(eigval);
        eigvec = eigvec(:,edx);
    end
    
    % Calculate lower bound
    epsilon_gap  = abs(c'*u - b'*y - d'*z + s0);
    epsilon_feas = max(0,-eigval);
    lb = Params.Offset + c'*u - epsilon_gap - epsilon_feas*(1+sum(uVopt(:).^2));
    
    % Store outputs
    Uopt = [1,zeros(1,r-1);uVopt];
    fopt.epsilon_gap = epsilon_gap;
    fopt.epsilon_feas = epsilon_feas;
    
    % Check KKT conditions and report objective value and lower bound
    if Params.Display
        ineqcon = [A_lin*u-b_lin;mycon(uVopt)]; % ineqcon <= 0
        [~,eqcon] = mycon(uVopt);               % eqcon = 0
        fprintf('\n<strong>KKT Conditions Check</strong>\n');
        maxerr_u  = max(abs(s + 0.5*S_mv(lambda, u)));
        maxerr_V  = max(max(abs(0.5*S_mv(lambda, V))));
        maxerr_y  = max(abs(ineqcon.*y));
        maxerr_z  = max(abs(eqcon));
        fprintf('max{|dL(u,V,y,z)/du|}:  %7.5e\n', maxerr_u);
        fprintf('max{|dL(u,V,y,z)/dV|}:  %7.5e\n', maxerr_V);
        fprintf('max{|dL(u,V,y,z)/dz|}:  %7.5e\n', maxerr_z);
        fprintf('max{|ineqcon.*y|}: %7.5e\n', maxerr_y);
        fprintf('numel(ineqcon > 0): %d\n', sum(ineqcon > 1e-6));
        fprintf('numel(y < 0): %d\n', sum(y < -1e-6));
    
        fprintf('\n<strong>Objective value and lower bound</strong>\n');
        fprintf('Obj (PGD)        : % 10.8f\n', fopt.pgd);
        fprintf('Obj (BM)         : % 10.8f\n', fopt.bm);
        fprintf('||x{1}-Image||_2 : % 10.8f\n', norm(u(img)-Image,2));
        fprintf('Radius           : % 10.8f\n', Radius);
        fprintf('epsilon_feas     : % 8.6e\n',  fopt.epsilon_feas);
        fprintf('epsilon_gap      : % 8.6e\n',  fopt.epsilon_gap);
        fprintf('lb               : % 10.8f\n', lb);
        fprintf('\n')
    end
        
    % Escape saddle point
    if epsilon_feas > 1e-6
        escape_dir = eigvec(1)*u + eigvec(2:end);  % compute escape direction
        uV0 = [uVopt(:); 0.01*escape_dir];         % increase search rank
        extfeas.ambdaInitial = lambda;             % initalize dual variable (only works for knitro)
    else
        break;
    end
end

%--------------------------------------------------------------------------
%                             Debug Functions
%--------------------------------------------------------------------------
function debug_grad()
seed = rng;
DEBUG_TESTS = 100;     % How many random tests?
FD_SIZE     = 1e-6;    % Perturbation size
fprintf('\n<strong>(DEBUG MESSAGE) Gradient Check (%d trials)</strong>\n', DEBUG_TESTS);
numvar = nx*r;
maxerr_obj = 0;
maxerr_ineq = 0;
maxerr_eq = 0;
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
    err_obj  =  ((f2 - f0)/(2*t) - df1'*dir) / norm(df1'*dir);
    maxerr_obj = max(norm(err_obj), maxerr_obj);

    % constraints inequality
    [g0,~]  =  mycon(x1-t*dir);
    [g2,~]  =  mycon(x1+t*dir);
    [~,~,dg0,~]  =  mycon(x1-t*dir);
    [~,~,dg1,~]  =  mycon(x1);
    [~,~,dg2,~]  =  mycon(x1+t*dir);
    err_ineq     =  ((g2 - g0)/(2*t) - dg1'*dir) / norm(dg1'*dir);
    maxerr_ineq  = max(norm(err_ineq), maxerr_ineq);

    % constraints equality
    [~,h0]  =  mycon(x1-t*dir);
    [~,h2]  =  mycon(x1+t*dir);
    [~,~,~,dh0]  =  mycon(x1-t*dir);
    [~,~,~,dh1]  =  mycon(x1);
    [~,~,~,dh2]  =  mycon(x1+t*dir);
    err_eq     =  ((h2 - h0)/(2*t) - dh1'*dir) / norm(dh1'*dir);
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

fprintf('Max objective gradient rel err:   %g\n', maxerr_obj');
fprintf('Max ineq constr gradient rel err: %g\n', maxerr_ineq');
fprintf('Max eq constr gradient rel err:   %g\n', maxerr_eq');
fprintf('Max Lagrangian Hessian rel err:   %g\n', maxerr_hess');
fprintf('Max Hessian mvprod rel err:       %g\n', maxerr_hessmv'); 
rng(seed)
end
end

