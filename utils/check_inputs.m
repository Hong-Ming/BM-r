function [Weight, Bias, Cost, Image, Radius, Params, ell, n] = check_inputs(Weight, Bias, Cost, Image, Radius, Params)

% Check "Weight" and "Bias"
assert(iscell(Weight) && iscell(Bias), 'must provide weights and biases as cells')
assert(numel(Bias) == numel(Weight), 'number of weight matrices must match number of bias vectors');
ell = numel(Weight);       % ell is the number of hidden layers
n   = zeros(1,ell);        % n(k) is the number of neurons at the k-th hiden layer
for k = 1:ell
    n(k) = size(Weight{k},1); Bias{k} = Bias{k}(:);
    assert(size(Weight{k},1) == numel(Bias{k}), 'number of rows in the %i-th weight matrix mismatches the number of elements in the %i-th bias vector',k,k);
    if k > 1
        assert(size(Weight{k},2) == n(k-1), 'number of columns in the %i-th weight matrix mismatches the number of rows in the %i-th weight matrix',k,k-1)
    end
end
ni = numel(Image);       % Number of neurons at the input layer
nu = sum(n);             % Number of neurons excluding the input
nx = nu + ni;            % Total number of neurons
if ell == 0              % Number of neurons at the output layer
    no = numel(Cost);
else
    no = n(ell);
end

% Check "Cost", "Image" and "Radius"
Cost = Cost(:); Image = Image(:);
assert(numel(Cost)   == no, 'number of elements in "Cost" mismatches number of outputs at the final layer');
assert(numel(Image)   == ni,    'number of elements in "Orig" must match number of neurons in the first layer');
assert(numel(Radius) == 1,      'radius must be a scalar');
% Check "Params"
if nargin < 6 || isempty(Params); Params = struct();     end
assert(isstruct(Params), '"Params" must be a structure array')
if ~isfield(Params,'Offset');  Params.Offset  = 0;      end
if ~isfield(Params,'UB');      Params.UB      = inf(nx,1);  end
if ~isfield(Params,'LB');      Params.LB      = -inf(nx,1); end
if ~isfield(Params,'Debug');   Params.Debug   = false;  end
if ~isfield(Params,'Display'); Params.Display = true;   end
assert(numel(Params.Offset) == 1, 'number of elements in "Params.Offset" mismatches number of input images');
assert(numel(Params.UB) == nx, 'number of elements in "Params.UB" mismatches number of neurons');
assert(numel(Params.LB) == nx, 'number of elements in "Params.LB" mismatches number of neurons');
assert(islogical(Params.Debug), '"Params.Debug" must to logical, set it to true of false');
assert(islogical(Params.Display), '"Params.Display" must to logical, set it to true of false');
valid_field = {'Offset','LB','UB','Debug','Display'}; field_names = fieldnames(Params);
for k = 1:numel(field_names)
    assert(any(strcmp(field_names{k},valid_field)), 'unknown field "%s" in "Params"',field_names{k})
end

end