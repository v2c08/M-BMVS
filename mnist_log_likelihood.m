function L = mnist_log_likelihood(P,M,U,Y)
% MDP log-likelihood function
% P    - parameter structure
% M    - generative model
% U    - inputs (observations or stimuli)
% Y    - observed responses (or choices)

if ~isstruct(P); P = spm_unvec(P,M.pE); end

% place MDP in trial structure
%--------------------------------------------------------------------------
n            = numel(U);

% solve MDP and accumulate log-likelihood
%--------------------------------------------------------------------------
for i = 1:n
    
    % Set trial stimulus
    setStimulus(M.STIM{i})
    
    % Initialise Model
    mdp     = M.G(P, M.map, M.T{i}, M.digit{i}, true);
    
    % Trial Observations & Responses
    mdp.o   = U{i};
    mdp.u   = Y{i};
    
    % Init states
    mdp.s = zeros(size(mdp.u,1),1);
    mdp.s(1) = 25;
    mdp.s(2) = M.digit{i};
    mdp.s(3) = 11;
    mdp.alpha = 2;
    MDP(i) = spm_MDP_VB_X(mdp);     
    
end

L     = 0;
for i = 1:n
    for t = 1:size(Y{i},2)
        sub = num2cell(Y{i}(:,t));
        L   = L + log(MDP(i).P(sub{:},t));
        assert(~isinf(L))
    end
end

