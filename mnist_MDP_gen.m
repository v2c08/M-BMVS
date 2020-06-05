function MDP = mnist_MDP_gen(pE, map_mode,T,d,is_dem)           
% Generate MDP structure with priors
% pE                    - Prior expectations about parameters
% map_mode              - one of 'itti', ,'aws' or 'gbvs'
% T                     - Maximum number of saccades
% d                     - True digit
% is_dem                - Use continuous model 
% 
% MDP                   - MDP structure
% MDP.U(1,P,F)          - P allowable actions at each move
% MDP.T                 - number of outcomes
%
% MDP.A{G}(O,N1,...,NF) - likelihood of O outcomes given hidden states
% MDP.B{F}(NF,NF,MF)    - transitions among states under MF control states
% MDP.C{G}(O,T)         - (log) prior preferences for outcomes (modality G)
% MDP.D{F}(NF,1)        - prior probabilities over initial states
%
% MDP.s(F,T)            - matrix of true states - for each hidden factor
% MDP.o(G,T)            - matrix of outcomes    - for each outcome modality
% or .O{G}(O,T)         - likelihood matrix     - for each outcome modality
% MDP.u(F,T - 1)        - vector of actions     - for each hidden factor
%
% MDP.alpha             - precision - action selection [512]
% MDP.beta              - precision over precision (Gamma hyperprior - [1])
% MDP.demi.C            - Mixed model: cell array of true causes (DEM.C)
% MDP.demi.U            - Bayesian model average (DEM.U) see: spm_MDP_DEM
% MDP.link              - link array to generate outcomes from
%                         subordinate MDP; for deep (hierarchical) models

mdp = level_2_F(pE,map_mode,T);

% Known initial states
mdp.T = T;
mdp.o = [25 11 1]';                 
mdp.s = [25 d 11]';

if is_dem

    % Init DCM
    dem  = gen_dcm();
    demi = mnist_demi();

    % Known initial outcomes and priors
    o     = [round(49/2), d, 1];
    O{1}  = spm_softmax(full(sparse(round(49/2), 1, 1, 50, 1)));
    O{2}  = spm_softmax(full(sparse(1:10,    1, 1, 10, 1)));   
    O{3}  = spm_softmax(full(sparse(1, 1, 1, 3, 1)));            

    % Initialise (first glimpse)
    dem = spm_MDP_DEM(dem, demi, O, o);
    
    % Add to MDP structure
    mdp.DEM = dem;
    mdp.demi = demi;
    mdp.D{3}(1:10) = dem.X{2}(:,end);

end

% Verify model structure
MDP = spm_MDP_check(mdp);
            
end


function mdp = level_2_F(pE,map_mode,T)
% Generate discrete MDP structure 
% Inputs:
% pE         - Prior expectations about parameters
% map_mode   - one of 'itti', ,'aws' or 'gbvs'
% T          - Maximum number of saccades 

% Salience maps (see salience folder)
switch map_mode 
    case {'itti','gbvs'}
        A = load('itti_a.mat');
        A = A.itti_a;            
    case 'aws'
        A = load('aws_a.mat');
        A = A.aws_a;
end

% Initial States (D)
% -----------------------
% D{1} = Location (1:50)
% D{2} = Digit    (1:11)
% D{3} = Report   (1:11)
% D{N} = Feature  (1:N)

% Location
D{1} = zeros(49+1, 1);
D{1}(round(49/2)) = 1;

% Digit
D{2} = full(sparse(1:10,1,1,10,1));

% Report
D{3} = full(sparse(11,1,1,11,1));

% Feature
switch map_mode 
    case 'itti'
        D{4} = full(sparse(1:5, 1, 1, 5,1)); % contrast
        D{5} = full(sparse(1:5, 1, 1, 5,1)); % or 1
    case 'aws' 
        D{4} = full(sparse(1:5, 1, 1, 5,1));
    case 'gbvs'
        D{4} = full(sparse(1:5, 1, 1, 5,1));% or 3 
        D{5} = full(sparse(1:5, 1, 1, 5,1));% or 3 
end

% Hidden State dimensions Ns
Nf    = numel(D);
for f = 1:Nf
    Ns(f) = numel(D{f});
end

switch map_mode
    case 'itti'
        features = [1 1];

    case 'aws'
        features = 1;

    case 'gbvs'
        features = [1 1];
end

% Transition Matrices (B)
% -----------------------
% B{1} = Location (1:50)
% B{2} = Digit    (1:11)
% B{3} = Report   (1:11)
% B{N} = Feature  (1:N)

% controllable fixation points: move to the k-th location
%--------------------------------------------------------------------------
policy = 1;
B{1} = zeros(Ns(1), Ns(1), Ns(1));
for k = 1:Ns(1)
    B{1}(:,:,k) = 0;
    B{1}(k,:,k) = 1;
    if k < 50
        U(1,policy,:) = [k 1 11 features];
        policy = policy+1;
    end
end

% No action over digit factor
B{2} = eye(Ns(2));

% Report k-th digit
B{3} = zeros(Ns(3), Ns(3), Ns(3));
for k = 1:Ns(3)

    B{3}(k,:,k) = 1;
    B{3}(:,11,k) = 0;
    B{3}(k,11,k) = 1;
    if k < 11
        U(1,policy,:) = [50 1 k features];
        policy = policy +1;
    end
end

for f = 4:Nf
    B{f} = eye(Ns(f));
end

% Observation Matrices (A)
% -----------------------
% A{1} = Location (1:50)
% A{2} = Digit    (1:11)
% A{3} = Feedback (1:3)

switch map_mode

    case {'itti','gbvs'}
        for f2 = 1:10
            for f3 = 1:11
                for f4 = 1:5
                    for f5 = 1:5

                        if f3 == 11
                            A{3}(1,50,:, 11,f4,f5) = 1;% * exp(pE.uncertainty);
                            A{3}(2,50,f2,11,f4,f5) = 0;
                            A{3}(3,50,f2,11,f4,f5) = 0;

                        else

                            A{3}(1,50,f2,f3,f4,f5) = 0; %
                            % Right
                            A{3}(2,50,f2,f3,f4,f5) = f2 == f3; 
                            % Wrong
                            A{3}(3,50,f2,f3,f4,f5) = f2 ~= f3; 
                        end

                        % Always undecided until f1==50
                        A{3}(1,1:49,f2,f3,f4,f5) = 1;
                        A{3}(2,1:49,f2,f3,f4,f5) = 0;
                        A{3}(3,1:49,f2,f3,f4,f5) = 0;    

                    end
                end
            end
        end

    case 'aws'
        for f2 = 1:10
            for f3 = 1:11
                for f4 = 1:5
                    if f3 == 11
                        A{3}(1,50,:,11,f4)  = 1;% * exp(pE.uncertainty);
                        A{3}(2,50,f2,11,f4) = 0;
                        A{3}(3,50,f2,11,f4) = 0;

                    else

                        A{3}(1,50,f2,f3,f4) = 0; %
                        % Right
                        A{3}(2,50,f2,f3,f4) = f2 == f3; 
                        % Wrong
                        A{3}(3,50,f2,f3,f4) = f2 ~= f3; 
                    end

                    % Always undecided until f1==50
                    A{3}(1,1:49,f2,f3,f4) = 1;
                    A{3}(2,1:49,f2,f3,f4) = 0;
                    A{3}(3,1:49,f2,f3,f4) = 0;    

                end
            end
        end
end

Ng    = numel(A);
for g = 1:Ng
    No(g) = size(A{g},1);
end

% Prior Preferences
% ---------------------
% C{1} = Location (1:50)
% C{2} = What     (1:11)
% C{3} = Feedback (1:3)

% Agent wants to make a decision (location 50)
C{1}      = zeros(No(1),1);
C{1}(50,:) = .5;

% No bias toward any particular digit
C{2}      = zeros(No(2),1);
C{3}      = zeros(No(3),1);
C{3}(1,50) =  1;

% Feedback Preferences
% ---------------------
% C{3},(1,:) = Undecided;
% C{3},(2,:) = Correct;
% C{3},(3,:) = Incorrect;

% Incentivise fast responses
C{3}(1,1:T) =  0:-exp(pE.C)/(T-1):-exp(pE.C);
% Agent wants to be correct
C{3}(2,:) =  exp(pE.C);
% And not wrong
C{3}(3,:) = -4; 

%--------------------------------------------------------------------------
mdp.A     = A;                      % observation model
mdp.B     = B;                      % transition probabilities
mdp.C     = C;                      % prior preferences
mdp.D     = D;                      % prior over initial states
mdp.U     = U;                      % allowable policies
mdp.beta  = exp(pE.beta);

if isfield(pE,'alpha')
    mdp.alpha = exp(pE.alpha);
else
    % for data generation, not inversion 
    mdp.alpha = 512;
end

end

function DEM = gen_dcm()
% Generate DCM
% 
% Generative Model M
%--------------------------------------------------------------------------
%   M(i).g  = y(t)  = g(x,v,P)    {inline function, string or m-file}
%   M(i).f  = dx/dt = f(x,v,P)    {inline function, string or m-file}
%
%   M(i).pE = prior expectation of p model-parameters
%   M(i).pC = prior covariances of p model-parameters
%   M(i).v  = initial inputs
%   M(i).x  = initial states
%   M(i).a  = initial_actions

% hierarchical process G(i)
%--------------------------------------------------------------------------
%   G(i).g  = y(t)  = g(x,v,a,P)    {inline function, string or m-file}
%   G(i).f  = dx/dt = f(x,v,a,P)    {inline function, string or m-file}
%
%   G(i).pE = model-parameters
%   G(i).v  = initial inputs
%   G(i).x  = initial states
%   G(i).a  = initial_actions

% DEM parameters
N_dem = 8; nh = 10; nl = 49;
M(1).E.n = 2; M(1).E.d = 2;   

% Initial states / causes
v(1:2)   = [ 0.5 ; 0.5 ]; % Attracting coord
v(3:12)  = zeros(nh,1);   % hypothesis
x(1:2)   = [0.5;0.5];     % Initial fixation location

% Recognition Model (M)
%--------------------------------------------------------------------------
M(1).f  = @fx_dem;  % dynamics ( see bottom of this script )
M(1).g  = @gx_dem;  % predictions
M(1).x  = x;        % hidden states
M(1).pE = zeros(5, 1); % continuous parameters (rho)
M(1).pC = eye(5); % and their covariance

% level 2:
%--------------------------------------------------------------------------
M(2).v = v;  % digit & location priors

% Generative Process G (Real World)
%--------------------------------------------------------------------------
G(1).f  = @fx_adem;      % dynamics
G(1).g  = @gx_adem;      % observations
G(1).x  = [ 0.5 ; 0.5];  % hidden states

G(2).v = v(1:2);         
G(2).a = [0;0];       	 % actions

% Assemble DCM structure
%--------------------------------------------------------------------------
DEM.G  = G;
DEM.M  = M;
DEM.U = sparse(N_dem,N_dem);
DEM.C = sparse(N_dem,N_dem);
DEM.db = 1; % graphics on

end


function g = gx_dem(x, v, P) 
% Observation/Response function (Generative Model)
%--------------------------------------------------------------------------
    v = v(:); x = x(:);
    o = 1:2; hs = 3:12;
    
    % categorical latent embeddings
    hypotheses = eye(10);
    
    % categorical hypothesis
    ph = spm_softmax(v(hs));
    
    s = 0;
    for h = 1:size(hypotheses,1)
        
        % joint latent embedding
        l = spm_vec(P, hypotheses(h,:));
        
        % weighted sum of (visual) hypotheses
        fov_y = im_centred_on_x(py_decode(l),32,8,x(o));
        s = s + ph(h)*fov_y;
        
    end
    
    g   = spm_vec(x(o), s(:));
    
end

function g = gx_adem(x, v, a, P)
% Observation/Response function (Real World)
%--------------------------------------------------------------------------
    v = v(:); x = x(:);    
    % fixation location
    o = x(1:2); 
    % stimulus
    s = getStimulus(); 
    % foveated stimulus
    s = im_centred_on_x(s,32,8,o); 
    % [proprio extero]
    g   = spm_vec(o, s(:));
    
end

function f = fx_adem(x,v,a,P)
% State function (real world)
%-------------------------------
    x = x(:); a = a(:); f = x;
    f(1:2) = a - x/16; % saccade

end

function f = fx_dem(x,v,P)
% State function (generative model)
%-----------------------------------
    v = v(:); x = x(:);
    f = x; o = 1:2; h = 3:12;
    f(o) = (v(o) - x(o)) / 2; % saccade
end


function demi = mnist_demi()
% Build a set of reduced (competing) models
% \vartheta_m in the paper
%--------------------------------------------------------------------------
% Discrete to continuous grid
Xm = linspace(0,1,7);
Ym = linspace(0,1,7);   

% locs    digits   time
nl = 50 ; nh = 11; N = 8;
for i = 1:nl
    if i < 50
        [row, column] = ind2sub([7 7], i);
        ext_coord(1) = Xm(column);
        ext_coord(2) = Ym(row);
    else
        % cue location 
        ext_coord(1) = -1;
        ext_coord(2) = -1;
    end

    for j = 1:nh
        for k = 1:3

            if j == 11
                % init undecided 
                hyp = full(sparse(1:10, 1, 1, 10, 1));
            else
                hyp = full(sparse(j,1,1,10,1));
            end

            % Map between causes & observations
            demi.U{i,j,k} = repmat([ext_coord'; hyp]',N,1)'; % M
            demi.C{i,j,k} = repmat(ext_coord,N,1)'; % G    
        end
    end
end

end


