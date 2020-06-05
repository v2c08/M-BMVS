% Fit parameters to human behaviour

map = [10 1 2 3 4 5 6 7 8 9];

% Initialise Decoder p_theta
decoder = struct(py.load_model.get_model('mnist_vae.pth'));
setDecoder(decoder.model);

% Load participant data 
subs = load('figure_data_humans.mat');
subs = subs.figure_vars.Task3;
subjids = fieldnames(subs);

for sub = 1:numel(subjids)
      

    sub_data = subs.(subjids{sub});
    trialids = fieldnames(sub_data);
    trialidx = 1;  
    for trial = 1:numel(trialids)
        sub_trial = sub_data.(trialids{trial});
        
        % Task behavioural metrics
        [acc, num_s, rt] = get_metrics(sub_trial);
        
        % Observed behaviour to MDP outcomes
        [o, u] = fill_fields(sub_trial);

        % Filter subset of valid trials
        if all(o(:)) && all(u(:)) 

            subject{sub}.MDP(trialidx).T = 8;
            subject{sub}.MDP(trialidx).o = o;
            subject{sub}.MDP(trialidx).u = u;
            sub_trial.digit_file(end-2:end) = 'png';
            subject{sub}.MDP(trialidx).path = fullfile('mnist_png','testing',num2str(sub_trial.true),sub_trial.digit_file); 
            subject{sub}.MDP(trialidx).digit = sub_trial.true;
            subject{sub}.MDP(trialidx).accuracy = acc;
            subject{sub}.MDP(trialidx).num_saccades = num_s;
            subject{sub}.MDP(trialidx).mean_rt = rt;
            trialidx  = trialidx+1;

        end
    end
end

% Specify fields to be estimates
pE.C     = log(1); % log preferences
pE.beta  = log(1); % log beta

map_modes = {'itti','aws','gbvs'};
state_dim = {5, 4, 5};

% specify model 
%--------------------------------------------------------------------------
DCM.M.G   = @mnist_MDP_gen;           % parameterisation of MDP (see below)
DCM.M.L   = @mnist_log_likelihood;             % log-likelihood function
DCM.M.pE  = pE;                     % prior expectations
DCM.M.pC  = eye(numel(fieldnames(pE))) * 1/16;                     % prior expectations

for s = 1:numel(subject)

    DCM.U       = {subject{s}.MDP(:).o};    % trial outcomes
    DCM.Y       = {subject{s}.MDP(:).u};    % responses (action)
    DCM.M.STIM  = {subject{s}.MDP(:).path}; % trial stimulus  
    DCM.M.T     = {subject{s}.MDP(:).T};    % trial duration
    DCM.M.digit = {subject{s}.MDP(:).digit}; 
    DCM.M.map   = 'aws';

    % Action space (no control over features - always singleton)
    for y = 1: numel(DCM.Y)
        DCM.Y{y}(4:state_dim{2},:) = 1;
    end    

    % Model inversion with Variational Laplace
    %--------------------------------------------------------------------------
    [Ep,Cp,F] = spm_nlsi_Newton(DCM.M,DCM.U,DCM.Y);

    % Store posterior densities and log evidnce (free energy)
    %--------------------------------------------------------------------------
    DCM.F   = F;
    DCM.Ep  = Ep;
    DCM.Cp  = Cp;
    DCM.M   = DCM.M;

    DCM.accuracy     = subject{s}.MDP(:).accuracy;
    DCM.num_saccades = subject{s}.MDP(:).num_saccades;
    DCM.mean_rt      = subject{s}.MDP(:).mean_rt;

    foldername = fullfile('figures','inversion','no_alpha');
    fname = strcat(subjids{s},'_t','.mat');
    save(fullfile(foldername,fname),'DCM');

end

function [acc,num_s,rt] = get_metrics(trial)

      % Accuracy
      if strcmp(trial.result,'correct'), acc=1; else acc=0; end
      
      % 4 is # of irrelevent subfileds,2 precludes start and decision
      num_s = numel(fieldnames(trial.saccades))-4-2;
      
      % Reaction Times
      fixationdata = trial.fixations;
      fixations = fieldnames(trial.fixations);
      rt = [];
      for k = 1:numel(fixations)-3
         rt = [rt fixationdata.(fixations{k}).duration];
      end
        rt = mean(rt);
end


function [o, u] = fill_fields(sub_trial)

    % Get subject response
    digit           = sub_trial.true;
    digit(digit==0) = 10;

    grid_locs = unique(sub_trial.saccades.grid_locations, 'stable');
    result    = sub_trial.result;
    pred      = sub_trial.predicted;

    % speficy trial duration 
    T = length(grid_locs);

    % specify initial & known outcomes
    o(1,:)       = grid_locs;
    o(1,end)     = 50;
    o(2,:)       = 11;
    o(2,end)     = pred;
    o(3,:)       = ones(T,1);

    switch result
        case 'correct'
            o(3,end) = 2;
        case 'incorrect'
            o(3,end) = 3;
        otherwise 
            o(3,end) = 1;
    end

    % specify known actions
    u(1,:)   = grid_locs(2:end);
    u(1,end) = 50;  % cue location
    u(2,:)   = 1;   % no aciton
    u(3,:)   = 11;  % undecided
    u(3,end) = pred;% decision
    u(4,:)   = 1;   % no action
end