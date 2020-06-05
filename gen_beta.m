% Output Dir
fdir=fullfile('figures','beta','data');

% Model Path
dec = struct(py.load_model.get_model('mnist_vae.pth'));
setDecoder(dec.model);

% VAE Path
mnist_paths = load('participant_mnist_paths.mat');
mnist_paths = mnist_paths.paths;

% Max trials
T = 8; 

pE.C    = log(3); 

beta = 1./(2*((0:8)/8) + 1/8);
beta = flip([1./(2*((0:8)/8) + 1/8)]);

for s = 1:100
    
    % Set stimulus 
    d = randi(10);  
    digit_paths = mnist_paths{d};
    idx=randi(numel(digit_paths));
    digpath = fullfile('mnist_png', 'testing', digit_paths(idx));  
    setStimulus(digpath);
    
    % generate MDP
    MDP = mnist_MDP_gen(pE, 'aws', T, d,true);
    MDP.alpha = 2;
    
    for b = 1:length(beta)

        MDP.beta = beta(b);
        
        % Initial states
        MDP.s(1) = 25;
        MDP.s(2) = d;
        
        % Generate data
        MDPo = spm_MDP_VB_X(MDP);
        fname = strcat('s',num2str(s),'b',num2str(b),'.mat');
        save(fullfile(fdir,fname), 'MDPo');
        clear MDPo
        
    end
end

