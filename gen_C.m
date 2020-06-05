% output dir
fdir=fullfile('figures','utility','test');

% stimulus paths
mnist_paths = load('participant_mnist_paths.mat');
mnist_paths = mnist_paths.paths;

% vae path
dec = struct(py.load_model.get_model('mnist_vae.pth'));
setDecoder(dec.model);

T = 8; 
digit_paths = mnist_paths{d};
pE.beta = log(0.4);

utilities  = 0:8;

for s = 1:100
        
    for c = 1:length(utilities)
        
        % Set stimulus
        d = randi(10);
        digit_paths = mnist_paths{d};
        idx=randi(numel(digit_paths));
        digpath = fullfile('mnist_png', 'testing', digit_paths(idx));  
        setStimulus(digpath);
        
        MDP = mnist_MDP_gen(pE, 'aws', T, d,true);
    %     MDP.tau = 2;
        MDP.alpha = 512;
        MDP.C{3}(2,:) = utilities(c)*2;
        MDP.s(1) = 25;
        MDP.s(2) = d;
        
        MDPo = spm_MDP_VB_X(MDP);
        fname = strcat('s',num2str(s),'c',num2str(c),'.mat');
        save(fullfile(fdir,fname), 'MDPo');
        clear MDPo

    end 
end
