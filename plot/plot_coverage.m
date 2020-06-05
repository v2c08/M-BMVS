% Plot attended pixels as a function of saccades
% Generates Figure 5D

folder = fullfile('figures','beta','data');
data = cell(0);
for s = 1:100
    for b = 1:9
        
        load(fullfile(folder,strcat('s',string(s),'b',string(b))))
        data{end+1} = MDPo;
        clear MDPo        
    end 
end

folder = fullfile('figures','utility','test');
for s = 1:100
    for c = 1:9
        load(fullfile(folder,strcat('s',string(s),'c',string(c))))
        data{end+1} = MDPo;
        clear MDPo
    end 
end

pixcoverage = []

for m = 1:length(data)
    MDP = data{m};
    pix = [];
    for i = 1:length(MDP.dem)-1
        x = MDP.dem(i).qU.x{1};
        for j = 1:size(x,2)
            im_sq = coverage_from_x(32,8,x);
            pix = [pix im_sq(:)'];  
        end
    end
    n_saccades = size(MDP.o,2)-1;
    pixcoverage(n_saccades,end+1) = (numel(unique(pix))/(28*28)) * 100;
end

for i = 1:size(pixcoverage,1)
    min_err(i) = min(nonzeros(pixcoverage(i,:)));
    max_err(i) = max(pixcoverage(i,:));
    pxc(i) = mean(nonzeros(pixcoverage(i,:)));
end

min_err = flip(min_err(2:end));
max_err = flip(max_err(2:end));

bar(pxc(2:end))
