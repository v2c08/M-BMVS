% Generates Salience maps with Itti, AWS & GBVS
% -----------------------------------------------
% The A matrices produced by this script define task outcomes 
% for every possible combination of hidden states

% Note simsal, aws_sal and GBVS hcan be downloaded from:
% http://www.vision.caltech.edu/~harel/share/gbvs.php a
% http://persoal.citius.usc.es/xose.vidal/research/aws/AWSmodel.html. 

digits_path = fullfile('mnist_png', 'training');
h_zero = [1 2 3 4 5 6 7 8 9 0];

samples = 1000;
subdivs = 5;
digits = 10;

% Observation Matrices (A)
% -----------------------
% A{1} = Location (1:50)
% A{2} = Digit    (1:11)
% A{3} = Feedback (1:3)

% Initialisation
%------------------------
% the following 3 loops instantiate the 
% A matrices, ensuring no all-zero columns,
% and map hidden states directly to outcomes 
% for the 'digit' and 'location' factors

% Digit modality  d   l   d   h   f  f
itti_a{2} = zeros(10, 50, 10, 11, 5, 5);
aws_a{2}  = zeros(10, 50, 10, 11, 5);
gbvs_a{2} = zeros(10, 50, 10, 11, 5, 5);

% Itti
for i = 1:11 % Report
    for j = 1:50 % location
        for p = 1:10 % digit
            for k = 1:5 % feature
                for l = 1:5 %feature
                    itti_a{1}(j,j,p,i,k,l) = 1;
                    itti_a{2}(i,j,p,i,k,l) = 1;
                end
            end
        end
    end
end

% AWS 
for i = 1:11
    for l = 1:10
        for j = 1:50
            for k = 1:5
                aws_a{1}(j,j,l,i,k) = 1;
                aws_a{2}(i,j,l,i,k) = 1;
            end
        end
    end
end

% GBVS
for i = 1:11
    for j = 1:50
        for m = 1:10
            for k = 1:5 
                for l = 1:5

                gbvs_a{1}(j,j,m,i,k,l) = 1;
                gbvs_a{2}(i,j,m,i,k,l) = 1;
            
                end
            end
        end
    end
end

% Discrete feature thresholds as %
subdiv_edges{2} = [0, 0.5, 1];
subdiv_edges{3} = [0, 0.33, 0.66, 1];
subdiv_edges{4} = [0, 0.25, 0.50, 0.75, 1];
subdiv_edges{5} = [0, 0.2, 0.4, 0.6, 0.8, 1];

for i = 1:samples
    for digit = 1:11
        % Uncertain features under uncertain hypothesis
        if digit == 11
            itti_a{2}(digit, n, 1:10, digit, :, :) = 0.2;
            aws_a{2}(digit,  n, 1:10, digit, :) = 0.2;
            gbvs_a{2}(digit, n, 1:10, digit, :, :) = 0.2;
            
        else
            
            % load digit
            digitsinf = dir([digits_path '//', num2str(h_zero(digit)), '//*.png']);
            idx = randi(numel(digitsinf));
            folder = fullfile([digits_path '//', num2str(h_zero(digit))]);
            digpath = fullfile(folder, digitsinf(idx).name);
            img = imread(digpath);

            % Calc Itti Map
            p = default_fast_param;
            p.blurRadius = 0.02;     % e.g. we can change blur radius 
            p.channels = 'IO';
            p.mapWidth = 7;
            p.nGaborAngles = 4;
            param.centerbias = 1;
            [map,chanmaps,maps,chans]  = simpsal(img,p);
            
            % Discretise Itti map
            c  = discretize(maps{1}{1}, subdiv_edges{5});
            o1 = discretize(maps{2}{1}, subdiv_edges{5});
            o2 = discretize(maps{2}{2}, subdiv_edges{5});
            o3 = discretize(maps{2}{3}, subdiv_edges{5});
            o4 = discretize(maps{2}{4}, subdiv_edges{5});

            % Calc & Discretise AWS map
            aws_sal = imresize(aws(img, 4), 0.12);
            aws_sal = (aws_sal - min(min(aws_sal)))/(max(max(aws_sal)) - min(min(aws_sal)));
            aws_sal = discretize(aws_sal,subdiv_edges{5});

            %  Calc & Discretise GBVS map
            p  = makeGBVSParams();
            p.salmapmaxsize = 7;
            p.channels = 'IO';
            p.salmapsize = [7 7];
            gbvs_salience = gbvs(imresize(img, 5), p);
            gbvs_int = discretize(gbvs_salience.top_level_feat_maps{1}, subdiv_edges{5});
            gbvs_ori = discretize(gbvs_salience.top_level_feat_maps{2}, subdiv_edges{5});
            
            % Embed feature probabilities in A matrices
            for n = 1:numel(o1)

                itti_o = mode([o1(n), o2(n), o3(n), o4(n)]);
                itti_a{2}(digit, n, digit, 1:10, c(n), itti_o) = itti_a{2}(digit, n, digit, 1:10,c(n), itti_o) + 1;
                aws_a{2}(digit,  n, digit, 1:10 , aws_sal(n))  = aws_a{2}(digit,n, digit, 1:10,aws_sal(n)) + 1;
                gbvs_a{2}(digit, n, digit, 1:10, gbvs_int(n), gbvs_ori(n)) = gbvs_a{2}(digit, n, digit, 1:10,gbvs_int(n), gbvs_ori(n)) + 1;

            end
        end
    end
end

save('itti_a.mat', 'itti_a')
save('aws_a.mat', 'aws_a')
save('gbvs_a.mat', 'gbvs_a')

