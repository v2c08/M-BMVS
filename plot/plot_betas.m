% Plot behavioural metrics as a function of beta
% Figure 5A-C
% Run gen_beta first

% Get human behavioural metrics
load('figure_data_humans.mat');
task3data = figure_vars.Task3;
subjs = fieldnames(task3data);

p_durations = [];
p_accuracy  = [];
p_saccades  = [];

for i = 1:numel(subjs)
   trialdata = task3data.(subjs{i});
   trials = fieldnames(trialdata);
   for j=1:numel(trials)
      
      % Accuracy
      if strcmp(trialdata.(trials{j}).result,'correct')
        p_accuracy = [p_accuracy 1];
      else
        p_accuracy = [p_accuracy 0];
      end
      
      % 4 is # of irrelevent subfileds,2 precludes start and decision
      sacs = numel(fieldnames(trialdata.(trials{j}).saccades))-4-2;
      p_saccades = [p_saccades sacs];
      
      % Reaction Times
      fixationdata = trialdata.(trials{j}).fixations;
      fixations = fieldnames(trialdata.(trials{j}).fixations);
      for k = 1:numel(fixations)-3
         p_durations = [p_durations fixationdata.(fixations{k}).duration];
      end
   end
end


folder = fullfile('figures','beta','data');
data = cell(100,9);
for s = 1:size(data,1)
    for b = 1:size(data,2)
        
        load(fullfile(folder,strcat('s',string(s),'b',string(b))))
        data{s,b} = MDPo;
        clear MDPo
        
    end 
end

beta = (0:8)/8;
beta = 1./(2*beta + 1/8);

for b = 1:size(data,2)
    for m = 1:size(data,1)
        MDP = data{m,b};
        accuracy(b,m) = double(MDP.o(3,end) == 2);  % accuracy
        nsac(b,m) = numel(MDP.o(1,:));            % number of saccades
        rt(b,m) = mean(MDP.rt);                     % reaction time
        
    end
end

EP1 = mean(accuracy,2);
EP2 = mean(nsac,2);
EP3 = mean(rt,2);

for i = 2:size(pixcoverage,1)
    min_err(i) = min(nonzeros(pixcoverage(i,:)));
    max_err(i) = max(pixcoverage(i,:));
    pxc(i) = mean(nonzeros(pixcoverage(i,:)));
end

pxc = flip(pxc(2:end));
min_err = flip(min_err(2:end));
max_err = flip(max_err(2:end));

PEB = 1.2;

hold on
subplot(1,3,1), bar(EP1*100,'k')
xlabel('$\frac{1}{2\beta+\frac{1}{8}}$','Interpreter','latex')
ylabel('%')
title('Accuracy','fontweight','bold','Fontsize',12), axis square
yline(100*(nnz(p_accuracy)/numel(p_accuracy)),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)
% axis([0.125 2 -0.1 1])
hold on
subplot(1,3,2), bar(EP2,'k')
xlabel('$\frac{1}{2\beta+\frac{1}{8}}$','Interpreter','latex')
ylabel('Seconds')
title('Decision time','fontweight','bold','Fontsize',12), axis square
yline(mean(p_saccades),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)
ylim([2,7])
% axis([0.125 2 1 7])
hold on
subplot(1,3,3), bar(EP3/10,'k')
xlabel('$\frac{1}{2\beta+\frac{1}{8}}$','Interpreter','latex')
ylabel('Milliseconds')
title('Reaction Time','fontweight','bold','Fontsize',12), axis square
yline(mean(p_durations),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)

