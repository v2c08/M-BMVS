% Plot behavioural metrics as a function of C
% Figure 5E-F
% Run gen_C first

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

folder = fullfile('figures','utility','test');
data = cell(12,9);
for s = 1:size(data,1)
    for c = 1:size(data,2)
        
        fname = strcat('s',string(s),'c',string(c));
        load(fullfile(folder,fname))
        data{s,c} = MDPo;
        clear MDPo
        
    end 
end

C = 0:8;

for s = 1:size(data,1)
    for c = 1:size(data,2)
        MDP = data{s,c};
        accuracy(c,s) = double(MDP.o(3,end) == 2);  % accuracy
        nsac(c,s) = numel(MDP.o(1,:)) - 2;            % number of saccades
        rt(c,s) = mean(MDP.rt)/size(MDP.o,1);                     % reaction time
    end
end

PEB = 3.0478;

EP1 = mean(accuracy,2);
EP2 = mean(nsac,2);
EP3 = mean(rt,2);

hold on
subplot(1,3,1), bar(C,EP1*100,'k')
xlabel('Prior Preference'),ylabel('Classificaiton Accuracy')
title('Accuracy','fontweight','bold','Fontsize',12), axis square
yline(100*nnz(p_accuracy)/numel(p_accuracy),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)
xlim([-0.5,8.5])

hold on
subplot(1,3,2), bar(C,EP2,'k')
xlabel('Prior Preference'),ylabel('Number of Saccades')
title('Decision time','Fontsize',10), axis square
yline(mean(p_saccades),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)
xlim([-0.5,8.5])
% axis([0.125 2 1 7])
hold on
subplot(1,3,3), bar(C,EP3/10,'k')
xlabel('Prior Preference'),ylabel('Seconds')
title('Reaction Time','Fontsize',10), axis square
yline(mean(p_durations),'-r','LineWidth',2)
xline(PEB,'-b','LineWidth',2)
xlim([-0.5,8.5])


hold on    

