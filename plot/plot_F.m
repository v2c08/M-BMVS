% Plot Free Energy as a function of parameters
% Generates Figure 5H

folder = fullfile('figures','beta','data');
datab = [];
for s = 1:100
    for b = 1:9
        
        load(fullfile(folder,strcat('s',string(s),'b',string(b))))
        [F,Fu,Fs,Fq,Fg,Fa] = spm_MDP_F(MDPo);

        datab(b,s) = Fu(end);
        clear MDPo        
    end 
end

folder = fullfile('figures','utility','test');
datac = [];
for s = 1:100
    for c = 1:9
        load(fullfile(folder,strcat('s',string(s),'c',string(c))))
        
        [F,Fu,Fs,Fq,Fg,Fa] = spm_MDP_F(MDPo);
        datac(b,s) = Fu(end);
        clear MDPo
    end 
end


for i = 1:9
    for j = 1:9
        F(i,j) = sum(datab(i,:))+sum(datac(j,:));
    end
end


beta = flip([1./(2*((0:8)/8) + 1/8)]);
C = 0:8;

clear yticklabels
[Y,X] = meshgrid(beta, C);
[M,c] = contourf(Y,X,F);
xlabel('C')
ylabel('Beta')
c.LineWidth = 0.5;
yticks([0:8])
labels = {0.4, 0.5, 0.6, 0.7, 0.9, 1.1, 1.6,2.6, 8}
yticklabels(labels)
xticks([0:8])
xticklabels(8:-1:0)
colormap(bone)

contourcmap('jet',10,'Colorbar','on','Location','horizontal')
