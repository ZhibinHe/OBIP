clear
clc

addpath /data/hzb/matlab_path/npy-matlab-master/npy-matlab/

t=1;
for i=100:199
    i
    label_score = readNPY(['label_score_epoch_' num2str(i) '.npy']);
    sys_score = readNPY(['sys_score_epoch_' num2str(i) '.npy']);
    
    
    label_score(1,:) = [];
    sys_score(1,:) = [];
    
    
    
    sys_score(isnan(label_score)) = 0;
    label_score(isnan(label_score)) = 0;
    
    
    
    
    corr_list = zeros(size(label_score,2),1);
    for i=1:size(label_score,2)
        corr_list(i,1) = corr(sys_score(:,i), label_score(:,i));
    end
    
    
    


     for i=1:size(label_score,2)
         if corr_list(i,1)<0
         
        sys_score(:,i) = -sys_score(:,i);
         end
    end
    
    
    cod_list = zeros(size(label_score,2),1);
    
    for i=1:size(label_score,2)
        cod_list(i,1) = 1-sum((label_score(:,i)- sys_score(:,i)).^2) / sum((label_score(:,i)).^2);
    end
    
%    save('corr_cod_list.mat','cod_list','corr_list');
    
    alllist(t,1) = mean(abs(corr_list));
    alllist(t,2) =  mean(cod_list);
    t=t+1;
end










%%%%%%%%%%%%%%%%%%%%%%%%%%%

   i = 75+100-1
    label_score = readNPY(['label_score_epoch_' num2str(i) '.npy']);
    sys_score = readNPY(['sys_score_epoch_' num2str(i) '.npy']);
    
    
    label_score(1,:) = [];
    sys_score(1,:) = [];
    
    
    
    sys_score(isnan(label_score)) = 0;
    label_score(isnan(label_score)) = 0;
    
    
    
    
    corr_list = zeros(size(label_score,2),1);
    for i=1:size(label_score,2)
        corr_list(i,1) = corr(sys_score(:,i), label_score(:,i));
    end
    
    
    


     for i=1:size(label_score,2)
         if corr_list(i,1)<0
         
        sys_score(:,i) = -sys_score(:,i);
         end
    end
    
    
    cod_list = zeros(size(label_score,2),1);
    
    for i=1:size(label_score,2)
        cod_list(i,1) = 1-sum((label_score(:,i)- sys_score(:,i)).^2) / sum((label_score(:,i)).^2);
    end
    
%    save('corr_cod_list.mat','cod_list','corr_list');

mean(abs(corr_list))
mean((cod_list))
