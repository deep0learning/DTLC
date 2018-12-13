clear all;

% Set algorithm parameters
options.k = 100;
options.alpha = 1;       % ATLDA alpha
options.beta = 0.1;      % ATLDA beta
options.eta= 1;          % ATLDA eta
options.ker = 'primal';  % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;     % kernel bandwidth: rbf only
options.non = 1;         % 
T = 10;

srcStr = {'PIE05','PIE05','PIE05','PIE05','PIE07','PIE07','PIE07','PIE07','PIE09','PIE09','PIE09','PIE09','PIE27','PIE27','PIE27','PIE27','PIE29','PIE29','PIE29','PIE29'};
tgtStr = {'PIE07','PIE09','PIE27','PIE29','PIE05','PIE09','PIE27','PIE29','PIE05','PIE07','PIE27','PIE29','PIE05','PIE07','PIE09','PIE29','PIE05','PIE07','PIE09','PIE27'};
    result = [];
for iData = 1:20
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
    options.data = strcat(src,'_vs_',tgt);
    
    %% Preprocess data 
    load(strcat('../data/',src));
    Xs = fea';
    meanXs = mean(Xs, 2);
    Xs = bsxfun(@minus, Xs, meanXs);
    Xs = bsxfun(@times, Xs, 1./max(1e-12, sqrt(sum(Xs.^2))));
    Ys = gnd;
    load(strcat('../data/',tgt));
    Xt = fea';
    meanXt = mean(Xt, 2);
    Xt = bsxfun(@minus, Xt, meanXt);
    Xt = bsxfun(@times, Xt, 1./max(1e-12, sqrt(sum(Xt.^2))));
    Yt = gnd;
    
    %% 1NN evaluation
    Cls = knnclassify(Xt',Xs',Ys,1);
    acc = length(find(Cls==Yt))/length(Yt); fprintf('NN=%0.4f\n',acc);

    %% ATLDA evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        %% ATLDA transfer learning
        [Z,A] = ATLDA_TL(Xs,Xt,Ys,Cls,options);
        
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        
        Cls = knnclassify(Zt',Zs',Ys,1);
        
        %% ATLDA label consistency
        options.NN=5;
        [label_t,predict_t,~]=ATLDA_LC(Zs',Ys,Zt',Yt,options,Cls);
        Cls=predict_t; 
        
        acc = length(find(Cls==Yt))/length(Yt); 
        fprintf('ATLDA+NN=%0.4f\n',acc);
        Acc = [Acc;acc(1)];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
end
result_aver=mean(result);
Result=[result;result_aver]*100