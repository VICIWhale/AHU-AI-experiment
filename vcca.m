clc;
clear;

load("iris.mat");
%load("Zoo.mat");
%load("waveform.mat");
%load("wine.mat");
%load("waveform1.mat");
%load("bupa.mat");
%load("bupa1.mat");
%load("haberman.mat");

data = DATA;
Labels = data(:,1);
Feature = data(:,2:end);

% add vcca
K = 15;

LabelsForvaac = unique(Labels);


%% 划分数据集
tt = 10; % t交叉法
c = cvpartition(Labels, 'KFold', tt, 'Stratify', true);
dataFold = cell(1,c.NumTestSets);
for i = 1:c.NumTestSets
    testIdx = test(c,i);
    dataFold{i} = data(testIdx,:);
end

[warnMsg, warnId] = lastwarn;  % 获取最后的警告信息和ID
warning('off',warnId);         % 关闭此ID的警告

CNT = 0;
RIGHT = 0;
% YYSelect = 0;YY = 0;YYRight = 0;
% NNSelect = 0;NN = 0;NNRight = 0;

% 选择训练集和测试集
for i = 1:tt
    testSet = dataFold{i}; % 该轮测试集

    trainSet = [];
    for j = 1:tt
        if j ~= i
            trainSet = [trainSet;dataFold{j}];
        end
    end

    trainLabel = trainSet(:,1);
    trainFeature = trainSet(:,2:end);
    testLabel = testSet(:,1);
    testFeature = testSet(:,2:end);

    
    

% % %     训练集中 所有相同标签聚集
% % %     separatedData{1} 包含了标签为 uniqueLabels(1)
% % %     uniqueLabel = unique(trainLabel);
% % %     separatedFeature = cell(numofLable,1);
% % %     separatedIdx = cell(numofLable,1);
% % %     for j = 1:numofLable
% % %         label = uniqueLabel(j);
% % %         labelindices = trainLabel == label;
% % %         separatedIdx{j} = labelindices;
% % %         separatedFeature{j} = trainFeature(labelindices);
% % %     end

    % 训练过程
    CclusterForvacc = cell(K,1);
    for k = 1:K
        numTrain = size(trainSet,1);
        Flag = zeros(numTrain,1);
        cluster = [];
        clusterSamples = cell(num,1);
    
        % yyselect = 0;yy = 0;
        % nnselect = 0;nn = 0;
    
        while true
            % 随机选择一个仍为被覆盖的样本
            uncovered = find(Flag == 0);
    
            if isempty(uncovered) % exit condition
                break;
            end
    
            randIdx = uncovered(randi(numel(uncovered)));
            Flag(randIdx) = 1;
            SelectedFeature = trainFeature(randIdx,:);
            SelectedLabel = trainLabel(randIdx,:);
    
    
            % 获得同类和异类
            sameIdx = find(trainLabel == SelectedLabel);
            sameFeature = trainFeature(sameIdx,:);
            diffIdx = find(trainLabel ~= SelectedLabel);
            diffFeature = trainFeature(diffIdx,:);
    
            % 同类集合 去除已经被覆盖样本
            unmarkedIdx = ~Flag(sameIdx);
            sameFeature = sameFeature(unmarkedIdx,:);
            sameidx = sameIdx(unmarkedIdx,:);

            % 将已经被学习的样本并入异类样本
            covered = find(Flag == 1);
            coveredFeature = trainFeature(covered,:);
            diffIdx = [diffIdx;covered];
            diffFeature = [diffFeature;coveredFeature];
    
    
            % 对 异类 内积
            innerdiff = diffFeature * SelectedFeature';
            % 对 同类 内积
            innersame = sameFeature * SelectedFeature';
            % 得到 异类 最大内积 -> 最小距离
            [d1,~] = max(innerdiff);
    
            if(~isempty(find(innersame > d1,1)))
                % 存在 d2
                tempidx = find(innersame > d1);
                sameidx = sameidx(tempidx);
                temp = innersame(tempidx);
    
                [d2,~] = min(temp);
                count = numel(sameidx) + 1;
                Flag(sameidx) = Flag(sameidx) + 1; % 这些样本已经被覆盖
            else
                % 不存在 d2 周围都是异类点
                % continue; % 删除对应点
                d2 = SelectedFeature * SelectedFeature';
                count = 1;
            end
    
            result = [SelectedFeature d2 SelectedLabel count];
            cluster = [cluster;result];
%             cnt = cnt + 1;
        end
        CclusterForvacc{k} = cluster;
    end

    testnum = numel(testLabel);
    right = 0;

    for j = 1:testnum
        S = zeros(numofLabel,1);
        Label = testLabel(j);
        Feature = testFeature(j,:);

        for k = 1:K
            Ccluster = CclusterForvacc{k};
            clusterFeatures = Ccluster(:,1:numofFeature);
            clusterDist = Ccluster(:,numofFeature + 1);
            clusterLabel = Ccluster(:,numofFeature + 2);

            testDist = clusterFeatures * Feature';
            less = find(testDist >= clusterDist);
            bigger = find(testDist < clusterDist);

            if(size(less,1) == 1)
                lessIdx = less(1,1);
            else
                [~,lessIdx] = max(testDist);
            end

            KLabel = clusterLabel(lessIdx);
            Ccidx = find(LabelsForvaac == KLabel);
            S(Ccidx) = S(Ccidx) + 1; 
        end

         [~,yuce] = max(S);
         VoteLabel = LabelsForvaac(yuce);
         if(Label == VoteLabel)
             right = right + 1;
         end
    end

    result = right / testnum;
    RIGHT = RIGHT + result;
end

fprintf('平均正确率：%.2f\n',RIGHT / tt);

   


