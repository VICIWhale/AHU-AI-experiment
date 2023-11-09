clc;
clear;
load("dataFolds.mat");

%% 十交叉法
% 选择训练集和测试集
CNT = 0;
RESULT = 0;
for i = 1:10
        testSet = dataFolds{i};
    
        trainSet = [];
        for j = 1:10
            if j ~= i
                trainSet = [trainSet;dataFolds{j}];
            end
        end
    
    trainLabel = trainSet(:,1);
    trainFeatures = trainSet(:,2:end);
    testLabel = testSet(:,1);
    testFeatures = testSet(:,2:end);
    
    Aidx = find(trainLabel == -1);
    A = trainFeatures(Aidx,:);
    
    Bidx = find(trainLabel == 0);
    B = trainFeatures(Bidx,:);
    
    Cidx = find(trainLabel == 1);
    C = trainFeatures(Cidx,:);
    
    % 训练过程
    numTrain = num * 0.9;
    Flag = zeros(numTrain,1); % 标记数组 1：已经被覆盖；0：仍未被覆盖
    cluster = [];
    clustersample = cell(num,1);
    cnt = 1;
    while true
        % 选择一个仍未学习的样本
        uncovered = find(Flag == 0);
        if isempty(uncovered)
            break;
        end

        randIdx = uncovered(randi(numel(uncovered)));
        Flag(randIdx) = 1; % 将被选中的学习样本标记为一
    
        SelectedSample = trainFeatures(randIdx,:);
        SelectedSampleLabel = trainLabel(randIdx);
    
        % 选择同类和异类  (这里写的有点臃肿 可以调节）
        if(SelectedSampleLabel == -1)
            same = A;
            sameidx = Aidx;
            diff = [B;C];
            diffidx = [Bidx;Cidx];
        elseif(SelectedSampleLabel == 0)
            same = B;
            sameidx = Bidx;
            diff = [A;C];
            diffidx = [Aidx;Cidx];
        elseif(SelectedSampleLabel == 1)
            same = C;
            sameidx = Cidx;
            diff = [A;B];
            diffidx = [Aidx;Bidx];
        end
        
        %unmarkedIdx = ~Flag(diffidx);
        %diff = diff(unmarkedIdx,:);
        %diffidx = diffidx(unmarkedIdx);
    
        unmarkedIdx = ~Flag(sameidx);
        same = same(unmarkedIdx,:);
        sameidx = sameidx(unmarkedIdx);
    
        innerdiff = diff * SelectedSample';
        innersame = same * SelectedSample';
        
        [d1,d1idx] = max(innerdiff);  % 最大内积->最小距离 距离所有不同点的最小距离
    
        if(~isempty(find(innersame > d1, 1)))
            tempidx = find(innersame > d1);
            sameidx = sameidx(tempidx);
            temp = innersame(tempidx);
    
            [d2,d2idx] = min(temp);
        
            count = numel(sameidx) + 1;
            Flag(sameidx) = Flag(sameidx) +  1;
        else
            %continue;   % 删除点
            d2 = SelectedSample * SelectedSample';
            count = 1;

        end
    
        result = [SelectedSample d2 SelectedSampleLabel count];
        cluster = [cluster;result];

        clustersample{cnt} = [sameidx;temp];
        cnt = cnt + 1;
    end
    CNT = CNT + cnt - 1;

    % 测试过程
    testnum = numel(testLabel);
    clusterNum = size(cluster,1);
    clusterFeatures = cluster(:,1:5);
    clusterDist = cluster(:,6);
    clusterLabel = cluster(:,7);
    right = 0; % 记录验证分类正确的样本个数
    for j = 1:testnum
        Label = testLabel(j);
        Feature = testFeatures(j,:);
        % 计算测试点距离每一个覆盖中心的内积
        testDist = clusterFeatures * Feature';
        % 比较距离情况
        less = find(testDist >= clusterDist);
        bigger = find(testDist < clusterDist);

        if(size(less,1) == 1) % 该点落在了一个覆盖内
            lessIdx = less(1,1);
            if(Label == clusterLabel(lessIdx))
                right = right + 1;
            end
        else % 该点落在了边界域（所有覆盖之外）
            % 采用距离中心最近原则
            [~,lessIdx] = max(testDist);
            if(Label == clusterLabel(lessIdx))
                right = right + 1;
            end
        end
    end
    result = right / testnum;
    RESULT = RESULT + result;
end


fprintf('平均覆盖数：%.2f\n',CNT / 10);
fprintf('评价正确率：%.2f\n',RESULT / 10);











