clc
clear
warning off
addpath(genpath('./'))
data = {'MSRC'};
orders = [1];
k = 5;
for idx = 1: length(data)
    load(data{idx})
    for j=1:length(X)
        colnum=size(X{j},2);
        mole = repmat(std(X{j},0,2),1,colnum);
        mole(mole==0) = 1;
        X{j}=(X{j}-repmat(mean(X{j},2),1,colnum))./mole;
    end
    c = length(unique(Y));
    [G, K, F, evs]=prepare(X, k, c);
    KF = cross_con(K, F, orders(idx));
    [Y_g, y0, it, obj] = main_max(KF, G, c, c, 40, eps - evs);
    result(idx, :) = ClusteringMeasure_new(Y, Y_g);
end
