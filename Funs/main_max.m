%% 将卷积核与重构图分开学习
function [Y_graph, y0, it, obj] = main_max(KF, Graph,c, dim_W, niter, evs)
num = size(KF{1}, 1);
m = size(KF, 1);%number of fea
n = length(Graph);%number of graphs
dim_F = size(KF{1}, 2);
%% init
Y = full(Init_Y(Graph, c));
[~, y0] = max(Y, [], 2);
W = cell(m, m);
for i = 1 : m
    for j = 1 : m
        W{i,j} = eye(dim_F, dim_W) / m;
    end
end
a = ones(n + 1) / sqrt(n + 1);
p = ones(m, m) ./ sqrt(m * m);
YYYY = (Y * (Y' * Y)^(-1)) * Y';
p_flatten = reshape(p, [1, m * m]);
Q = zeros(num);
for i = 1 : n
    Q = Q + Graph{i} * a(i);
end
Q = Q + YYYY * a(end);
G = zeros(num, dim_W);
for i = 1 : m
    for j = 1 : m
        G = G + p(i,j) * KF{i,j} * W{i,j};
    end
end
for it=1:niter
    %% update W
    XX = reshape(KF, [1, m * m]);
    K = [];
    for i = 1 : m * m
        K = [K, XX{i}];
    end
    H = [];
    for i = 1 : m * m
        H = blkdiag(H, p_flatten(i) * eye(dim_F));
    end
%     H = ((K * H)' * Q) * K * H;
    KH = K * H;
    H = (KH' * Q) * KH;

    F_H = eig1(H, dim_W, 1);
    W_flatten = cell(1, m*m);
    for i = 1 : m * m
        W_flatten{i} = F_H((i - 1) * dim_F + 1: i * dim_F, :);
    end
    W = reshape(W_flatten, [m, m]);


    %% update a
    beta = zeros(1, m + 1);

    for j = 1 : n
        beta(j) = trace ((G' * Graph{j}) * G + evs * (G' * G));
    end
    beta(n + 1) = trace (G' * (YYYY + evs * eye(num)) * G);
    a = beta ./ norm(beta, 2);

    %% update p
    Q = zeros(num);
    for i = 1 : n
        Q = Q + Graph{i} * a(i);
    end
    Q = Q + YYYY * a(end);

    D = cell(m, m);
    for i = 1 : m
        for j = 1 : m
            D{i,j} = KF{i, j} * W{i, j};
        end
    end
    D_flatten = reshape(D, [1, m * m]);
%     U = cell(1, m * m);
%     for i = 1 : m * m
%         U{i} = Q * D_flatten{i};
%     end
    U = cellfun(@(x) Q * x, D_flatten, 'UniformOutput', 0);
    A = Vec(D_flatten);
    B = Vec(U);
    p_flatten = eig1(A'*B, 1, 1, 0);
    p = reshape(p_flatten, [m, m]);

    %% update Y
    G = zeros(num, dim_W);
    for i = 1 : m
        for j = 1 : m
            G = G + p(i,j) * KF{i,j} * W{i,j};
        end
    end
    y = CDKM(G', Y);
    Y = full(ind2vec(y'))';
%     Y = coordinate_descend(G * G', Y);
    YYYY = (Y * (Y' * Y)^(-1)) * Y';
    obj(it) = get_obj(Y, a, p, W, KF, Graph);
    if it>2 && (obj(it) -obj(it-1)) / obj(it-1) < 1e-4
        break
    end
end
[~, Y_graph] = max(Y, [], 2);
end
function B=Vec(A)
m=length(A);
[d1,d2]=size(A{1});
B=zeros(d1*d2,m);
for j=1:m
    B(:,j)=A{j}(:);
end
end
function obj = get_obj(Y, a, p, W, KF, Graph)
m = size(KF, 1);
num = size(KF{1}, 1);
G = zeros(num, size(W{1}, 2));
for i = 1 : m
    for j = 1 : m
        G = G + p(i,j) * KF{i,j} * W{i,j};
    end
end
Q = zeros(num);
for i = 1 : length(Graph)
    Q = Q + Graph{i} * a(i);
end
YYYY = (Y * (Y' * Y)^(-1)) * Y';
Q = Q + YYYY * a(end);
obj = trace((G' * Q) * G);
end