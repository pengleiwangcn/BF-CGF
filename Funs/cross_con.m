function KF = cross_con(K, F, order_K)
m = length(F);
for i = 1: m
    for j = 1: m
        x = F{j};
        for k = 1: order_K
            x = K{i} * x;
        end
        KF{i, j} = x * diag(diag(x'*x).^-.5);
    end
end
end