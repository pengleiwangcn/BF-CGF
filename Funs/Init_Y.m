function Y = Init_Y(A, c)
A_ = zeros(size(A{1}, 1));
for i = 1 : length(A)
    A_ = A_ + A{i};
end
Y = finchpp(A_, c);
