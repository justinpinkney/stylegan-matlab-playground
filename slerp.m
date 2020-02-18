function out = slerp(val, low, high)
    % https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
    omega = acos(dot(low./norm(low), high./norm(high)));
    so = sin(omega);
    if so == 0
        out = (1.0-val) .* low + val .* high;
    else
        out =  sin((1.0-val)*omega) / so * low + sin(val*omega)/so * high;
    end
end