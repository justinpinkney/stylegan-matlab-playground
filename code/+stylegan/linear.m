function y = linear(x, weight, bias, gain, use_wscale, lrmul)
    if use_wscale
        inputChannels = size(weight, 2);
        wmul = gain*inputChannels^(-0.5)*lrmul;
    else
        wmul = lrmul;
    end
    weight = wmul*weight;
	bias = lrmul*bias;
    y = fullyconnect(x, weight, bias);
end