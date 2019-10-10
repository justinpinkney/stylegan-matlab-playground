function x = synthesis(w, weights, x)

    if nargin < 3
        bias = weights.G_synthesis_4x4_Const_bias;
        constant = weights.G_synthesis_4x4_Const_const;
        input = constant + bias;
        input = permute(input, [3, 4, 2, 1]);
        x = dlarray(input, 'SSCB');
    end
    
    
    % epilogue 1
    x = inputBlock(x, w(:, 1), w(:, 2), weights);
    for iScale = 2:9
        x = synthesisBlock(x, w(:, iScale*2-1), w(:, iScale*2), 2^(iScale+1), weights);
    end
    
    weight = dlarray(weights.G_synthesis_ToRGB_lod0_weight, 'SSCU');
    
    bias = weights.G_synthesis_ToRGB_lod0_bias;
    x = conv(x, weight, bias, 0, 1);
    
end

function x = inputBlock(x, w1, w2, weights)
weight = weights.G_synthesis_4x4_Const_StyleMod_weight';
    bias = weights.G_synthesis_4x4_Const_StyleMod_bias';
    noiseWeight = shiftdim(weights.G_synthesis_4x4_Const_Noise_weight, -1);
    x = epilogue(x, w1, weight, bias, noiseWeight);
    
    % conv
    weight = dlarray(weights.G_synthesis_4x4_Conv_weight, 'SSCU');
    
    bias = weights.G_synthesis_4x4_Conv_bias';
    x = conv(x, weight, bias);
    
    % epilogue 2
    weight = weights.G_synthesis_4x4_Conv_StyleMod_weight';
    bias = weights.G_synthesis_4x4_Conv_StyleMod_bias';
    noiseWeight = shiftdim(weights.G_synthesis_4x4_Conv_Noise_weight, -1);
    x = epilogue(x, w2, weight, bias, noiseWeight);
end

function x = synthesisBlock(x, w1, w2, scale, weights)
    prefix = "G_synthesis_" + scale + "x" + scale + "_";
    name = @(x) strcat(prefix, x);
    
    % conv
    weight = dlarray(weights.(name("Conv0_up_weight")), 'SSCU');
    
    bias = weights.(name("Conv0_up_bias"))';
    x = conv(x, weight, zeros(size(bias), 'like', bias), 1, sqrt(2), true);
    x = blur(x);
    x = x + shiftdim(bias, -2);
    
    % epi1
    weight = weights.(name("Conv0_up_StyleMod_weight"))';
    bias = weights.(name("Conv0_up_StyleMod_bias"));
    noiseWeight = shiftdim(weights.(name("Conv0_up_Noise_weight")), -1);
    x = epilogue(x, w1, weight, bias, noiseWeight);
    
    % conv1
    weight = dlarray(weights.(name("Conv1_weight")), 'SSCU');
    
    bias = weights.(name("Conv1_bias"))';
    x = conv(x, weight, bias);
    
    % epilogue 2
    weight = weights.(name("Conv1_StyleMod_weight"))';
    bias = weights.(name("Conv1_StyleMod_bias"))';
    noiseWeight = shiftdim(weights.(name("Conv1_Noise_weight")), -1);
    x = epilogue(x, w2, weight, bias, noiseWeight);
end


function x = conv(x, weight, bias, padding, gain, upblock)
    if nargin < 4
        padding = 1;
    end
    if nargin < 5
        gain = sqrt(2);
    end
    if nargin < 6
        upblock = false;
    end
    wmul = gain*numel(weight(:,:,:,1)) ^ (-0.5);
    weight = wmul*weight;
    if upblock && size(x, 1) >= 64
        isGPU = isa(extractdata(weight), "gpuArray");
        if isGPU
            weight = gather(weight);
        end
        weight = padarray(weight, [1, 1, 0, 0]);
        weight = weight(1:end-1, 1:end-1, :, :) + ...
                weight(1:end-1, 2:end, :, :) + ...
                weight(2:end, 2:end, :, :) + ...
                weight(2:end, 1:end-1, :, :);
        tweight = dlarray(weight, 'SSUC');
        if isGPU
            tweight = gpuArray(tweight);
        end
        x = dltranspconv(x, tweight, bias, "Stride", 2, "Cropping", 1);
    else
        if upblock
            x = upscale(x);
        end
        x = dlconv(x, weight, bias, "Padding", padding);
    end
end

function x = epilogue(x, w, weight, bias, noiseWeight)
    noiseSize = size(x(:,:,1,:));
    x = x + randnCached(noiseSize).*noiseWeight;
    x = leakyrelu(x, 0.2);
    x = batchnorm(x, zeros(size(x, 3), 1), ones(size(x, 3), 1));
    
    style = stylegan.linear(w, weight, bias, 1, true, 1);
    x = x.*shiftdim(1 + style(1:size(x, 3)), -2) + shiftdim(style(size(x, 3)+1:end), -2);
end

function x = upscale(x)
    h = 1:size(x, 1);
    w = 1:size(x, 2);
    hIdxs = repelem(h, 2);
    wIdxs = repelem(w, 2);
    x = x(hIdxs, wIdxs, :, :);
end

function x = blur(x)
    blurFilter = [1, 2, 1];
    weight = blurFilter.*blurFilter';
    weight = weight./sum(weight(:));
    weight = repmat(weight, 1, 1, 1, 1, size(x, 3));
    x = dlconv(x, weight, 0, "Padding", 1);
end

function noise = randnCached(noiseSize)
    persistent cache
    if isempty(cache)
        cache = struct();
    end
    
    s = matlab.lang.makeValidName(num2str(noiseSize));
    if ~isfield(cache, s)
        cache.(s) = randn(noiseSize);
    end
    noise = cache.(s);
end