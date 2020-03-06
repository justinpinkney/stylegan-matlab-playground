classdef Generator < handle
    
    properties
        Scale % 2^Scale = resolution
        CurrentScale
        Weights
        UseGPU
        NumMappingLayers = 8
        NoiseMethod = @randn
        PreBlockCallback
        PostBlockCallback
        TruncationPsi = 0.7;
        MaxTruncationScale = 8;
    end
    
    methods
        function this = Generator(weightsFile)
            this.Weights = load(weightsFile);
            this.Scale = this.determineScale(this.Weights);
            
            % move things to the gpu if required
            this.UseGPU = gpuDeviceCount > 0;
            if this.UseGPU
                this.Weights = dlupdate(@gpuArray, this.Weights);
            end
        end
        
        function im = image(this, generated)
            im = gather(1+extractdata(generated))/2;
        end
        
        function output = generate(this, z)
            if nargin < 2
                z = dlarray(single(randn(1, 1, 512, 1)), 'SSCB');
            end
            
            if this.UseGPU
                z = gpuArray(z);
            end
            
            w = this.mapping(z);
            w = this.applyTruncation(w);
            output = this.synthesis(w);
        end
        
        function wOut = applyTruncation(this, wIn)
            wOut = wIn;
            avg = this.Weights.dlatent_avg';
            for iScale = 1:2*(this.MaxTruncationScale - 1)
                wOut(:, 1, iScale) = (wOut(:, 1, iScale) - avg).*this.TruncationPsi + avg;
            end
        end
        
        function w = mapping(this, z)
            w = z;
    
            % pixel norm
            w =  w.*sqrt(mean(w.^2) + 1e-8);
            
            for iLayer = 1:this.NumMappingLayers
                weight = this.Weights.("G_mapping_Dense" + (iLayer-1) + "_weight")';
                bias = this.Weights.("G_mapping_Dense" + (iLayer-1) + "_bias")';
                w = stylegan.linear(w, weight, bias, sqrt(2), true, 0.01);
                w = leakyrelu(w, 0.2);
            end

            w = repmat(w, 1, 1, 18);
        end
        
        function output = synthesis(this, w)
            w = squeeze(w);

            x = this.getInput(this.Weights);

            this.CurrentScale = 2;
            [x, w] = this.doCallback("pre", x, w);
            x = this.inputBlock(x, w(:, 1), w(:, 2), this.Weights, this.NoiseMethod);
            [x, w] = this.doCallback("post", x, w);
            
            for iScale = 2:(this.Scale-1)
                this.CurrentScale = iScale + 1;
                [x, w] = this.doCallback("pre", x, w);
                
                resolution = 2^(iScale+1);
                styleWeights1 = w(:, iScale*2-1);
                styleWeights2 = w(:, iScale*2);
                x = this.synthesisBlock(x, styleWeights1, ...
                                            styleWeights2, ...
                                            resolution, ...
                                            this.Weights, ...
                                            this.NoiseMethod);

                [x, w] = this.doCallback("post", x, w);
            end

            output = this.toRGB(x, this.Weights);
        end
        
        function [x, w] = doCallback(this, location, x, w)
            switch location
                case "pre"
                    if ~isempty(this.PreBlockCallback)
                        [x, w] = this.PreBlockCallback(this.CurrentScale, x, w);
                    end
                case "post"
                    if ~isempty(this.PostBlockCallback)
                        [x, w] = this.PostBlockCallback(this.CurrentScale, x, w);
                    end
                otherwise
                    error("stylegan:unkownCallback", ...
                        "Didn't recognise '%s'", location);
            end
        end
        
        function x = synthesisBlock(this, x, w1, w2, scale, weights, noiseMethod)
            
            combinedUpblock = this.CurrentScale >= 7;
            prefix = "G_synthesis_" + scale + "x" + scale + "_";
            name = @(x) strcat(prefix, x);

            % conv
            weight = dlarray(weights.(name("Conv0_up_weight")), 'SSCU');

            bias = weights.(name("Conv0_up_bias"))';
            x = stylegan.Generator.conv(x, weight, zeros(size(bias), 'like', bias), 1, sqrt(2), true, combinedUpblock);
            x = stylegan.Generator.blur(x);
            x = x + shiftdim(bias, -2);

            % epi1
            weight = weights.(name("Conv0_up_StyleMod_weight"))';
            bias = weights.(name("Conv0_up_StyleMod_bias"));
            noiseWeight = shiftdim(weights.(name("Conv0_up_Noise_weight")), -1);
            x = stylegan.Generator.epilogue(x, w1, weight, bias, noiseWeight, noiseMethod);

            % conv1
            weight = dlarray(weights.(name("Conv1_weight")), 'SSCU');

            bias = weights.(name("Conv1_bias"))';
            x = stylegan.Generator.conv(x, weight, bias);

            % epilogue 2
            weight = weights.(name("Conv1_StyleMod_weight"))';
            bias = weights.(name("Conv1_StyleMod_bias"))';
            noiseWeight = shiftdim(weights.(name("Conv1_Noise_weight")), -1);
            x = stylegan.Generator.epilogue(x, w2, weight, bias, noiseWeight, noiseMethod);
        end
    end
    
    methods (Static)
        function scale = determineScale(weights)
            names = string(fieldnames(weights));
            convNames = names(contains(names, "Conv1"));
            sizes = str2double(extractBetween(convNames, ...
                                            "G_synthesis_", "x"));
            scale = log(max(sizes))/log(2);
        end
        
        function x = getInput(weights)
            bias = weights.G_synthesis_4x4_Const_bias;
            constant = weights.G_synthesis_4x4_Const_const;
            input = constant + bias;
            input = permute(input, [3, 4, 2, 1]);
            x = dlarray(input, 'SSCB');
        end
                
        function output = toRGB(x, weights)
            weight = dlarray(weights.G_synthesis_ToRGB_lod0_weight, 'SSCU');
            bias = weights.G_synthesis_ToRGB_lod0_bias;
            output = stylegan.Generator.conv(x, weight, bias, 0, 1);
        end

        
        function x = inputBlock(x, w1, w2, weights, noiseMethod)
            weight = weights.G_synthesis_4x4_Const_StyleMod_weight';
            bias = weights.G_synthesis_4x4_Const_StyleMod_bias';
            noiseWeight = shiftdim(weights.G_synthesis_4x4_Const_Noise_weight, -1);
            x = stylegan.Generator.epilogue(x, w1, weight, bias, noiseWeight, noiseMethod);

            % conv
            weight = dlarray(weights.G_synthesis_4x4_Conv_weight, 'SSCU');

            bias = weights.G_synthesis_4x4_Conv_bias';
            x = stylegan.Generator.conv(x, weight, bias);

            % epilogue 2
            weight = weights.G_synthesis_4x4_Conv_StyleMod_weight';
            bias = weights.G_synthesis_4x4_Conv_StyleMod_bias';
            noiseWeight = shiftdim(weights.G_synthesis_4x4_Conv_Noise_weight, -1);
            x = stylegan.Generator.epilogue(x, w2, weight, bias, noiseWeight, noiseMethod);
        end

        function x = conv(x, weight, bias, padding, gain, upblock, combinedUpscale)
            if nargin < 4
                padding = 1;
            end
            if nargin < 5
                gain = sqrt(2);
            end
            if nargin < 6
                upblock = false;
            end
            if nargin  < 7
                combinedUpscale = false;
            end
            wmul = gain*numel(weight(:,:,:,1)) ^ (-0.5);
            weight = wmul*weight;
            if upblock && combinedUpscale
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
                    x = stylegan.Generator.upscale(x);
                end
                x = dlconv(x, weight, bias, "Padding", padding);
            end
        end

        function [x, mu, sigma] = epilogue(x, w, weight, bias, noiseWeight, noiseMethod, mu, sigma)
            
            if nargin < 7
                mu = [];
                sigma = [];
            end

            noiseSize = size(x(:,:,1,:));
            x = x + noiseMethod(noiseSize).*noiseWeight;
            x = leakyrelu(x, 0.2);
            if isempty(mu)
                [x, mu, sigma] = batchnorm(x, zeros(size(x, 3), 1), ones(size(x, 3), 1));
            else
                x = batchnorm(x, zeros(size(x, 3), 1), ones(size(x, 3), 1), mu, sigma);
            end

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
    end
    
end