function [ws, filenames] = makeTrainingData(destination, n, useGPU)

    if nargin < 3
        useGPU = false;
    end

    if useGPU
        env = @(x) gpuArray(x);
    else
        env = @(x) x;
    end

    rng("default");
    mkdir(destination);
    filename = fullfile(projectRoot(), "weights", "ffhq.mat");
    weights = load(filename);
    weights = dlupdate(env, weights);
    
    zPool = env(dlarray(single(randn(1, 1, 512, n)), 'SSCB'));
    wPool = stripdims(stylegan.mapping(zPool, weights));

    for iFile = 1:n
        if mod(iFile, 100) == 0
            disp(iFile);
        end
        
        splitPos = randi(17);
        w1 = randi(n);
        w2 = randi(n);
        thisW = cat(3, wPool(:, w1, 1:splitPos), ...
                        wPool(:, w2, splitPos+1:end));
        thisW = dlarray(thisW, "CBU");
        
        im = stylegan.synthesis(thisW, weights);
        outIm = (1+extractdata(im))/2;
        outIm = uint8(255*gather(outIm));
        outIm = imresize(outIm, [256, 256]);
        
        id = java.util.UUID.randomUUID.toString;
        filename = sprintf("%s.jpg", id);
        imwrite(outIm, fullfile(destination, filename));
        
%         ws(iFile, :) = squeeze(gather(extractdata(w(:,1))))';
%         filenames{iFile} = filename;
    end
end