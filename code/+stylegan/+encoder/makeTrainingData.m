function [ws, filenames] = makeTrainingData(destination, n, useGPU, weightFile)

    dataFile = "data.mat";

    if nargin < 3
        useGPU = false;
    end
    
    if nargin < 4
        weightFile = "ffhq.mat";
    end

    if useGPU
        env = @(x) gpuArray(x);
    else
        env = @(x) x;
    end

    rng("default");
    mkdir(destination);
    filename = fullfile(projectRoot(), "weights", weightFile);
    weights = load(filename);
    weights = dlupdate(env, weights);
    
    for iFile = 1:n
        if mod(iFile, 100) == 0
            disp(iFile);
        end
        
    
        z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

        w = stylegan.mapping(z, weights);

    
        im = stylegan.synthesis(w, weights);
        outIm = (1+extractdata(im))/2;
        outIm = uint8(255*gather(outIm));
        outIm = imresize(outIm, [256, 256]);
        
        id = java.util.UUID.randomUUID.toString;
        filename = sprintf("%s.jpg", id);
        imwrite(outIm, fullfile(destination, filename));
        
        ws(iFile, :) = squeeze(gather(extractdata(w(:,1))))';
        filenames{iFile} = filename;
    end
    
    save(dataFile, "ws", "filenames");
end