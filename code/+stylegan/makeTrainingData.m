function [ws, filenames] = makeTrainingData(destination, n)
    rng("default");
    mkdir(destination);
    filename = fullfile(projectRoot(), "weights", "ffhq.mat");
    weights = load(filename);
    weights = dlupdate(@gpuArray, weights);
    
    ws = zeros(n, 512, "single");
    filenames = cell(n, 1);

    for iFile = 1:n
        if mod(iFile, 100) == 0
            disp(iFile);
        end
        z = gpuArray(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));
        w = stylegan.mapping(z, weights);
        im = stylegan.synthesis(w, weights);
        outIm = (1+extractdata(im))/2;
        outIm = uint8(255*gather(outIm));
        outIm = imresize(outIm, [256, 256]);
        
        filename = sprintf("%08d.jpg", iFile);
        imwrite(outIm, fullfile(destination, filename));
        
        ws(iFile, :) = squeeze(gather(extractdata(w(:,1))))';
        filenames{iFile} = filename;
    end
end