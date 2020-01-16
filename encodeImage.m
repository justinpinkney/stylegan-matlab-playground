function [w, outIm] = encodeImage(im)
    if ischar(im) || isStringScalar(im)
        im = imread(im);
    end
    load('weights/encoder-100k.mat')
    
    useGPU = false;
    
    if useGPU
        env = @(x) gpuArray(x);
    else
        env = @(x) x;
    end
    
    [im, landmarks] = stylegan.encoder.padAndDetect(im);
    if isempty(landmarks)
        f = figure();
        imshow(im)
        landmarks = ginput(5); 
        close(f);
    end
    outIm = stylegan.encoder.alignFace(im, landmarks);

    im = imresize(outIm, [256, 256]);
    % im = fliplr(im);
    % im = circshift(im, 125, 2);
    pred = net.predict(im);

    w = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
%     output = stylegan.synthesis(w, weights, [], "randnCached");
%     outIm = uint8(255*(1+extractdata(output))/2);
    % f = figure();
    % imshowpair(im, imresize(gather(outIm), [256, 256]), "montage")
end