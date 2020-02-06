% im = imread(valData{370,1});
filename = "https://images.immediate.co.uk/production/volatile/sites/3/2018/01/GettyImages-459889298-ba32cf7.jpg?quality=90&lb=940,626&background=white";
im = imread(filename);

load('ukiyoe-encoder.mat')

useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

filename = fullfile(projectRoot(), "weights", "ukiyoe-faces.mat");
weights = load(filename);
weights = dlupdate(env, weights);
w = [];
% filename = "C:\Users\jpinkney\Downloads\ezgif.com-optimize (1).gif";
for i = 1
%     [im, m] = imread(filename, i);
%     im = 255*ind2rgb(im, m);
% im = imread(filename);
[im, landmarks] = stylegan.encoder.padAndDetect(im);
% if ~isempty(newLandmarks)
%     landmarks = newLandmarks;
% end
    im = stylegan.encoder.alignFace(im, landmarks);

    im = imresize(im, [256, 256]);
    % im = fliplr(im);
    % im = circshift(im, 125, 2);
    pred = net.predict(im);

%     if isempty(w)
        w = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
%     else
%         wNew = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
%         w = (wNew + w)./2;
%         s = 4;
%         w(:, s:end) = wNew(:, s:end);
%     end
    output = stylegan.synthesis(w, weights, [], "randnCached");
    outIm = uint8(255*(1+extractdata(output))/2);
    % f = figure();
    imshowpair(im, imresize(gather(outIm), [256, 256]), "montage")

    imwrite(outIm, sprintf("frames/%06d.jpg", i));
end