% im = imread(valData{27,1});
% filename = "https://media.fromthegrapevine.com/assets/images/2018/4/einstein-0417-social.jpg.824x0_q71_crop-scale.jpg";
% im = imread(filename);

load('encoder-100k.mat')

useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(env, weights);
w = [];
filename = "C:\Users\jpinkney\Downloads\ezgif.com-optimize (1).gif";
for i = 1:54
    [im, m] = imread(filename, i);
    im = 255*ind2rgb(im, m);
% im = imread(filename);
[im, newLandmarks] = padAndDetect(im);
if ~isempty(newLandmarks)
    landmarks = newLandmarks;
end
    im = stylegan.encoder.alignFace(im, landmarks);

    im = imresize(im, [256, 256]);
    % im = fliplr(im);
    % im = circshift(im, 125, 2);
    pred = net.predict(im);

    if isempty(w)
        w = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
    else
        wNew = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
        w = (wNew + w)./2;
%         s = 4;
%         w(:, s:end) = wNew(:, s:end);
    end
    output = stylegan.synthesis(w, weights, [], "randnCached");
    outIm = uint8(255*(1+extractdata(output))/2);
    % f = figure();
    % imshowpair(im, imresize(gather(outIm), [256, 256]), "montage")

    imwrite(outIm, sprintf("frames/%06d.jpg", i));
end