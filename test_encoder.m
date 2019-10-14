% im = imread(valData{27,1});
im = imread("face.jpg");
load('weights/encoder.mat')
im = imresize(im, [256, 256]);
pred = net.predict(im);

useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(env, weights);
w = dlarray(env(repmat(pred', 1, 1, 18)), "CUB");
output = stylegan.synthesis(w, weights, [], "randn");
outIm = uint8(255*(1+extractdata(output))/2);
f = figure();
imshowpair(im, imresize(gather(outIm), [256, 256]), "montage")
% imshow(imresize(gather(outIm), [256, 256]))