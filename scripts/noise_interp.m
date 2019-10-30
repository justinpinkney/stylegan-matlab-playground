useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end
% 
% filename = fullfile(projectRoot(), "weights", "ffhq.mat");
% weights = load(filename);
% weights = dlupdate(env, weights);
% % 
% z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));
% 
% w = stylegan.mapping(z, weights);

f = figure();
for i = 1:180
% r = @(x) returnNoise(x, i/100);
im = stylegan.synthesis(w, weights, [], "randnCached", i);

outIm = (1+extractdata(im))/2;
imwrite(outIm, sprintf("frames/%04d.jpg", i));

imshow(outIm)
end