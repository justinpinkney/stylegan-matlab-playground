useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end
% 
filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(env, weights);
% 
z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

w = stylegan.mapping(z, weights);

% jack = load("jack.mat");
% cleve = load("cleve.mat");
%%
f = figure();
n = 100;
for i = 1:n
% r = @(x) returnNoise(x, i/100);
alpha = 5*(i-1)/n;
im = stylegan.synthesis(w, weights, [], "randnCached", alpha);

outIm = (1+extractdata(im))/2;
imwrite(outIm, sprintf("frames/%04d.jpg", i));

% imshow(outIm)
end