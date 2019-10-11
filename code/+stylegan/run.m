useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(@env, weights);

z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

w = stylegan.mapping(z, weights);
tic
im = stylegan.synthesis(w, weights);
toc
outIm = (1+extractdata(im))/2;
f = figure();
imshow(outIm)