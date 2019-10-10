z = gpuArray(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));
filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(@gpuArray, weights);
w = stylegan.mapping(z, weights);
tic
im = stylegan.synthesis(w, weights);
toc
outIm = (1+extractdata(im))/2;
f = figure();
imshow(outIm)