rng("default")
z = dlarray(single(randn(1, 1, 512, 1)), 'SSCB');
filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
w = stylegan.mapping(z, weights);
im = stylegan.synthesis(w, weights);
outIm = (1+extractdata(im))/2;
f = figure();
imshow(outIm)