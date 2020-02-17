weightFile = 'C:\code\internal\stylegan-matlab\weights\ffhq.mat';
g = stylegan.Generator(weightFile);
g.PreBlockCallback = @callback;
g.NoiseMethod = @stylegan.randnCached;
z = dlarray(single(randn(1, 1, 512, 1)), 'SSCB');
out = g.generate(z);
imshow(g.image(out));

function [x, w] = callback(scale, x, w)
    if scale == 2
        x = 0 .* x;
    end
end