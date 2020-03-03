weightFile = 'C:\code\internal\stylegan-matlab\weights\mccabe.mat';
g = stylegan.Generator(weightFile);
% g.PreBlockCallback = @callback;
g.NoiseMethod = @stylegan.randnCached;
z = dlarray(single(randn(1, 1, 512, 1)), 'SSCB');
out = g.generate(z);
imshow(g.image(out));

function [x, w] = callback(scale, x, w)
    if scale == 2
        x = cat(1, x, x.*randn(1, 1, 512));
        x = cat(2, x, x.*randn(1, 1, 512));
    end
end