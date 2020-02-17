v1 = rand(512, 1);
v2 = rand(512, 1);

n = 10;

weightFile = 'C:\code\internal\stylegan-matlab\weights\ffhq.mat';
g = stylegan.Generator(weightFile);
g.NoiseMethod = @stylegan.randnCached;



for i = 0:n
    z = slerp(i/n, v1, v2);
    z = dlarray(single(reshape(z, 1, 1, [])), 'SSCB');
    out = g.generate(z);
    im = g.image(out);
    imwrite(im, sprintf("outputs/walk/%05d.jpg", i));
end
