weightFile = 'C:\code\internal\stylegan-matlab\weights\mccabe.mat';
load(weightFile);

v1 = rand(512, 1);

scale = 1;
v1 = scale.*v1;
v2 = scale.*v2;

avg = weights.dlatent_avg;

n = 20;


g = stylegan.Generator(weightFile);
g.NoiseMethod = @stylegan.randnCached;

p = progress.Bar("Interpolating", n);
for i = 0:n
    p.update();
    z = circleInterp(i/n, v1, v2);
    z = avg' + z;
    z = dlarray(single(reshape(z, 1, 1, [])), 'SSCB');
    out = g.generate(z);
    im = g.image(out);
    imwrite(im, sprintf("outputs/walk/%05d.jpg", i));
end
