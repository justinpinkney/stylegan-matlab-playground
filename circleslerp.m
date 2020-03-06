% full circle slerp

seeds = [12, 534];

rng(12);
v1 = randn(512, 1);
rng(534);
v2 = randn(512, 1);
v1neg = -v1;
v2neg = -v2;

omega = acos(dot(v1./norm(v1), v2./norm(v2)));


nTotal = 100;

n1 = round(nTotal/2*omega/pi);
n2 = round(nTotal/2*(pi-omega)/pi);


weightFile = 'C:\code\internal\stylegan-matlab\weights\ffhq.mat';
g = stylegan.Generator(weightFile);
g.NoiseMethod = @stylegan.randnCached;

count = 0;

for iSeg = 1:4
    switch iSeg
        case 1
            n = n1;
            vA = v1;
            vB = v2;
        case 2
            n = n2;
            vA = v2;
            vB = v1neg;
        case 3
            n = n1;
            vA = v1neg;
            vB = v2neg;
        case 4
            n = n2;
            vA = v2neg;
            vB = v1;
    end
    
    for i = 0:n-1
        z = slerp(i/(n-1), vA, vB);
        z = dlarray(single(reshape(z, 1, 1, [])), 'SSCB');
        out = g.generate(z);
        im = g.image(out);
        imwrite(im, sprintf("outputs/walk/%05d.jpg", count));
        count = count + 1;
    end
end
