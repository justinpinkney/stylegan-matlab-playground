useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

% filename = fullfile(projectRoot(), "weights", "ukiyoe-faces.mat");
filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(env, weights);
%%
z1 = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));
z2 = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

w1 = stylegan.mapping(z1, weights);
w2 = stylegan.mapping(z2, weights);

out1 = makeIm(w1, weights);
out2 = makeIm(w2, weights);
figure;
imshowpair(out1, out2, "montage");
%%

d = 4;

im = zeros(1024, 1024, 3, 'single');
startOffset = 128;
finishOffset = 128;
im(:, 1:startOffset, :) = out2(:, 1:startOffset, :);
im(:, end-finishOffset+1:end, :) = out1(:, end-finishOffset+1:end, :);
range = 1024 - startOffset - finishOffset;
n = range/d;
for i = 1:n
    disp(i)
    alpha = (i-1)/(n-1);
    w = w1.*alpha + w2.*(1-alpha);
    outIm = makeIm(w, weights);
    start = startOffset + (i-1)*d + 1;
    finish = startOffset + i*d;
    im(:, start:finish, :) = outIm(:, start:finish, :);
end


function out = makeIm(w, weights)
    im = stylegan.synthesis(w, weights, [], "randnCached");
    out = (1+extractdata(im))/2;
end