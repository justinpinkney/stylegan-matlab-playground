nW = 1;

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
z = env(dlarray(single(randn(1, 1, 512, nW+1)), 'SSCB'));
w = stylegan.mapping(z, weights);

me1 = load("me1.mat");
me2 = load("me2.mat");
w(:,1,:) = me1.w;
n = 50;
nPerW = n/nW;
w1 = w(:,1,:);
wCount = 2;
w2 = w(:,wCount,:);

%%
% w1 = me1.w;
% w2 = me2.w;
for i = 1:n
    disp(i)
    if i>1 && mod(i-1, nPerW) == 0
        w1 = w2;
        wCount = wCount + 1;
        w2 = w(:,wCount,:);
    end
    alpha = (mod(i-1, nPerW)+1)/nPerW;
    thisW = w2.*alpha + w1.*(1-alpha);
    outIm = makeIm(thisW, weights);
    imwrite(outIm, sprintf("outputs/walk/%05d.jpg", i));
end


function out = makeIm(w, weights)
    im = stylegan.synthesis(w, weights, [], "randnCached");
    out = (1+extractdata(im))/2;
end