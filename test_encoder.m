% im = imread(valData{27,1});
% im = imread("testim.jpg");
load('encoder_net.mat')
%%
im = imread("jack.jpg");
im = imresize(im, [256, 256]);
pred = net.predict(im);

filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
weights = dlupdate(@gpuArray, weights);
w = dlarray(gpuArray(repmat(pred', 1, 18)), "CB");
output = stylegan.synthesis(w, weights);
outIm = uint8(255*(1+extractdata(output))/2);
f = figure();
imshowpair(im, imresize(gather(outIm), [256, 256]), "montage")