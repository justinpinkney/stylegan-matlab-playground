% embed
gradAvgW = [];
gradSqAvgW = [];
% gradAvgX = [];
% gradSqAvgX = [];

%%
vgg = vgg16();
featureNet = dlnetwork(layerGraph(vgg.Layers(1:end-2)));

z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

w = stylegan.mapping(z, weights);

%%
ax = axes(figure);
lr = 0.01;
for i = 1:200
    [featureLoss, gradsW, gradsX, im] = dlfeval(@step, w, weights, featureNet);
    [w, gradAvgW, gradSqAvgW] = adamupdate(...
               w, gradsW, gradAvgW, gradSqAvgW, i, lr);
    disp(extractdata(featureLoss));
    imagesc(ax, (extractdata(tanh(im)) + 1)/2);
%     axis equal
    drawnow();
end


function [featureLoss, gradsW, gradsX, im] = step(w, pretrained, featureNet)
    im = stylegan.synthesis(w, pretrained, [], "randnCached");
%     mseLoss = mean((im - imTarget).^2, 'all');
    
    prepIm = @(x) avgpool((tanh(x)+1)/2*255, 4, "Stride", 4);
    input = prepIm(im);
    imFeatures = featureNet.forward(input(1:224,1:224,:,:), "Output", "fc8");
    featureLoss = -imFeatures(838);
    gradsW = dlgradient(featureLoss, w);
    gradsX = [];
end