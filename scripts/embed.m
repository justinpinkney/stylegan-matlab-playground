% embed

imTarget = 2*(single(imresize(im, [256, 256]))./255) - 1;
gradAvgW = [];
gradSqAvgW = [];
% gradAvgX = [];
% gradSqAvgX = [];

%%
vgg = vgg16();
featureNet = dlnetwork(layerGraph(vgg.Layers(1:end-2)));
weights = load('C:\code\internal\stylegan-matlab\weights\ffhq.mat');

%%
ax = axes(figure);
lr = 0.01;
for i = 1:1000
    [featureLoss, gradsW, gradsX, im] = dlfeval(@step, w, imTarget, weights, featureNet);
    [w, gradAvgW, gradSqAvgW] = adamupdate(...
               w, gradsW, gradAvgW, gradSqAvgW, i, lr);
    disp(extractdata(featureLoss));
    imagesc(ax, (extractdata(im) + 1)/2);
%     axis equal
    drawnow();
end


function [featureLoss, gradsW, gradsX, im] = step(w, imTarget, pretrained, featureNet)
    im = stylegan.synthesis(w, pretrained, [], "randn");
%     mseLoss = mean((im - imTarget).^2, 'all');
    
    prepIm = @(x) avgpool((x+1)/2*255, 4, "Stride", 4);
    imFeatures = featureNet.forward(prepIm(im), "Output", "conv3_2");
    targetFeatures = featureNet.forward(dlarray((imTarget+1)/2*255, "SSCB"), "Output", "conv3_2");
    featureLoss = mean((imFeatures - targetFeatures).^2, 'all');
    gradsW = dlgradient(featureLoss, w);
    gradsX = [];
end