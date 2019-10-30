filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights = load(filename);
z = dlarray(randn(512, 1, "single"), "UB");
w = stylegan.mapping(z, weights);
gradAvgW = [];
gradSqAvgW = [];
% gradAvgX = [];
% gradSqAvgX = [];

%%
% arcface = load("C:\code\internal\mtcnn\arcface.mat");
lg = layerGraph(vggface);
lg = lg.removeLayers('RegressionOutput');
% lg = lg.removeLayers('output');
featureNet = dlnetwork(lg);

w0 = w;
[im0, a] = makeImageAndEmbed(w0, weights, featureNet);
imwrite((extractdata(im0) + 1)/2, sprintf("identities/%06d.jpg", 0));
% ax = axes(figure);
% w = w + 0.1*randn(size(w), 'like', w);
z = dlarray(randn(512, 1, "single"), "UB");
w = 0.9*w0 + 0.1*stylegan.mapping(z, weights);
%%
lr = 0.01;
for i = 1:1000
    [featureLoss, gradsW, gradsX, im] = dlfeval(@step, w, w0, weights, featureNet, a);
    [w, gradAvgW, gradSqAvgW] = adamupdate(...
               w, gradsW, gradAvgW, gradSqAvgW, i, lr, 0.5);
%     disp(extractdata(featureLoss));
%     imagesc(ax, (extractdata(im) + 1)/2);
    imwrite((extractdata(im) + 1)/2, sprintf("identities/%06d.jpg", i));
%     axis equal
%     drawnow();
end


function [loss, gradsW, gradsX, im] = step(w, w0, pretrained, featureNet, a)
    [im, b] = makeImageAndEmbed(w, pretrained, featureNet);
    a = a./sqrt(sum(a.^2));
    b = b./sqrt(sum(b.^2));
%     featureLoss = (1-sum(a.*b, 'all'));
featureLoss = sum((a-b).^2, 'all');
    penalty = 0.1*sqrt(sum((w - pretrained.dlatent_avg').^2, 'all'));
    distanceLoss = -0.1*sqrt(sum((w - w0).^2, 'all'));
    loss = featureLoss + distanceLoss + penalty;
    
    show = @(x) fprintf('%f ', extractdata(x));
    show(featureLoss)
    show(distanceLoss)
    show(penalty);
    show(loss);
    fprintf('\n')
    gradsW = dlgradient(loss, w);
    gradsX = [];
end

function [im, embedding] = makeImageAndEmbed(w, pretrained, featureNet)
    im = stylegan.synthesis(w, pretrained, [], "randnCached");
    prepIm = @(x) avgpool((x+1)/2*255, 3, "Stride", 3);
    imIn = prepIm(im) - shiftdim([131.0912, 103.8827, 91.4953], -1);
    embedding = featureNet.predict(dlarray(imIn(58:281, 58:281, :), "SSCB"));
end