function features = extractVggFeatures(im, imRange)
    
    if nargin < 2
        imRange = [-1, 1];
    end
    
    if ~isa(im, "dlarray")
        im = dlarray(single(im), "SSCB");
    end
    
    layer = "conv3_2";
    imSize = size(im);
    vggInputSize = 224;
    
    persistent featureNet
    if isempty(featureNet)
        vgg = vgg16();
        featureNet = dlnetwork(layerGraph(vgg.Layers(1:end-2)));
    end
    
    downSampleFactor = floor(min(imSize(1:2))/vggInputSize);
    
    % normalise and scale
    im = 255*(im - imRange(1))./diff(imRange);
    % downsample
    im = avgpool(im, downSampleFactor, "Stride", downSampleFactor);
    features = featureNet.forward(im, "Output", layer);

end