function [bboxes, scores, landmarks] = detector(im)

    landmarks = [];
    
    %% Stage 1
    minSize = 24;
    imSize = size(im);
    maxScale = ceil(log(min(imSize(1:2)./minSize))/log(sqrt(2)));
    pnetThreshold = 0.6;
    rnetThreshold = 0.7;
    onetThreshold = 0.8;
    
    bboxes =[];
    scores = [];
    for iScale = 1:maxScale
        scale = sqrt(2)^(iScale+1);
        [thisBox, thisScore] = mtcnn.proposeRegions(im, scale, pnetThreshold);
        bboxes = cat(1, bboxes, thisBox);
        scores = cat(1, scores, thisScore);
    end
    
    % TODO DRY
    [bboxes, scores] = selectStrongestBbox(bboxes, scores, "RatioType", "Min");
    
    %% Stage 2
    rnetSize = 24;
    bboxes = mtcnn.makeSquare(bboxes);
    bboxes = round(bboxes);
   
    % make crops
    nBoxes = size(bboxes, 1);
    cropped = zeros(rnetSize, rnetSize, 3, nBoxes);
    for iBox = 1:nBoxes
        thisBox = bboxes(iBox, :);
        cropped(:,:,:,iBox) = imresize(imcrop(im, thisBox), [rnetSize, rnetSize]);
    end
    
    cropped = dlarray(single(cropped)./255*2 - 1, "SSCB");
    weights = load('C:\Users\justin\Downloads\rnet.mat');
    [probs, correction] = rnet(cropped, weights);
    faceProbs = extractdata(probs(2, :));
    scaledOffset = bboxes(:, 3)'.*extractdata(correction);
    bboxes = bboxes + scaledOffset'; % plus one?
    bboxes(faceProbs < rnetThreshold, :) = [];
    scores = faceProbs(faceProbs > rnetThreshold)';

    [bboxes, scores] = selectStrongestBbox(bboxes, scores, "RatioType", "Min");
    
    %% Stage 3
    onetSize = 48;
    bboxes = mtcnn.makeSquare(bboxes);
%     bboxes = round(bboxes);
   
    % TODO DRY
    nBoxes = size(bboxes, 1);
    cropped = zeros(onetSize, onetSize, 3, nBoxes);
    for iBox = 1:nBoxes
        thisBox = bboxes(iBox, :);
        cropped(:,:,:,iBox) = imresize(imcrop(im, thisBox), [onetSize, onetSize]);
    end
    cropped = dlarray(single(cropped)./255*2 - 1, "SSCB");
    weights = load('C:\Users\justin\Downloads\onet.mat');
    [probs, correction, landmarks] = onet(cropped, weights);
    % landmarks are relative to uncorrected bbox
    landmarks = extractdata(landmarks);
    x = bboxes(:, 1) + landmarks(1:5, :)'.*(bboxes(:, 3)-1);
    y = bboxes(:, 2) + landmarks(6:10, :)'.*(bboxes(:, 4)-1);
    landmarks = cat(3, x, y);
    
    faceProbs = extractdata(probs(2, :));
    scaledOffset = bboxes(:, 3)'.*extractdata(correction);
    bboxes = bboxes + scaledOffset';
    bboxes(faceProbs < onetThreshold, :) = [];
    landmarks(faceProbs < onetThreshold, :, :) = [];
    scores = faceProbs(faceProbs> onetThreshold)';
    [bboxes, scores] = selectStrongestBbox(bboxes, scores, "RatioType", "Min");
    
%     landmarks = extractdata(landmarks);
end