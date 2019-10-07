function [bboxes, scores] = proposeRegions(im, scale, threshold)
    
    stride = 2;
    pnetSize = 12;
    
    im = imresize(im, 1/scale);
    im = dlarray(single(im)./255*2 - 1, "SSCB");
    weights = load('C:\Users\justin\Downloads\pnet.mat');
    
    [probability, correction] = pnet(im, weights);
    
    faces = probability(:,:,2) > threshold;
    if sum(faces, 'all') == 0
        bboxes = [];
        scores = [];
        return
    end
    [iY, iX] = find(extractdata(faces));
    
    bboxes = [scale*stride*(iX - 1) + 1, scale*stride*(iY - 1) + 1];
    bboxes(:, 3:4) = scale*pnetSize;
    
    scores = zeros(size(bboxes(:,1)));
    offsets = extractdata(correction(iY, iX,:));
    for iBox = 1:size(bboxes, 1)
        thisOffset = squeeze(correction(iY(iBox), iX(iBox), :));
        scaledOffset = scale*pnetSize*thisOffset;
        bboxes(iBox, :) = bboxes(iBox, :) + extractdata(scaledOffset)';
        scores(iBox) = extractdata(probability(iY(iBox), iX(iBox), 2));
    end

end
