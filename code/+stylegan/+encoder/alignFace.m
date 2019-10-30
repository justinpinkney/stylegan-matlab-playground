function imOut = alignFace(im, landmarks)
    % Expects an image with one face
    
    outputSize = 1024;
    
    landmarks = squeeze(landmarks);
    %%
    imSize = size(im);
    eye2eye = landmarks(2,:) - landmarks(1,:);
    eyeAvg = (landmarks(2,:) + landmarks(1,:))/2;
    mouthAvg = (landmarks(4,:) + landmarks(5,:))/2;
    eye2mouth = mouthAvg - eyeAvg;
    
    x = eye2eye - fliplr(eye2mouth).*[-1, 1];
    x = x./hypot(x(1), x(2));
    x = x*max(hypot(eye2eye(1), eye2eye(2)) .* 2, hypot(eye2mouth(1),eye2mouth(2)) * 1.8);
    y = fliplr(x).*[-1, 1];
    c = eyeAvg + eye2mouth * 0.1;
    quad =    [c - x - y; c - x + y; c + x + y; c + x - y];
    polygon = [c - x - y, c - x + y, c + x + y, c + x - y];
    displayIm = insertShape(im, 'Polygon', polygon);
    
    qsize = hypot(x(1), x(2)) * 2;
    
    border = max(round(qsize * 0.1), 3);
    crop = [floor(min(quad(:,1))), floor(min(quad(:,2))), ceil(max(quad(:,1))), ceil(max(quad(:,2)))];
    crop = [max(crop(1) - border, 0), max(crop(2) - border, 0), min(crop(3) + border, imSize(1)), min(crop(4) + border, imSize(2))];
    if crop(3) - crop(1) < imSize(1) || crop(4) - crop(2) < imSize(2)
        im = imcrop(im, crop);
        quad = quad - crop(1:2);
    end
    
    % Transform.
    ref = [0, 0; 0, 1; 1, 1; 1, 0];
    ref = ref.*outputSize;
    RA = imref2d(size(im(:,:,1)));
    RB = imref2d([outputSize,outputSize]);
    T = fitgeotrans(quad, ref, 'affine');
    imOut = imwarp(im, RA, T, 'cubic', 'OutputView', RB);
end