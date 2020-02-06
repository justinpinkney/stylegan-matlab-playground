function [im, landmarks] = padAndDetect(im)
    padSize = 50;
    im2 = padarray(im, [padSize, padSize], "symmetric");
    
    im2 = imgaussfilt(im2, 10);
    imSize = size(im);
    im2(padSize:padSize+imSize(1)-1, padSize:padSize+imSize(2)-1, :) = im;
    im = im2;
    
    [bboxes, scores, landmarks] = mtcnn.detectFaces(im);
end