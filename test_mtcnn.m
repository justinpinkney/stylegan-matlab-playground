im = imread("pic.PNG");
% im = imread("https://raw.githubusercontent.com/TropComplique/mtcnn-pytorch/master/images/office4.jpg");
originalIm = im;
[bboxes, scores, landmarks] = detector(im);
% 
%%
displayIm = im;
displayIm = insertShape(displayIm, "rectangle", bboxes);    
imshow(displayIm)


hold on
for iBox = 1:size(bboxes, 1)
    x = landmarks(iBox, :, 1);
    y = landmarks(iBox, :, 2);
    scatter(x, y, 'filled')
end