urls = ["https://raw.githubusercontent.com/ox-vgg/vgg_face2/master/samples%20(test%20set)/tight_crop%20(used%20for%20training)/n000106/0001_03.jpg"
    "https://github.com/ox-vgg/vgg_face2/raw/master/samples%20(test%20set)/tight_crop%20(used%20for%20training)/n000106/0004_01.jpg"
    "https://github.com/ox-vgg/vgg_face2/raw/master/samples%20(test%20set)/tight_crop%20(used%20for%20training)/n007241/0007_01.jpg"];

for iPic = 1:numel(urls)
    imOut = single(imread(urls(iPic)));
    imOut = imresize(imOut, [224, 224]);
    imOut = imOut - shiftdim([131.0912, 103.8827, 91.4953], -1);
    f(iPic, :) = vggface.predict(imOut.);
%     f(iPic, :) = f(iPic, :)./norm(f(iPic, :));
end
%%
D = pdist2(f, f, "euclidean");
S = pdist2(f, f, "cosine");

%%
arcnetLmarks = [30.2946, 51.6963;
          65.5318, 51.5014;
          48.0252, 71.7366;
          33.5493, 92.3655;
          62.7299, 92.2041] + [8, 0];
      
lmarks =    [42.4355   52.2551
   70.8226   53.2347
   53.7903   71.1939
   42.6935   84.5816
   70.8226   84.5816];

%%
% ref = [0, 0; 0, 1; 1, 1; 1, 0];
transform_size = 112;
% ref = ref.*transform_size;
RA = imref2d(size(imOut(:,:,1)));
RB = imref2d([transform_size,transform_size]);

tform = fitgeotrans(lmarks, arcnetLmarks, "similarity")
imNew = imwarp(imOut, RA, tform, 'cubic', 'OutputView', RB);
imshowpair(imOut, imNew)
