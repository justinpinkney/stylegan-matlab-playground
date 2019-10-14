% im = imread("pic.PNG");
% filename = "https://www.syfy.com/sites/syfy/files/styles/1200x680/public/wire/legacy/spock_3.jpg";
% im = imread(filename);
% im = padarray(im, 100, "symmetric");
% imshow(im)
% landmarks = ginput(5);
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
imshow(displayIm);

qsize = hypot(x(1), x(2)) * 2;

% Shrink.
% output_size = [512, 512];
% shrink = round(floor(qsize./ output_size * 0.5));
% if shrink > 1
%     rsize = (round(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
%     img = img.resize(rsize, PIL.Image.ANTIALIAS)
%     quad /= shrink
%     qsize /= shrink
% end
% Crop.
border = max(round(qsize * 0.1), 3);
crop = [floor(min(quad(:,1))), floor(min(quad(:,2))), ceil(max(quad(:,1))), ceil(max(quad(:,2)))];
crop = [max(crop(1) - border, 0), max(crop(2) - border, 0), min(crop(3) + border, imSize(1)), min(crop(4) + border, imSize(2))];
if crop(3) - crop(1) < imSize(1) || crop(4) - crop(2) < imSize(2)
    im = imcrop(im, crop);
    quad = quad - crop(1:2);
end

% Pad.
% pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
% pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
% if enable_padding and max(pad) > border - 4:
% pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
% img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
% h, w, _ = img.shape
% y, x, _ = np.ogrid[:h, :w, :1]
% mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
% blur = qsize * 0.02
% img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
% img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
% img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
% quad += pad[:2]

% Transform.
ref = [0, 0; 0, 1; 1, 1; 1, 0];
transform_size = 1024;
ref = ref.*transform_size;
RA = imref2d(size(im(:,:,1)));
RB = imref2d([transform_size,transform_size]);
T = fitgeotrans(quad, ref, 'affine');
imOut = imwarp(im, RA, T, 'cubic', 'OutputView', RB);
imshow(imOut)
% img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
% if output_size < transform_size:
% img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

imwrite(imOut, "face.jpg");