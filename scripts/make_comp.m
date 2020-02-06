

baseNoise = 150 + rand(6);
ims = imageDatastore("outputs/walk/");
ims = ims.readall;
ims = cat(4, ims{:});

maxInd = size(ims, 4);

masks = imageDatastore("outputs/masks");
masks = masks.readall;
masks = cat(3, masks{:});
%%
flatIms = reshape(ims, [], 50);
idx = (1:3145728)';

for iFrame = 1:size(masks, 3)/2
    disp(iFrame);
%     baseNoise = baseNoise + 0.3*randn(6);
%     im = imresize(baseNoise, [1024, 1024]);

    mask = masks(:,:,iFrame*2);
    mask = imresize(mask, [1024, 1024]);
    mask = double(mask)./255;
    mask = (maxInd-1)*mask + 1;
    
    mask = round(mask);
    m = repmat(mask, 1, 1, 3);
    linMask = m(:);
    
    f = sub2ind([3145728, 50], idx, linMask);
    aa = flatIms(f);
    composite  = reshape(aa, 1024, 1024, 3);
%     im = round(im+0.5*abs(round(im) - im).*randn(size(im)));

%     composite = zeros(1024, 1024, 3);
%     for i = 1:maxInd
%         selected = im == i;
%         selected = repmat(mask==i, 1, 1, 3);
%         if sum(selected(:)) == 0
%             continue
%         end
%         thisIm = ims(:,:,:,i);
%         composite(selected) = thisIm(selected);
%         for iChan = 1:3
%             channel = zeros(1024, 1024);
%             thisChan = thisIm(:, :, iChan);
%             channel(selected) = thisChan(selected);
%             composite(:, :, iChan) = composite(:, :, iChan) + channel;
%         end
%     end
    
    imshow(composite);
%     imagesc(im, [1 300])
    drawnow()

%     imwrite(composite, sprintf("outputs/comp/%06d.jpg", iFrame));
end