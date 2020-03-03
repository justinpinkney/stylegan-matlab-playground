weightFile = 'C:\code\internal\stylegan-matlab\weights\ffhq.mat';
g = stylegan.Generator(weightFile);

nA = 3;
styleRanges = [ 1,4;
                5,9;
                10,18;];
            
zA = dlarray(single(randn(1, 1, 512, nA)), "SSCB");
wA = g.applyTruncation(g.mapping(zA));

nB = size(styleRanges, 1);
zB = dlarray(single(randn(1, 1, 512, nB)), "SSCB");
wB = g.applyTruncation(g.mapping(zB));

im = cell(nA+1, nB+1);
im{1,1} = ones(512,512,3);

for iA = 1:nA
    im{iA+1, 1} = g.image(g.synthesis(wA(:,iA,:)));
end

for iB = 1:nB
    im{1, iB+1} = g.image(g.synthesis(wB(:,iB,:)));
end

for iA = 1:nA
    for iB = 1:nB
        styleRange = styleRanges(iB, :);
        w = wB(:,iB,:);
        w(:, 1, styleRange(1):styleRange(2)) = ...
            wA(:, iA, styleRange(1):styleRange(2));
        im{iA+1, iB+1} = g.image(g.synthesis(w));
    end
end

tiled = imtile(im(:), "GridSize", [nB+1, nA+1]);
imshow(tiled)
imwrite(tiled, "style_mixing.jpg")