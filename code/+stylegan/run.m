useGPU = false;

if useGPU
    env = @(x) gpuArray(x);
else
    env = @(x) x;
end

% filename = fullfile(projectRoot(), "weights", "ukiyoe-faces.mat");
filename = fullfile(projectRoot(), "weights", "vases.mat");
weights1 = load(filename);

filename = fullfile(projectRoot(), "weights", "ffhq.mat");
weights2 = load(filename);
%%
weights = updateW(weights1, weights2);

weights = dlupdate(env, weights);
z = env(dlarray(single(randn(1, 1, 512, 1)), 'SSCB'));

w = stylegan.mapping(z, weights);
tic
im = stylegan.synthesis(w, weights, [], "randn");
toc
outIm = (1+extractdata(im))/2;
f = figure();
imshow(outIm)

function z = updateW(x, y)
    alpha = 0;
    fields = fieldnames(x);
    for iField = 1:numel(fields)
        thisField = fields{iField};
%         if contains(thisField, "mapping")
            z.(thisField) = x.(thisField);
%             continue
%         end
        
        if contains(thisField, "4x4_Con")
            thisX = x.(thisField);
            thisY = y.(thisField);
            z.(thisField) = alpha*thisX+(1-alpha)*thisY;
        end
    end
end