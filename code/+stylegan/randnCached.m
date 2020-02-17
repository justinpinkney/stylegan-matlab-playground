function noise = randnCached(noiseSize, seed)
% return the same noise for a given size and seed

    if nargin < 2
        seed = 0;
    end
    persistent cache
    if isempty(cache)
        cache = struct();
    end
    
    s = matlab.lang.makeValidName(num2str([noiseSize, seed]));
    if ~isfield(cache, s)
        cache.(s) = randn(noiseSize);
    end
    noise = cache.(s);
end