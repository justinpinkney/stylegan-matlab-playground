function noise = returnNoise(noiseSize, alpha)
    persistent cache
    if isempty(cache)
        cache = struct();
    end
    
    s = matlab.lang.makeValidName(num2str(noiseSize));
    if ~isfield(cache, s)
        cache.(s) = {0*randn(noiseSize), 2*randn(noiseSize)};
    end
    noise = alpha*cache.(s){1} + (1-alpha)*cache.(s){2};
end