function w = mapping(z, pretrained)
    w = z;
    
    % pixel norm
    w =  w.*sqrt(mean(w.^2) + 1e-8);
    nLayers = 8;
    for iLayer = 1:nLayers 
        weight = pretrained.("G_mapping_Dense" + (iLayer-1) + "_weight")';
        bias = pretrained.("G_mapping_Dense" + (iLayer-1) + "_bias")';
        w = stylegan.linear(w, weight, bias, sqrt(2), true, 0.01);
        
        w = leakyrelu(w, 0.2);
    end
    
    w = repmat(w, 1, 18, 1);
end