% https://github.com/TropComplique/mtcnn-pytorch/tree/master/src/weights

function [a, b] = rnet(x, weights)

    for layer = 1:3
        weightName = sprintf("features_conv%d_weight", layer);
        biasName = sprintf("features_conv%d_bias", layer);
        x = dlconv(x, ...
            dlarray(weights.(weightName), "UCSS"), ...
            weights.(biasName));
        
        preluName = sprintf("features_prelu%d_weight", layer);
        x = prelu(x, weights.(preluName));
        
        if layer == 1 || layer == 2
            if layer == 1
                padding = 1;
            else
                padding = 0;
            end
            x = maxpool(x, 3, "Stride", 2, "Padding", padding);
        end
    end
    
    nBatch = size(x,4);
    x = dlarray(reshape(x, [], nBatch), "CB");
    x = fullyconnect(x, dlarray(weights.features_conv4_weight, "CU"), weights.features_conv4_bias);
    x = prelu(x, weights.features_prelu4_weight);
    a = fullyconnect(x, dlarray(weights.conv5_1_weight, "CU"), weights.conv5_1_bias);
    b = fullyconnect(x, dlarray(weights.conv5_2_weight, "CU"), weights.conv5_2_bias);
    a = softmax(a);
end

function x = prelu(x, params)
    params = shiftdim(params, -1);
    negX = params.*x;
    x(x<0) = negX(x<0);
end