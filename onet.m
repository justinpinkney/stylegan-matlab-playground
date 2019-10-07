% https://github.com/TropComplique/mtcnn-pytorch/tree/master/src/weights

function [a, b, c] = onet(x, weights)

    for layer = 1:4
        weightName = sprintf("features_conv%d_weight", layer);
        biasName = sprintf("features_conv%d_bias", layer);
        x = dlconv(x, ...
            dlarray(weights.(weightName), "UCSS"), ...
            weights.(biasName));
        
        preluName = sprintf("features_prelu%d_weight", layer);
        x = prelu(x, weights.(preluName));
        
        if layer < 4
            if layer == 3
                kernel = 2;
            else
                kernel = 3;
            end
            if layer == 1
                x(end+1, :, :, :) = 0;
                x(:, end+1, :, :) = 0;
                padding = 0;
            else
                padding = 0;
            end
            x = maxpool(x, kernel, "Stride", 2, "Padding", padding);
        end
    end
    
    nBatch = size(x,4);
    x = dlarray(reshape(x, [], nBatch), "CB");
    x = fullyconnect(x, dlarray(weights.features_conv5_weight, "CU"), weights.features_conv5_bias);
    % inference only so no dropout
    x = prelu(x, weights.features_prelu5_weight);
    
    a = fullyconnect(x, dlarray(weights.conv6_1_weight, "CU"), weights.conv6_1_bias);
    b = fullyconnect(x, dlarray(weights.conv6_2_weight, "CU"), weights.conv6_2_bias);
    c = fullyconnect(x, dlarray(weights.conv6_3_weight, "CU"), weights.conv6_3_bias);
    a = softmax(a);
end

function x = prelu(x, params)
    params = shiftdim(params, -1);
    negX = params.*x;
    x(x<0) = negX(x<0);
end