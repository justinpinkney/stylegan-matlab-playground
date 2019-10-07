% https://github.com/TropComplique/mtcnn-pytorch/tree/master/src/weights

function [a, b] = pnet(x, weights)

    for layer = 1:3
        weightName = sprintf("features_conv%d_weight", layer);
        biasName = sprintf("features_conv%d_bias", layer);
        x = dlconv(x, ...
            dlarray(weights.(weightName), "UCSS"), ...
            weights.(biasName));
        
        preluName = sprintf("features_prelu%d_weight", layer);
        x = prelu(x, weights.(preluName));
        
        if layer == 1
            x = maxpool(x, 2, "Stride", 2);
        end
    end
    
    a = dlconv(x, dlarray(weights.conv4_1_weight, "UCSS"), weights.conv4_1_bias);
    b = dlconv(x, dlarray(weights.conv4_2_weight, "UCSS"), weights.conv4_2_bias);
    a = softmax(a);

end

function x = prelu(x, params)
    params = shiftdim(params, -1);
    negX = params.*x;
    x(x<0) = negX(x<0);
end