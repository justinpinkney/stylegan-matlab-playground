function sheet
    count = 1;
    lim = 5;
    n = 64;
    z = zeros(n, n);
    s = zeros(n,n);
    a = zeros(n,n);
    [x, y] = meshgrid(1:n, 1:n);
    % s = 10*30*fspecial('gaussian',[64, 64],5);
    alpha = 0.3;
    damping = 0.15;
    restoring = 0.02;
    mass = 30;
    f = figure;
    ax = axes(f);
    drawn = imagesc(ax, z, [-lim, lim]);
    drawn.ButtonDownFcn = @doClick;

    while true
        zPad = padarray(z, [1, 1], 0);
        dxdz1 = diff(zPad(1:end-1,2:end-1), 1, 1);
        dxdz2 = fliplr(diff(fliplr(zPad(2:end, 2:end-1)), 1, 1));

        dydz1 = diff(zPad(2:end-1, 1:end-1), 1, 2);
        dydz2 = flipud(diff(flipud(zPad(2:end-1, 2:end)), 1, 2));
        a = -alpha.*(dxdz1 - dxdz2 + dydz1 - dydz2);
        a = a + -restoring.*z;
        a = a + -damping.*s;
        s = s + a/mass;
        z = z + s;
        drawn.CData = z;
        drawnow
        
        count = count + 1;
        if mod(count, 2) == 0
            filename = sprintf("outputs/masks/%05d.png", count/2);
            imwrite(z./lim/2 + 0.5, filename);
        end
        
    end

    function doClick(src, evt)
        if evt.Button == 1
            direction = 1;
        else
            direction = -1;
        end
        coords = round(evt.IntersectionPoint);
        box = 5;
        scale = 20;
        rows = coords(2)-box:coords(2)+box;
        cols = coords(1)-box:coords(1)+box;
        offset = scale*fspecial('gaussian',[2*box + 1, 2*box + 1],box/2);
        s(rows, cols) = s(rows, cols) + direction*offset;
    end
end