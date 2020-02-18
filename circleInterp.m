function out = circleInterp(val, vec1, vec2)
    % vec1 and vec2 should be perpendicular
    out = vec1.*sin(2*pi*val) + vec2.*cos(2*pi*val);
end