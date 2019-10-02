function install()
    pathsToAdd = projectPaths();
    for iPath = 1:numel(pathsToAdd)
        addpath(pathsToAdd(iPath));
    end
end