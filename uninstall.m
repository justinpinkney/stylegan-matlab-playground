function uninstall()
    pathsToRemove = projectPaths();
    for iPath = 1:numel(pathsToRemove)
        rmpath(pathsToRemove(iPath));
    end
end