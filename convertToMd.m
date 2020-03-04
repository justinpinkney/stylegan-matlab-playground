exampleDir = "examples";

livescripts = dir(fullfile(exampleDir, "*.mlx"));

for iFile = 1:numel(livescripts)
    filename = fullfile(livescripts(iFile).folder, livescripts(iFile).name);
    texFilename = strrep(filename, '.mlx', '.tex');
    mdFilename = strrep(filename, '.mlx', '');
    matlab.internal.liveeditor.openAndConvert(filename, texFilename);
    latex2markdown(texFilename, 'outputfilename', mdFilename);
    delete(texFilename);
    delete("examples/*.sty")
end