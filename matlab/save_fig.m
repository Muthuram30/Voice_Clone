function save_fig(f, path)
% Save a figure as PNG; optionally keep it visible for interactive MATLAB use.
% Set the root appdata flag 'DSP_SHOW_FIGURES' to true before calling a
% pipeline. API/batch execution remains headless by default.
if exist('exportgraphics', 'file') == 2
    exportgraphics(f, path, 'Resolution', 110);
else
    print(f, path, '-dpng', '-r110');
end
showFigures = false;
if isappdata(0, 'DSP_SHOW_FIGURES')
    showFigures = getappdata(0, 'DSP_SHOW_FIGURES');
end
if showFigures
    set(f, 'Visible', 'on');
    drawnow;
else
    close(f);
end
end
