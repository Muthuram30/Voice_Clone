function matlab_runner(requestJsonPath)
% MATLAB_RUNNER  Single fixed entry-point triggered by the Python application.
%
%   matlab_runner('path/to/request.json')
%
% The Python application writes a JSON *request* file (schema below), then
% calls this function via the MATLAB Engine API or the -batch CLI.  This
% function dispatches to the appropriate DSP stage, writes all figures and
% result artefacts into the 'outDir' declared in the request, and finally
% overwrites 'path/to/request.json' with a JSON *response* (schema below).
%
% Request schema
% --------------
%   {
%     "task"     : "analysis" | "postprocess"   (required)
%     "wavPath"  : "/abs/path/to/audio.wav"     (required)
%     "outDir"   : "/abs/path/to/output/dir"    (required)
%     "params"   : {                            (optional, defaults shown)
%       "targetFs"      : 16000,   % analysis only
%       "hpHz"          : 70,      % postprocess only
%       "lpHz"          : 0,       % postprocess only (0 = off)
%       "targetRmsDbfs" : -20      % postprocess only
%     }
%   }
%
% Response schema (written back to requestJsonPath)
% -------------------------------------------------
%   {
%     "status"  : "ok" | "error",
%     "task"    : <echo of request task>,
%     "outDir"  : <echo of outDir>,
%     "message" : <only when status == "error">,
%     ...        all fields from results.json on success
%   }
%
% Toolboxes: Signal Processing Toolbox only (MATLAB R2020a+ recommended,
%            R2019a minimum for stft/istft).
%
% See also: main_analysis, main_postprocess

% ---- 0. Validate argument ------------------------------------------------
if nargin < 1 || isempty(requestJsonPath)
    error('matlab_runner: requestJsonPath argument is required.');
end
requestJsonPath = char(requestJsonPath);
if ~isfile(requestJsonPath)
    error('matlab_runner: request file not found: %s', requestJsonPath);
end

% ---- 1. Parse request JSON -----------------------------------------------
try
    req = jsondecode(fileread(requestJsonPath));
catch e
    write_response(requestJsonPath, struct('status','error', ...
        'task','', 'outDir','', ...
        'message', ['Failed to parse request JSON: ' e.message]));
    return
end

task    = char(req.task);
wavPath = char(req.wavPath);
outDir  = char(req.outDir);

if ~isfile(wavPath)
    write_response(requestJsonPath, make_err(task, outDir, ...
        ['wavPath not found: ' wavPath]));
    return
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

% ---- 2. Verify Signal Processing Toolbox is available -------------------
need = {'resample','butter','filtfilt','stft','istft','spectrogram', ...
        'lpc','dct','pwelch','medfilt1'};
miss = need(cellfun(@(f) isempty(which(f)), need));
if ~isempty(miss)
    write_response(requestJsonPath, make_err(task, outDir, ...
        ['Signal Processing Toolbox functions missing: ' strjoin(miss,', ')]));
    return
end

% ---- 3. Dispatch to analysis or postprocess ------------------------------
try
    switch lower(task)
        case 'analysis'
            targetFs = get_param(req, 'targetFs', 16000);
            main_analysis(wavPath, outDir, targetFs);
            res = read_results(outDir);
            res.task   = task;
            res.outDir = outDir;

        case 'postprocess'
            hpHz        = get_param(req, 'hpHz',         70);
            lpHz        = get_param(req, 'lpHz',           0);
            targetRmsDb = get_param(req, 'targetRmsDbfs', -20);
            main_postprocess(wavPath, outDir, hpHz, lpHz, targetRmsDb);
            res = read_results(outDir);
            res.task   = task;
            res.outDir = outDir;

        otherwise
            error('matlab_runner: unknown task "%s". Use "analysis" or "postprocess".', task);
    end
catch err
    write_response(requestJsonPath, make_err(task, outDir, err.message));
    return
end

% ---- 4. Write unified response back to the request file -----------------
write_response(requestJsonPath, res);
end

% ==========================================================================
% Local helper functions
% ==========================================================================

function v = get_param(req, name, default)
% Return req.params.<name> if present, otherwise default.
if isfield(req, 'params') && isstruct(req.params) && isfield(req.params, name)
    v = req.params.(name);
else
    v = default;
end
end

function res = read_results(outDir)
% Read results.json produced by main_analysis / main_postprocess.
p = fullfile(outDir, 'results.json');
if ~isfile(p)
    error('matlab_runner: results.json not produced in %s', outDir);
end
res = jsondecode(fileread(p));
if ~strcmp(char(res.status), 'ok')
    if isfield(res, 'message')
        msg = char(res.message);
    else
        msg = 'Unknown error in inner function.';
    end
    error('matlab_runner: inner function failed: %s', msg);
end
end

function s = make_err(task, outDir, msg)
s = struct('status','error','task',task,'outDir',outDir,'message',msg);
end

function write_response(path, s)
% Atomically write JSON response: write to .tmp then rename.
tmp = [path '.tmp'];
fid = fopen(tmp, 'w');
if fid == -1
    % Fall back to direct overwrite if temp write fails.
    fid = fopen(path, 'w');
    if fid ~= -1
        fwrite(fid, jsonencode(s), 'char');
        fclose(fid);
    end
    return
end
fwrite(fid, jsonencode(s), 'char');
fclose(fid);
try
    movefile(tmp, path, 'f');
catch
    % Cross-device rename failed; direct write.
    fid = fopen(path, 'w');
    if fid ~= -1
        fwrite(fid, jsonencode(s), 'char');
        fclose(fid);
    end
    if isfile(tmp), delete(tmp); end
end
end
