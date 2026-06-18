function cfg  = configdataset (dataset, dir_main)
% CONFIGDATASET configures testing datasets.
%
%   CFG = configdataset(DATASET, ROOT_DIR)
%
%   DATASET  : string with dataset name, ie 'oxford5k' or 'paris6k'
%   ROOT_DIR : directory where datasets are stored
%
%   CFG      : structure with dataset configuration

switch lower(dataset)
    case 'oxford5k'
        params.ext = '.jpg';
        params.qext = '.jpg';
        params.dir_data=dir_main ;
        cfg = config_oxford (params);
        
    case 'paris6k'
        params.ext = '.jpg';
        params.qext = '.jpg';
        params.dir_data=dir_main;
        cfg = config_paris (params);
        
    case 'roxford5k'
        params.ext = '.jpg';
        params.qext = '.jpg';
        params.dir_data =dir_main;
        cfg = config_roxford (params);
        
    case 'rparis6k'
        params.ext = '.jpg';
        params.qext = '.jpg';
        params.dir_data = dir_main;
        cfg = config_rparis (params);
        
    case 'r1m'
        params.ext = '.jpg';
        params.dir_data = dir_main;
        cfg = config_revisitop1m (params);
        
    case 'cub'
        [~,params,~] = cub_load;
        cfg.imlist = params.filepath;
        cfg.qimlist = params.filepath;
        cfg.n = length (cfg.imlist);   % number of database images
        cfg.nq = length (cfg.qimlist);    % number of query images
        
    otherwise, error ('Unkown dataset %s\n', dataset);

end

cfg.dataset = dataset;

if strcmp(dataset, 'roxford5k')

    dataset = 'oxford5k';

elseif strcmp(dataset, 'rparis6k')

    dataset = 'paris6k';
    
end

% some filename overwriting
cfg.dir_images = sprintf ('%s/dataset/%s/%s_images/',cfg.dir_data,dataset,dataset);

cfg.im_fname = @config_imname;

cfg.qim_fname = @config_qimname;

%----------------------------------------------------
function cfg = config_oxford (cfg)
% Load groundtruth
%----------------------------------------------------
cfg.gnd_fname = [cfg.dir_data '/gnd_oxford5k.mat'];
load (cfg.gnd_fname); % Retrieve list of image names, ground truth and query numbers
cfg.imlist = imlist;
cfg.qimlist = imlist(qidx);
cfg.gnd = gnd;
cfg.qidx = qidx;
cfg.n = length (cfg.imlist);   % number of database images
cfg.nq = length (cfg.qidx);    % number of query images

%----------------------------------------------------
function cfg = config_paris (cfg)
% Load groundtruth
%----------------------------------------------------
cfg.gnd_fname = [cfg.dir_data '/gnd_paris6k.mat'];
load (cfg.gnd_fname); % Retrieve list of image names, ground truth and query numbers
% Specific variables to handle paris's groundtruth
cfg.imlist = imlist;
cfg.qimlist = imlist(qidx);
cfg.gnd = gnd;
cfg.qidx = qidx;
cfg.n = length (cfg.imlist);   % number of database images
cfg.nq = length (cfg.qidx);    % number of query images

%----------------------------------------------------
function cfg = config_roxford (cfg)
% Load groundtruth
%----------------------------------------------------
cfg.gnd_fname = [cfg.dir_data '/gnd_roxford5k.mat'];
load (cfg.gnd_fname); % Retrieve list of image names, ground truth and query numbers
cfg.imlist = imlist';
cfg.qimlist = qimlist';
cfg.gnd = gnd;
cfg.n = length (cfg.imlist);   % number of database images
cfg.nq = length (cfg.qimlist);    % number of query images

%----------------------------------------------------
function cfg = config_rparis (cfg)
% Load groundtruth
%----------------------------------------------------
cfg.gnd_fname = [cfg.dir_data '/gnd_rparis6k.mat'];
load (cfg.gnd_fname); % Retrieve list of image names, ground truth and query numbers
cfg.imlist = imlist';
cfg.qimlist = qimlist';
cfg.gnd = gnd;
cfg.n = length (cfg.imlist);   % number of database images
cfg.nq = length (cfg.qimlist);    % number of query images

%----------------------------------------------------
%----------------------------------------------------

function cfg = config_revisitop1m (cfg)
  % load image list
  cfg.imlist_fname = [cfg.dir_data '/revisitop1m.txt'];
  cfg.imlist = textread(cfg.imlist_fname,'%s');
  cfg.n = length (cfg.imlist);   % number of images
  
function fname = config_imname (cfg, i)
%----------------------------------------------------
[~, ~, ext] = fileparts(cfg.imlist{i});
if isempty(ext)
    fname = sprintf ('%s/%s%s', cfg.dir_images, cfg.imlist{i}, cfg.ext);
else
    fname = sprintf ('%s/%s%s',cfg.dir_images, cfg.imlist{i});
end

%----------------------------------------------------
function fname = config_qimname (cfg, i)
%----------------------------------------------------
[~, ~, ext] = fileparts(cfg.qimlist{i});
if isempty(ext)
    fname = sprintf ('%s/%s%s', cfg.dir_images, cfg.imlist{i}, cfg.ext);
else
    fname = sprintf ('%s/%s%s',cfg.dir_images, cfg.imlist{i});
end
%--------------------------------------------------------

