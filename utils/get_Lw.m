function Lw = get_Lw(net,muti_scales)


if muti_scales==1

    im_scales = [1,1/sqrt(2),sqrt(2)];

else

    im_scales = 1;

end

test_dim = 1024;

DIM = 2048;

dir_main = pwd;

train_whiten_file = [dir_main,'/retrieval-SfM-120k-whiten.mat']; % more images, better results but slower

ims_whiten_dir = [dir_main,'/ims'];

train_whiten = load(train_whiten_file);

cids  = train_whiten.cids;

qidxs = train_whiten.qidxs; % query indexes

pidxs = train_whiten.pidxs; % positive indexes

% learn whitening
fprintf('>> whitening: Extracting CNN descriptors for training images...\n');

%%

tic

vecs_whiten = cell(1, numel(cids));

count = 0;

for i = 1:numel(cids)

    imgPath = cid2filename(cids{i}, ims_whiten_dir);

    feature = features_process_(imgPath,test_dim,DIM,net,1,im_scales,4)';

    vecs_whiten{i} = feature;

    fprintf(repmat('\b',1,count));

    count = fprintf('>> vecs: %d/%d done...\n',i,numel(cids));

end

toc;

vecs_whiten = cell2mat(vecs_whiten);

fprintf('>> whitening: Learning...\n');

Lw = whitenlearn(vecs_whiten, qidxs, pidxs);

end

