%test the fine-tuning network
function  test_net_(net,im_scales,Lw)
tic;

warning off

dir_main = pwd;

%%
% max size of Im
test_dim = 1024;
% dim of feature
DIM = 2048;
%%
% test dataset {'roxford5k','rparis6k','oxford5k','paris6k'}
dataset = {'roxford5k','rparis6k','oxford5k','paris6k'};
%%
region = 1;

t_op = 0;

t_rop = 0;

tn = 4;

for t = 1:numel(dataset)
%% **********************get oxford/paris features***************************

    cfg = configdataset(dataset{t},dir_main);

    features = zeros(length(cfg.imlist),DIM);
 
    count = 0;

    for i = 1:cfg.n

        imgPath = cfg.im_fname(cfg,i);

        feature = features_process_(imgPath,test_dim,DIM,net,region,im_scales,tn);


        features(i,:) = feature;

        fprintf(repmat('\b',1,count));

        count = fprintf('>> vecs: %d/%d done...\n',i,cfg.n);

    end

    feature_normalize = features;
   
    %% **********************get query features***************************
    features = zeros(length(cfg.qimlist),DIM);

    for i = 1:length(cfg.qimlist)

        query_path = [cfg.dir_images,cfg.qimlist{i},'.jpg'];

        feature = features_process_(query_path,test_dim,DIM,net,region,im_scales,tn,cfg.gnd(i).bbx);

        features(i,:) = feature;

    end

    query_features = features;

%% ***********************pca_whitening & map********************************
    if size(feature_normalize,2) == 512
        cout = 7;%vgg
    elseif(size(feature_normalize,2) == 1024)
        cout = 8;%repvgg
    elseif(size(feature_normalize,2) == 1536)
        cout = 9;%ms-vgg
    elseif(size(feature_normalize,2) == 2048)
        cout = 10;%resnet
    else
        cout = 11;
    end 
    if contains(dataset{t},'rox')||contains(dataset{t},'rpa')
        map_e = zeros(1,cout);
        map_m = zeros(1,cout);
        map_h = zeros(1,cout);
    else
        map=zeros(1,cout);
    end   
    for m = 5:cout
        if m < 8
            dim = 8*2^(m-1);% 8 16 32 64 128 256 512 dimensions
        elseif(m == 8)
            dim = 1024;  %repvgg
        elseif(m == 9)
            dim = 1536; %ms-vgg
        elseif(m == 10)
            dim = 2048; %resnet
        else
            dim = 4096;
        end
        % LW_WHITEN
        if PcaW == 3

            feature_pca = whitenapply(feature_normalize', Lw.m, Lw.P,dim)';

            query_features_pca = whitenapply(query_features', Lw.m, Lw.P,dim)';
           
        end
        %% culculate the similarity and rank

        sim = feature_pca*query_features_pca';

        [~, ranks] = sort(sim, 'descend');
        
        %%
        % COMPUTE MAP
      
        if contains(dataset{t},'rox')||contains(dataset{t},'rpa')  

            t_rop = t_rop + 1;

            ks = [1, 5, 10];

            for i=1:length(cfg.qimlist)

            %easy (E)

                gnd_e(i).ok = (cfg.gnd(i).easy);

                gnd_e(i).junk = [(cfg.gnd(i).junk),(cfg.gnd(i).hard)];

            end

            [map_e(m), ~, ~, ~] = compute_map (ranks, gnd_e, ks);

            %easy & hard (M)
            for i=1:length(cfg.qimlist)

                gnd_m(i).ok = [(cfg.gnd(i).easy),(cfg.gnd(i).hard)];

                gnd_m(i).junk = (cfg.gnd(i).junk);

            end

            [map_m(m), ~, ~, ~] = compute_map (ranks, gnd_m, ks);

            %hard (H)

            for i=1:length(cfg.qimlist)

                gnd_h(i).ok = (cfg.gnd(i).hard);

                gnd_h(i).junk = [(cfg.gnd(i).junk),(cfg.gnd(i).easy)];

            end

            [map_h(m), ~, ~, ~] = compute_map (ranks, gnd_h, ks);  

            fprintf('>> %s: (%d)mAP E: %.2f, M: %.2f, H: %.2f\n', dataset{t}, dim, 100*map_e(m), 100*map_m(m), 100*map_h(m));

            map_ee(m-4,t_rop) = map_e(m);

            map_mm(m-4,t_rop) = map_m(m);

            map_hh(m-4,t_rop) = map_h(m);

        else     

            t_op = t_op + 1;

            for i = 1:length(cfg.qimlist)

                map(m) = compute_map (ranks, cfg.gnd);	

            end

            fprintf('>> %s: (%d)mAP: %.2f\n', dataset{t}, dim, 100*map(m));

            map_op(m-4,t_op) = map(m);
        end
    end

    toc;

end
