function features = features_process_(ImPath,Test_Dim,Dim,net,r,Im_scales,tn,bbx)    

if ~exist("Im_scales","var")

    Im_scales = 1;

end

if ~exist("r","var")

    r = 0;

end

if Im_scales == 1

    p = 1;

else

    p = net.Learnables.Value{end,1};

    p = extractdata(p);

end

im = imread(ImPath);

if exist("bbx","var") 

    im = crop_qim(im, bbx, Test_Dim);

else

    im = imresizemaxd(im, Test_Dim ,0);

end

features = zeros(Dim,length(Im_scales));

for t = 1:length(Im_scales)

    im_ = imresize(im,Im_scales(t));
    
    if size(im_, 3) == 1

        im_ = repmat(im_, [1 1 3]);

    end
    
    im_dla = dlarray(single(im_),"SSCB");
    
    im_dla = gpuArray(im_dla);
    
    if r == 0

        feature = predict(net,im_dla);

        feature = single(gather(extractdata(feature)));

        feature = squeeze(feature);
    else

        feature = predict(net,im_dla,'Outputs','relu1');
        
        feature = single(gather(extractdata(feature))); 

        feature = squeeze(gfa_(feature,3,tn))';

    end

    features(:,t) = feature;

end

features = (sum(features.^p,2)/size(features,2)).^(1/p)';

features = normalize(features,2,"norm");

end