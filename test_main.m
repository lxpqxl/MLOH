clear;
tic;
muti_scales = 1;
scales = [1,1/sqrt(2),sqrt(2)];
%%  resnet50
load('resnet50-epoch90.mat', 'net')
try
    load('resnet50-epoch90-Lw.mat', 'Lw')
catch
    Lw = get_Lw(net,muti_scales);
end
test_net_(net,scales,Lw);
%%  resnet101
load('resnet50-epoch101.mat', 'net')
try
    load('resnet101-epoch90-Lw.mat', 'Lw')
catch
    Lw = get_Lw(net,muti_scales);
end
test_net_(net,scales,Lw);
