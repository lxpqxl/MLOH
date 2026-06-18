function net = mergeBatchNormRes50(net)

if exist('net','var')

    lgraph = layerGraph(net);

else

lgraph = layerGraph(resnet50);

lgraph = removeLayers(lgraph,'avg_pool');

lgraph = removeLayers(lgraph,'fc1000');

lgraph = removeLayers(lgraph,'fc1000_softmax');

lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');

end
lgraph = merge_bn(lgraph);

net = dlnetwork(lgraph);

%%
    function net = merge_bn(net)

    Names_idx = zeros(1,500);

    j = 0;

    %get the index of BNLayer
    for i = 1:length(net.Layers)
        layerName = net.Layers(i).Name;
        if contains(layerName, "bn")
            j = j+1;
            Names_idx(1,j) = i;
        end
    end

    %if no BNLayer return
    if j ==0
        return;
    end

    Names = cell(1,j);

    for i = 1:j
        Names{1,i} = net.Layers(Names_idx(1,i)).Name;
    end

    i = 0;

    %merge into previous conv layer
    fprintf('merge into previous conv layer:\n')

    for name = Names
        i = i+1;

        Name = char(name);
        layer = net.Layers(Names_idx(1,i));
        [prename,prelayer] = getpre(net, Name);%get the layer before BNLayer
        filters = prelayer.Weights;%preLayer Weights
        biases = prelayer.Bias;    %preLayer Bias
        multipliers = layer.Scale; %BNLayer Scale
        offsets = layer.Offset;    %BNLayer Offset
        TrainedMean = layer.TrainedMean;%BNLayer TM
        TrainedVariance = layer.TrainedVariance;%BNlayer TV
        % merge process
        [filtersValue, biasesValue] = mergeBN(filters ...
            , biases ...
            , multipliers ...
            , offsets ...
            , TrainedMean ...
            , TrainedVariance);
        prelayer.Weights = filtersValue;
        prelayer.Bias = biasesValue;
        net = replaceLayer(net,prename,prelayer);
  
    end

    % Remove Batch Norm layers, they are already merged
    fprintf('Remove Batch Norm layers:\n')
 
    for i = 1:length(Names)
        [prename,~] = getpre(net, Names{1,i});
        nextname = getnextname(net, Names{1,i});
        net = removeLayers(net,Names{1,i});
        net = connectLayers(net,prename,nextname);
      
    end


    %%
        function [filtersValue, biasesValue] = mergeBN(filters, biases, multipliers, offsets, TrainedMean, TrainedVariance)
            TrainedMean = squeeze(TrainedMean);
            TrainedVariance = squeeze(TrainedVariance);
            offsets = squeeze(offsets);
            multipliers = squeeze(multipliers);
            sz_b = size(biases);
            biases = squeeze(biases);

            a = multipliers(:)./ (TrainedVariance).^(1/2);
            b = offsets(:) - TrainedMean.*a;
            biasesValue(:) = biases(:) + b(:);
            biasesValue = reshape(biasesValue,sz_b);
            sz_w = size(filters);
            numFilters = sz_w(4);
            filtersValue = reshape(bsxfun(@times, reshape(filters, [], numFilters), a'), sz_w) ;

        end

        function [prename,prelayer] = getpre(net, name)
            for m = 1:height(net.Connections)
                if any(strcmp(net.Connections.Destination(m), name))
                    prename = char(net.Connections.Source(m));
                    break;
                end
            end
            for m = 1:length(net.Layers)
                if any(strcmp(net.Layers(m).Name, prename))
                    prelayer = net.Layers(m);
                    break;
                end
            end
        end

        function nextname = getnextname(net, name)
            for m = 1:height(net.Connections)
                if any(strcmp(net.Connections.Source(m), name))
                    nextname = char(net.Connections.Destination(m));
                    break;
                end
            end
        end

    end

end