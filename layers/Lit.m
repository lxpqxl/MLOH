classdef Lit < nnet.layer.Layer
    properties     

        T %number of iterations
        dim %dimension of SuperFeature
        scale
    
    end

    methods

        function layer = Lit(Name,T,dim)

            arguments

               Name

               T           

               dim   

            end

            layer.NumInputs = 3;

            layer.InputNames{1} = 'q';

            layer.InputNames{2} = 'k';

            layer.InputNames{3} = 'v';

            layer.T = T;

            layer.dim = dim;

            layer.scale = dim^(-0.5);

            layer.Name = Name;

            layer.Description = " Iterative Multi-layer Integration";
        end
        
        function Z = predict(layer,Q,K,V)

            [h,w,c,b] = size(Q);

            Q = reshape(Q,h*w,c,b);

            K = reshape(K,h*w,c,b);

            V = reshape(V,h*w,c,b);

            for i = 1:layer.T

                Q_prev = Q;

                Z = pagemtimes(Q,permute(K,[2,1,3]));

                D = Z*layer.scale;

                attn = softmax(permute(D,[1,3,2]),"DataFormat","SSCB");

                attn = permute(attn,[1,3,2]);

                attn = attn + 1e-8;

                % update Q

                Q = Q_prev + pagemtimes(attn,V); 

            end

            Z = reshape(Q,h,w,c,b);

        end

    end
    
end

