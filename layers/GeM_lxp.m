classdef GeM_lxp < nnet.layer.Layer
    properties

        epsilon

    end

    properties(Learnable)

        p
        
    end

    methods

        function layer=GeM_lxp(Name)

            arguments
                Name
            end

            layer.Name = Name;

            layer.Description = "GEM";

            layer.epsilon = 1e-6;

            layer.p = 3;

            layer = layer.setLearnRateFactor('p',10);

        end
        
        function Z = predict(layer,X)

            N = 1/(size(X,1)*size(X,2));

            xp = power(X,layer.p);

            xpsum = max(sum(xp,[1,2]),layer.epsilon);

            Np = power(N,1./layer.p);

            Z = Np.*power(xpsum,1./layer.p);

        end

    end
end