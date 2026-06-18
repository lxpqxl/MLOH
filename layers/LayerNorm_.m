classdef LayerNorm_ < nnet.layer.Layer

    properties

        epsilon

    end  
       
    methods

        function layer = LayerNorm_(Name)

            arguments

                Name

            end

            layer.Name = Name;

            layer.Description = "LayerNorm";

            layer.epsilon = 1e-5;

        end

        function Z = predict(layer,X)

            u = mean(X,3);

            s = mean((X - u).^2,3);

            Z = (X - u)./sqrt(s + layer.epsilon);

        end

    end
    
end