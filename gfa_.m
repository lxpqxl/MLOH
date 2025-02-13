function A = gfa_(f,p,tn)

sigma = 2*pi;

lambda = 3;

gamma = pi/4;

phi = 0;

epsilon = 1e-6;

[h,w,c] = size(f);

dy2 = 1.0 / (2 * sigma * sigma);

dx2 = 1.0 / (2 * sigma * sigma);

[x,y] = meshgrid(-1:1);

[h_, w_] = size(x);


gd = zeros(h_, w_,tn);

Z = (zeros(h,w,c,tn));

F_ = zeros(c,numel(lambda));

for lam = 1:numel(lambda)

    for n= 1:tn

        ori = pi*(n-1)/tn;

        x2 = x.* cos(ori) + y.* sin(ori);

        y2 = -x.* sin(ori) + y.* cos(ori);

        gd(:,:,n) = exp(-(x2.* x2.* dx2 + (gamma * gamma).* y2.* y2.* dy2)).* cos((2 * pi).* x2 ./ lambda(lam) + phi);

        Z_1 = convn( f, gd(:,:,n),'same');

        Z(:,:,:,n) = Z_1;

    end
    
    F = reshape(Z,[(h*w),c,tn]);
    
    F1 = max(F,[],3);
    
    F1 = F1 + epsilon;    
    
    F2 = sum(F,3); 
    
    F2 = F2 + epsilon;
    
    F3 = F./F2;
    
    F4 = (sum(F3.^2,3)).^(1/2);

    F5 = mean(F,3);
    
    F5 = F5 + epsilon;    

    a = 1./(1 + exp(-F4));

    F = a.*F1 + (1-a).*F5;

    F = reshape(F,[h,w,c]);

    F = sum(max(F,1e-6).^p,[1,2]).^(1/p);  

    F = squeeze(F);

    F_(:,lam) = F;

end

A  = (sum(F_.^1, 2)/size(F_,2))'.^(1/1);

end