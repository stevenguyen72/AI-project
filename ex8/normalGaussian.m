function p = normalGaussian(X,mu,sigma2)
%normal gaussian wrote by Steven

p = prod((sqrt(2*pi*sigma2)).^(1/2) .* exp((-(X - mu).^2) ./ (2 * sigma2)));



end