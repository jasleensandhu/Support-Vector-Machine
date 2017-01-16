function sim = gaussianKernel(x1, x2, sigma)
% Ensure that x1 and x2 are column vectors

xny         =   x1-x2;
Normxny    =   xny'*xny;
sim         =   exp(-Normxny/(2*sigma^2));

end