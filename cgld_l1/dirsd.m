function [maxhR,d,indx] = dirsd(c,x,g,h,ups)

% September 26, 2007
%
% This routine computes the direction d whose ith component equals
% that of d_H(x) if |d_H(x)_j| is greater than or equal to
% upsilon*||d_H(x)||_infty, otherwise equals zero.
% Here H is a diagonal positive definite matrix.
% It also outputs maxhR, which is norm(Hd_H(x),inf) and will be used
% for checking the termination.
%%===== Inputs =============
%
%  c: a non-negative constant
%
%  x: the current point 
%
%  g: the gradient at x
%
%  h: the diagonal of Hessian approximation
%
%  ups: a threshold for choosing J
%
%%============ Outputs ==============================
%   maxhR: norm(Hd_H(x),inf)
%
%   d: a descent direction
%
%   indx: indices of nonzero components of d
%%========================================================

%% R=d_{H}(x)

   %%R = -median([ x' ; (g'+c)./h' ; (g'-c)./h' ]); 
   %%R = R'; 

   tmp = x - g./h; 
   R = sign(tmp).*max(abs(tmp)-c./h,0) - x; 
   absR = abs(R);
   maxR = max(absR);
   maxhR = max(h.*absR);
%%
%% set d(i)=R(i) if |R(i)| > ups*maxR
%%
   indx = find(absR > ups*maxR);
   d = zeros(length(R),1);
   d(indx) = R(indx); 
%%========================================================
