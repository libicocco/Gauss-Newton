function [maxhR,d,indx] = dirsqmin(c,x,g,h,ups)

% September 26, 2007
%
% This routine computes the direction d whose ith component equals
% that of (-q_H(x;1),...,-q_H(x;n) if -q_H(x;j) is greater than or equal to 
% upsilon*max_i{-q_H(x;i)}, otherwise equals zero.
% Here H is a diagonal positive definite matrix (or the identity matrix).
% It also outputs maxhR, which is norm(Hd_{H}(x),inf) and will be used
% for checking the termination.
%
%===== Inputs =============
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
%============ Outputs ==============================
%   maxhR: norm(Hd_H(x),inf)
%
%   d: a descent direction
%
%   nonx: indices of nonzero components of d
%========================================================

%% R = d_{H}(x)

  %%R = -median([ x' ; (g'+c)./h' ; (g'-c)./h' ]);  
  %%R = R'; 

  tmp = x - g./h; 
  R = sign(tmp).*max(abs(tmp)-c./h,0) - x; 
  hR = h.*R;
  Q = -g.*R-0.5*R.*hR-c*abs(x+R)+c*abs(x);
  maxhR = max(abs(hR)); 
%%
%% max_i{-q_H(x;i)}
%% set d(i)=R(i) if Q(i) > ups*maxQ
%%
  maxQ = max(Q);
  indx = find(Q > ups*maxQ);
  d = zeros(length(R),1);
  d(indx) = R(indx); 
%========================================================
