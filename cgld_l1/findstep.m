
function [alpha,breakpts] = findstep(Ad,resid,nonz,x,d,tau)

%% September 26, 2007
%%
%% This routine compute the stepsize by finding the minimum of 
%% piecewise quadratic function
%%
%%  min {0.5*norm(resid-alp*Ad)^2 + tau*norm(x+alp*d,1) : alp > 0}
%%
%%  Inputs: 
%%
%%  nonz: indices of nonzero components of d
%%  x   : the current point 
%%  d   :  the current descent direction
%%  tau : a non-negative real parameter of the objective function
%%
%% Outputs: 
%%
%% alpha: the stepsize
%%========================================================

%% Nonzero component of vector d 
   nd = d(nonz);
   nx = x(nonz);

%% Reformulate the piecewise quadratic function as 
%% 0.5*a*alp^2 - b*alp + tau*sum_{i \in nonz} |x_i+alp*d_i| + c

   if (size(Ad,2) == 1)
      a = Ad'*Ad;
      b = Ad'*resid;
   else
      a = sum(sum(Ad.*Ad));
      b = sum(sum(Ad.*resid));
   end
   [breakpts,idx] = sort(-nx./nd);
   nx = nx(idx); nd = nd(idx); 
   normnd = L1norm(nd);
   idxpos = find(breakpts > 0);
   if isempty(idxpos) 
      if (a==0)
         fprintf('\n Something is wrong'); 
      else
         alpha = (b-tau*normnd)/a;
      end
   else         
      %% pick the first occurence of tied break points 
      %%
      len = length(idxpos); alpold = -1; count = 1; 
      idxpos2 = zeros(len,1); 
      for k = 1:len
         alp = breakpts(idxpos(k)); 
         if (alp-alpold > 1e-13)
            idxpos2(count) = idxpos(k); 
            count = count+1; 
         end
	 alpold = alp; 
      end
      %% find the first positive break-point with 
      %% increased objective value
      %%
      idxpos2 = idxpos2(find(idxpos2)); 
      len = length(idxpos2);
      objold = tau*L1norm(nx); 
      findone = 0;
      for k = 1:len
         alp = breakpts(idxpos2(k)); 
         objnew = (0.5*a)*alp.*alp - b*alp + tau*L1norm(nx+alp*nd);
         if (objnew > objold); findone = 1; break; end
         objold = objnew; 
      end
      %% find no positive break-point with 
      %% increased objective value
      if (findone == 0)
          if (a==0)
              alpha = alp;
          else
              tmp1 = normnd - 2*L1norm(nd(idxpos2(k):end));
              alp1 = (b-tau*tmp1)/a;
              if (alp1 > alp); 
                  alp1 = alp; f1 = objold; 
              else
                  f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*L1norm(nx+alp1*nd); 
              end
              alp2 = (b-tau*normnd)/a;
              if (alp2 < alp); 
                 alp2 = alp; f2 = objold; 
              else
                 f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*L1norm(nx+alp2*nd); 
              end
              [dummy,idx] = min([f1,f2,objold]);
              if (idx==1); alpha=alp1; 
              elseif (idx==2); alpha=alp2; 
              elseif (idx==3); alpha=alp; end
          end
      else
        if (k==1)
           if (a==0)
              fprintf('\n Something is wrong'); 
           else
              tmp = normnd - 2*L1norm(nd(idxpos2(1):end)); 
              alpha = (b-tau*tmp)/a; 
           end
        else
        alptest = breakpts(idxpos2(k-1)); 
           if (a==0)
              alpha = alptest; 
           else
    	      %% find the exact minimum in either the interval
              %% [breakpts(idxpos2(k-2)), alptest] or 
              %% [alptest, breakpts(idxpos2(k))]
              %%
              tmp1 = normnd - 2*L1norm(nd(idxpos2(k-1):end));
              alp1 = (b-tau*tmp1)/a;
              if (alp1 > alptest); 
                 alp1 = alptest; f1 = objold; 
              else
                 f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*L1norm(nx+alp1*nd); 
              end
              tmp2 = normnd - 2*L1norm(nd(idxpos2(k):end));
              alp2 = (b-tau*tmp2)/a;
              if (alp2 < alptest); 
                 alp2 = alptest; f2 = objold; 
              else
                 f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*L1norm(nx+alp2*nd); 
              end
              [dummy,idx] = min([f1,f2,objold]); 
              if (idx==1); alpha=alp1; 
              elseif (idx==2); alpha=alp2; 
              elseif (idx==3); alpha=alptest; end
           end
        end
      end
   end
%%==================================================================
  

