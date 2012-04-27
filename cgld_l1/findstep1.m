
function [alpha,breakpts] = findstep1(step,Ad,resid,nonz,x,d,tau)

%% December 7, 2007
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

   if (size(Ad,2)==1)
      a = Ad'*Ad;
      b = Ad'*resid;
   else
      a = sum(sum(Ad.*Ad));
      b = sum(sum(Ad.*resid));
   end   
   [breakpts,idx] = sort(-nx./nd);
   nx = nx(idx); nd = nd(idx); 
   normnd = norm(nd,1);
   %% breakpts that are greater than step
   stepg = find(breakpts>step);   
   %% breakpts that are greater than 0 less than step
   stepl = find((breakpts>0)&(breakpts<step));  
   %% breakpts that are equal to step
   stepf = find(breakpts==step);                
%%
   if (isempty(stepg) & isempty(stepl))
      if (isempty(stepf))
          %% no positive breakpoints  
          if (a==0)
             fprintf('\n 1.Something is wrong'); 
          else
             alpha = (b-tau*normnd)/a;
          end
      else
          %% there are only breakpts that are equal to step
          if (a==0)
             alpha = step;
          else
             objstep = (0.5*a)*step.*step - b*step + tau*norm(nx+step*nd,1); 
             tmp1 = normnd - 2*norm(nd(stepf(1):end),1);
             alp1 = (b-tau*tmp1)/a;
             if (alp1 > step); 
                 alp1 = step; f1 = objstep; 
             else
                 f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
             end
             alp2 = (b-tau*normnd)/a;
             if (alp2 < step); 
                 alp2 = step; f2 = objstep; 
             else
                 f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
             end
             [dummy,idx] = min([f1,f2,objstep]); 
             if (idx==1); alpha=alp1; 
             elseif (idx==2); alpha=alp2; 
             elseif (idx==3); alpha=step; end
          end          
      end
   else         
      %% there are positive breakpts. 
      objstep = (0.5*a)*step.*step - b*step + tau*norm(nx+step*nd,1);
      if isempty(stepg) %% no breakpts that are greater than step
          %% pick the first occurence of tied break points 
          %%
          len = length(stepl); alpold = -1; count = 1; 
          idxpos2 = zeros(len,1); 
          for k = 1:len
              alp = breakpts(stepl(k)); 
              if (alp-alpold > 1e-13)
                  idxpos2(count) = stepl(k); 
                  count = count+1; 
              end
	          alpold = alp; 
          end
          %% find the first positive break-point with 
          %% increased objective value in reverse order
          %%
          idxpos2 = idxpos2(find(idxpos2)); 
          len = length(idxpos2);
          objold = objstep; 
          findone = 0;
          for k = 1:len
              alp = breakpts(idxpos2(len-k+1)); 
              objnew = (0.5*a)*alp.*alp - b*alp + tau*norm(nx+alp*nd,1);
              if (objnew > objold); findone = 1; break; end
              objold = objnew; 
          end
          if (findone == 0) %% no positive break-points with increased objective value
              if (a==0)
                  alpha = alp;
              else
                  if (k==1) %% only one positive break-point
                      if isempty(stepf) %% no breakpts that are equal to step
                          %% find the exact minimum in either the interval
                          %% (0, alp] or [alp, infty)
                          %%
                          tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                          alp1 = (b-tau*tmp1)/a;
                          if (alp1 > alp); 
                              alp1 = alp; f1 = objold; 
                          else
                              f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                          end
                          alp2 = (b-tau*normnd)/a;
                          if (alp2 < alp); 
                              alp2 = alp; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                          [dummy,idx] = min([f1,f2,objold]);
                          if (idx==1); alpha=alp1; 
                          elseif (idx==2); alpha=alp2; 
                          elseif (idx==3); alpha=alp; end
                      else  %% there are breakpts that are equal to step
                          %% find the exact minimum in either the interval
                          %% (0, alp] or [alp, step]
                          %%
                          tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                          alp1 = (b-tau*tmp1)/a;
                          if (alp1 > alp); 
                              alp1 = alp; f1 = objold; 
                          else
                              f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                          end
                          tmp2 = normnd - 2*norm(nd(stepf(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 < alp); 
                              alp2 = alp; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                          [dummy,idx] = min([f1,f2,objold]);
                          if (idx==1); alpha=alp1; 
                          elseif (idx==2); alpha=alp2; 
                          elseif (idx==3); alpha=alp; end
                      end                      
                  else %% more than 2 positive break-points
                      %% find the exact minimum in either the interval
                      %% (0, alp] or [alp, breakpts(idxpos2(2))]
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 < alp); 
                          alp1 = alp; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      tmp2 = normnd - 2*norm(nd(idxpos2(1):end),1);
                      alp2 = (b-tau*tmp2)/a;
                      if (alp2 > alp); 
                          alp2 = alp; f2 = objold; 
                      else
                          f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                      end
                      [dummy,idx] = min([f1,f2,objold]);
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alp; end
                  end
              end
          else %% find a positive break-point with increased objective value
              if (k==1) 
                  if (a==0)
                      if isempty(stepf) %% no breakpts that are equal to step
                          fprintf('\n 2.Something is wrong');
                      else %% there are breakpts that are equal to step
                          alpha = step;
                      end
                  else
                      if isempty(stepf) %% no breakpts that are equal to step
                          %% find the exact minimum in the interval [alp, infty)
                          %%
                          tmp = normnd; 
                          alpha = (b-tau*tmp)/a;
                          if (alp>alpha)
                              fprintf('\n 3.Something is wrong'); 
                          end
                      else %% there are breakpts that are equal to step
                          %% find the exact minimum in either the interval
                          %% [alp, step] or [step, infty)
                          %%
                          tmp1 = normnd - 2*norm(nd(stepf(1):end),1);
                          alp1 = (b-tau*tmp1)/a;
                          if (alp1 > step); 
                              alp1 = step; f1 = objstep; 
                          else
                              f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                          end
                          alp2 = (b-tau*normnd)/a;
                          if (alp2 < step); 
                              alp2 = step; f2 = objstep; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                          [dummy,idx] = min([f1,f2,objstep]);
                          if (idx==1); alpha=alp1; 
                          elseif (idx==2); alpha=alp2; 
                          elseif (idx==3); alpha=step; end
                      end
                  end
              elseif (k==2)
                  alptest = breakpts(idxpos2(end-k+2)); 
                  if (a==0)
                      alpha = alptest; 
                  else
    	              %% find the exact minimum in either the interval
                      %% [breakpts(idxpos2(end-k+1)), alptest] or 
                      %% ([alptest, step] or [alptest, infty))
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(end-k+2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 > alptest); 
                          alp1 = alptest; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      if isempty(stepf) %% no breakpts that are equal to step
                          alp2 = (b-tau*normnd)/a;
                          if (alp2 < alptest); 
                              alp2 = alptest; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                      else %% there are breakpts that are equal to step
                          tmp2 = normnd - 2*norm(nd(stepf(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 < alptest); 
                              alp2 = alptest; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                      end
                      [dummy,idx] = min([f1,f2,objold]); 
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alptest; end
                  end
              else
                  alptest = breakpts(idxpos2(end-k+2)); 
                  if (a==0)
                      alpha = alptest;
                  else
    	              %% find the exact minimum in either the interval
                      %% [breakpts(idxpos2(end-k+1)), alptest] or 
                      %% [alptest, breakpts(idxpos2(end-k+3))]
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(end-k+2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 > alptest); 
                          alp1 = alptest; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      tmp2 = normnd - 2*norm(nd(idxpos2(end-k+3):end),1);
                      alp2 = (b-tau*tmp2)/a;
                      if (alp2 < alptest); 
                          alp2 = alptest; f2 = objold; 
                      else
                          f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                      end
                      [dummy,idx] = min([f1,f2,objold]); 
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alptest; end
                  end
              end
          end
      elseif isempty(stepl) %% no breakpts that are greater than 0 less than step
          %% pick the first occurence of tied break points 
          %%
          len = length(stepg); alpold = -1; count = 1; 
          idxpos2 = zeros(len,1); 
          for k = 1:len
              alp = breakpts(stepg(k)); 
              if (alp-alpold > 1e-13)
                  idxpos2(count) = stepg(k); 
                  count = count+1; 
              end
	          alpold = alp; 
          end
          %% find the first positive break-point with 
          %% increased objective value
          %%
          idxpos2 = idxpos2(find(idxpos2)); 
          len = length(idxpos2);
          objold = objstep; 
          findone = 0;
          for k = 1:len
              alp = breakpts(idxpos2(k)); 
              objnew = (0.5*a)*alp.*alp - b*alp + tau*norm(nx+alp*nd,1);
              if (objnew > objold); findone = 1; break; end
              objold = objnew; 
          end
          if (findone == 0) %% no positive break-points with increased objective value
              if (a==0)
                  alpha = alp;
              else
                  %% find the exact minimum in either the interval
                  %% [breakpts(idxpos2(k-1)), alp] or [alp, infty)
                  %%
                  tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                  alp1 = (b-tau*tmp1)/a;
                  if (alp1 > alp); 
                      alp1 = alp; f1 = objold; 
                  else
                      f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                  end
                  alp2 = (b-tau*normnd)/a;
                  if (alp2 < alp); 
                      alp2 = alp; f2 = objold; 
                  else
                      f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                  end
                  [dummy,idx] = min([f1,f2,objold]);
                  if (idx==1); alpha=alp1; 
                  elseif (idx==2); alpha=alp2; 
                  elseif (idx==3); alpha=alp; end
              end
          else
              if (k==1)
                  if isempty(stepf) %% no breakpts that are equal to step
                      if (a==0)
                          fprintf('\n 4.Something is wrong'); 
                      else
                          %% find the exact minimum in either the interval
                          %% (0, alp]
                          %%
                          tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                          alpha = (b-tau*tmp1)/a;
                          if (alpha > alp); 
                              fprintf('\n 5.Something is wrong'); 
                          end
                      end
                  else %% there are breakpts that are equal to step
                      obj1 = tau*norm(nx,1);
                      if (obj1<=objstep)
                          if (a==0)
                              fprintf('\n 6.Something is wrong'); 
                          else
                              %% find the exact minimum in the interval
                              %% (0, step]
                              %%
                              tmp1 = normnd - 2*norm(nd(stepf(1):end),1);
                              alpha = (b-tau*tmp1)/a;
                              if (alpha > step); 
                                  fprintf('\n 7.Something is wrong'); 
                              end
                          end
                      else
                          if (a==0)
                              alpha = step;
                           else
                              %% find the exact minimum in either the interval
                              %% (0, step], [step, breakpts(idxpos2(k)), alptest] or 
                              %% ([alptest, step] or [alptest, infty))
                              %%
                              tmp1 = normnd - 2*norm(nd(stepf(1):end),1);
                              alp1 = (b-tau*tmp1)/a;
                              if (alp1 > step); 
                                  alp1 = step; f1 = objstep; 
                              else
                                  f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                              end
                              tmp2 = normnd - 2*norm(nd(idxpos2(k):end),1);
                              alp2 = (b-tau*tmp2)/a;
                              if (alp2 < step); 
                                  alp2 = step; f2 = objstep; 
                              else
                                  f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                              end
                              [dummy,idx] = min([f1,f2,objold]);
                              if (idx==1); alpha=alp1; 
                              elseif (idx==2); alpha=alp2; 
                              elseif (idx==3); alpha=step; end
                          end
                      end
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
                      tmp1 = normnd - 2*norm(nd(idxpos2(k-1):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 > alptest); 
                          alp1 = alptest; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      tmp2 = normnd - 2*norm(nd(idxpos2(k):end),1);
                      alp2 = (b-tau*tmp2)/a;
                      if (alp2 < alptest); 
                          alp2 = alptest; f2 = objold; 
                      else
                          f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                      end
                      [dummy,idx] = min([f1,f2,objold]); 
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alptest; end
                  end
              end
          end
      else
        salpl = breakpts(stepl(end));
        objstepl = (0.5*a)*salpl.*salpl - b*salpl + tau*norm(nx+salpl*nd,1);
        salpg = breakpts(stepg(1));
        objstepg = (0.5*a)*salpg.*salpg - b*salpg + tau*norm(nx+salpg*nd,1);
        if (objstepl<objstep)
          %% pick the first occurence of tied break points 
          %%
          len = length(stepl); alpold = -1; count = 1; 
          idxpos2 = zeros(len,1); 
          for k = 1:len
              alp = breakpts(stepl(k)); 
              if (alp-alpold > 1e-13)
                  idxpos2(count) = stepl(k); 
                  count = count+1; 
              end
	          alpold = alp; 
          end
          %% find the first positive break-point with 
          %% increased objective value
          %%
          idxpos2 = idxpos2(find(idxpos2)); 
          len = length(idxpos2);
          objold = objstep; 
          findone = 0;
          for k = 1:len
              alp = breakpts(idxpos2(len-k+1)); 
              objnew = (0.5*a)*alp.*alp - b*alp + tau*norm(nx+alp*nd,1);
              if (objnew > objold); findone = 1; break; end
              objold = objnew; 
          end
          if (findone == 0) %% no positive break-points with increased objective value
              if (a==0)
                  alpha = alp;
              else
                  if (k==1)
                      if isempty(stepf) %% no breakpts that are equal to step
                          %% find the exact minimum in either the interval
                          %% (0, alp] or [alp, infty)
                          %%
                          tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                          alp1 = (b-tau*tmp1)/a;
                          if (alp1 > alp); 
                              alp1 = alp; f1 = objold; 
                          else
                              f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                          end
                          tmp2 = normnd - 2*norm(nd(stepg(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 < alp); 
                              alp2 = alp; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                          [dummy,idx] = min([f1,f2,objold]);
                          if (idx==1); alpha=alp1; 
                          elseif (idx==2); alpha=alp2; 
                          elseif (idx==3); alpha=alp; end
                      else %% there are breakpts that are equal to step
                          %% find the exact minimum in either the interval
                          %% (0, alp] or [alp, step]
                          %%
                          tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                          alp1 = (b-tau*tmp1)/a;
                          if (alp1 < alp); 
                              alp1 = alp; f1 = objold; 
                          else
                              f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                          end
                          tmp2 = normnd - 2*norm(nd(stepf(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 > alp); 
                              alp2 = alp; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                          [dummy,idx] = min([f1,f2,objold]);
                          if (idx==1); alpha=alp1; 
                          elseif (idx==2); alpha=alp2; 
                          elseif (idx==3); alpha=alp; end
                      end
                  else
                      %% find the exact minimum in either the interval
                      %% (0, alp] or [alp, breakpts(idxpos2(2))]
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 < alp); 
                          alp1 = alp; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      tmp2 = normnd - 2*norm(nd(idxpos2(1):end),1);
                      alp2 = (b-tau*tmp2)/a;
                      if (alp2 > alp); 
                          alp2 = alp; f2 = objold; 
                      else
                          f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                      end
                      [dummy,idx] = min([f1,f2,objold]);
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alp; end
                  end
              end
          else
              if (k==2)
                  alptest = breakpts(idxpos2(end-k+2)); 
                  if (a==0)
                      alpha = alptest;
                  else
    	              %% find the exact minimum in either the interval
                      %% [breakpts(idxpos2(end-k+1)), alptest] or 
                      %% ([alptest, step] or [alptest, infty))
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(end-k+2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 > alptest); 
                          alp1 = alptest; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      if isempty(stepf) %% no breakpts that are equal to step
                          tmp2 = normnd - 2*norm(nd(stepg(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 < alptest); 
                              alp2 = alptest; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                      else %% there are breakpts that are equal to step
                          tmp2 = normnd - 2*norm(nd(stepf(1):end),1);
                          alp2 = (b-tau*tmp2)/a;
                          if (alp2 < alptest); 
                              alp2 = alptest; f2 = objold; 
                          else
                              f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                          end
                      end
                      [dummy,idx] = min([f1,f2,objold]); 
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alptest; end
                  end
              else
                  alptest = breakpts(idxpos2(end-k+2)); 
                  if (a==0)
                      alpha = alptest;
                  else
    	              %% find the exact minimum in either the interval
                      %% [breakpts(idxpos2(end-k+1)), alptest] or 
                      %% [alptest, breakpts(idxpos2(end-k+3))]
                      %%
                      tmp1 = normnd - 2*norm(nd(idxpos2(end-k+2):end),1);
                      alp1 = (b-tau*tmp1)/a;
                      if (alp1 > alptest); 
                          alp1 = alptest; f1 = objold; 
                      else
                          f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                      end
                      tmp2 = normnd - 2*norm(nd(idxpos2(end-k+3):end),1);
                      alp2 = (b-tau*tmp2)/a;
                      if (alp2 < alptest); 
                          alp2 = alptest; f2 = objold; 
                      else
                          f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                      end
                      [dummy,idx] = min([f1,f2,objold]); 
                      if (idx==1); alpha=alp1; 
                      elseif (idx==2); alpha=alp2; 
                      elseif (idx==3); alpha=alptest; end
                  end
              end
          end
        elseif (objstepg<objstep)
          %% pick the first occurence of tied break points 
          %%
          len = length(stepg); alpold = -1; count = 1; 
          idxpos2 = zeros(len,1); 
          for k = 1:len
              alp = breakpts(stepg(k)); 
              if (alp-alpold > 1e-13)
                  idxpos2(count) = stepg(k); 
                  count = count+1; 
              end
	          alpold = alp; 
          end
          %% find the first positive break-point with 
          %% increased objective value
          %%
          idxpos2 = idxpos2(find(idxpos2)); 
          len = length(idxpos2);
          objold = objstep; 
          findone = 0;
          for k = 1:len
              alp = breakpts(idxpos2(k)); 
              objnew = (0.5*a)*alp.*alp - b*alp + tau*norm(nx+alp*nd,1);
              if (objnew > objold); findone = 1; break; end
              objold = objnew; 
          end
          if (findone == 0) %% no positive break-points with increased objective value
              if (a==0)
                  alpha = alp;
              else
                  %% find the exact minimum in either the interval
                  %% [breakpts(idxpos2(k-1)), alp] or [alp, infty)
                  %%
                  tmp1 = normnd - 2*norm(nd(idxpos2(k):end),1);
                  alp1 = (b-tau*tmp1)/a;
                  if (alp1 > alp); 
                      alp1 = alp; f1 = objold; 
                  else
                      f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                  end
                  alp2 = (b-tau*normnd)/a;
                  if (alp2 < alp); 
                      alp2 = alp; f2 = objold; 
                  else
                      f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                  end
                  [dummy,idx] = min([f1,f2,objold]);
                  if (idx==1); alpha=alp1; 
                  elseif (idx==2); alpha=alp2; 
                  elseif (idx==3); alpha=alp; end
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
                  tmp1 = normnd - 2*norm(nd(idxpos2(k-1):end),1);
                  alp1 = (b-tau*tmp1)/a;
                  if (alp1 > alptest); 
                      alp1 = alptest; f1 = objold; 
                  else
                      f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                  end
                  tmp2 = normnd - 2*norm(nd(idxpos2(k):end),1);
                  alp2 = (b-tau*tmp2)/a;
                  if (alp2 < alptest); 
                      alp2 = alptest; f2 = objold; 
                  else
                      f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                  end
                  [dummy,idx] = min([f1,f2,objold]); 
                  if (idx==1); alpha=alp1; 
                  elseif (idx==2); alpha=alp2; 
                  elseif (idx==3); alpha=alptest; end
              end
          end  
        else  
          if (a==0)
              alpha = step;
          else
              if isempty(stepf) %% no breakpts that are equal to step
                  %% find the exact minimum in the interval
                  %% [breakpts(stepl(end)), breakpts(stepg(1))]
                  %%  
                  tmp1 = normnd - 2*norm(nd(stepg(1):end),1);
                  alpha = (b-tau*tmp1)/a;
                  if (alpha > breakpts(stepg(1))); 
                      fprintf('\n 7.Something is wrong');  
                  end              
              else %% there are breakpts that are equal to step
                  %% find the exact minimum in either the interval
                  %% [breakpts(stepl(end)), step] or 
                  %% [step, breakpts(stepg(end))]
                  %% 
                  tmp1 = normnd - 2*norm(nd(stepf(1):end),1);
                  alp1 = (b-tau*tmp1)/a;
                  if (alp1 > step); 
                      alp1 = step; f1 = objstep; 
                  else
                      f1 = (0.5*a)*alp1.*alp1 - b*alp1 + tau*norm(nx+alp1*nd,1); 
                  end
                  tmp2 = normnd - 2*norm(nd(stepg(1):end),1);
                  alp2 = (b-tau*tmp2)/a;
                  if (alp2 < step); 
                      alp2 = step; f2 = objstep; 
                  else
                      f2 = (0.5*a)*alp2.*alp2 - b*alp2 + tau*norm(nx+alp2*nd,1); 
                  end
                  [dummy,idx] = min([f1,f2,objstep]); 
                  if (idx==1); alpha=alp1; 
                  elseif (idx==2); alpha=alp2; 
                  elseif (idx==3); alpha=step; end
              end
          end
        end
      end
   end
%%==================================================================
