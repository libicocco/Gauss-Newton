%%************************************************************************
%% December, 2007
%%
%% This is a Matlab implementation of a coordinate gradient descent method
%% for solving the L_1 regularized least squares problem
%%
%%       min_x  0.5*norm(y-Ax,2)^2 + tau*norm(x,1) 
%%
%% The method, for a given x, computes a direction d as the solution of
%% the subproblem
%%
%%    min   g'*d + 0.5*d'*H*d + tau*norm(x+d,1)
%%    subject to d(j)=0 for j not in J,
%%
%% where g = -A'(y-Ax), H is a positive definite diagonal matrix,
%% and J is a nonempty subset of {1,...,n}.  
%% Then x is updated by
%%       x = x + step*d,
%% and this is repeated until a termination criterion is met.
%% The index subset J is chosen by a Gauss-southwell-d rule
%% or Gauss-southwell-q rule
%% (in the M-files dirsd.m, dirsq.m)
%%
%%===== Required inputs =============
%%
%%  y: 1D vector
%%
%%  A: k*n (where k is the size of y and n the size of x)
%%     matrix or a handle to a function that computes
%%     products of the form A*v, for some vector v.
%%     A has to be passed as a handle to a function which computes
%%     products of the form A*x; another handle to a function
%%     AT which computes products of the form A'*x is also required
%%     in this case. The size of x is determined as the size
%%     of the result of applying AT.
%%
%%  n: the size of x
%%
%%  tau: a non-negative real parameter of the objective function (see above).
%%
%%===== Optional inputs =============
%%
%%
%%  'AT'    = function handle for the function that implements
%%            the multiplication by the conjugate of A, when A
%%            is a function handle. If A is an array, AT is ignored.
%%
%%  'StopCriterion' = type of stopping criterion to use
%%                    0 = stop when norm(Hd_H(x),inf)
%%                        falls below 'Tolerance'
%%                    1 = algorithm stops when the relative
%%                        change in the number of non-zero
%%                        components of the estimate falls
%%                        below 'Tolerance'
%%                    2 = stop when the relative
%%                       change in the objective function
%%                       falls below 'Tolerance'
%%                    3 = stop when the relative change in the norm of
%%                        the estimated solution falls below 'Tolerance'
%%                    4 = stop when the objective function
%%                        becomes equal or less than Tolerance.
%%                    Default = 0.
%%
%%  'ChooseH' = choose a matrix H for directional subproblem
%%                    0 = H is the identity matrix.
%%                    1 = H is scalar multiple of the identity matrix. 
%%                    2 = positive approximation of Hessian diagonal
%%                    Default = 1.
%%
%%  'Steprule' = choose a stepsize rule.
%%                    0 = minimization rule.
%%                    1 = minimization rule using previous stepsize.
%%                    2 = limited minimization rule by fixed number of
%%                    beakpoints. 
%%                    Default = 0.
%%
%%  'Tolerance' = stopping threshold; Default = 1e-3.
%%
%%  'Maxiter' = maximum number of iterations allowed in the
%%              main phase of the algorithm.
%%              Default = 1000
%%
%%  'xtrue' = if the xtrue underlying x is passed in
%%            this argument, MSE plots are generated.
%%
%%  'Verbose'  = work silently (0) or verbosely (1)
%%
%%============ Outputs ==============================
%%   x = solution of the main algorithm
%%
%%   objective = sequence of values of the objective function
%%
%%   ttime = CPU time after each iteration
%%
%%   mse = sequence of MSE values, with respect to Xtruex,
%%          if it was given; if it was not given, mse is empty,
%%          mse = [].
%%************************************************************************

   function [x,objective,ttime,mse] = CGD_exactline(y,A,AT,tau,par)

   if (nargin < 4)
      error('Wrong number of required parameters');
   end
   if (nargin < 5); par = []; end
   dir = 'sd';
   stopCriterion = 0;
   chooseH  = 1;
   steprule = 0;
   tol      = 1e-3;
   maxiter  = 1000;
   init     = 0;
   verbose  = 1;
   mse      = [];
   comp_mse = 0;
   randnstate = randn('state');
   randn('state',0);
   rand('state',0);
   if isfield(par,'Direction');     dir = par.Direction; end
   if isfield(par,'StopCriterion'); stopCriterion = par.StopCriterion; end
   if isfield(par,'ChooseH');       chooseH = par.ChooseH; end
   if isfield(par,'Steprule');      steprule = par.Steprule; end
   if isfield(par,'Tolerance');     tol = par.Tolerance; end
   if isfield(par,'Maxiter');       maxiter = par.Maxiter; end
   if isfield(par,'Init');          init = par.Init; end
   if isfield(par,'x0');            x0 = par.x0; end
   if isfield(par,'xtrue');         xtrue = par.xtrue; comp_mse = 1; end
   if isfield(par,'Verbose');       verbose = par.Verbose; end
   t0  = cputime;
%%
%% If A is a function handle, we have to check presence of AT.
%% If A is a matrix, we find out dimensions of y and x,
%% and create function handles for multiplication by A and A',
%% so that the code below doesn't have to distinguish between
%% the handle/not-handle cases.
%%
   if isa(A, 'function_handle') & ~isa(AT,'function_handle')
      error(['The function handle for transpose of A is missing']);
   end
   if ~isa(A, 'function_handle')
      Amat = A;  ATmat = A';
      A  = @(x) Amat*x;
      %%AT = @(x) (x'*Amat)';
      AT = @(x) ATmat*x;
   end
   xtmp  = AT(zeros(size(y,1),size(y,2))); 
   [n,p] = size(xtmp); 
%%   
%% Initialization
%%
   switch init
     case 0   
        x = xtmp; 
     case 1   
        x = randn(n,p); 
     case 2   
        x = AT(y); 
     case 33333
       x = x0;
       if (size(x) ~= [n,p])
          error(['Size of initial x is not compatible with A']);
       end 
     otherwise
        error(['Unknown ''Initialization'' option']);
   end
   if (comp_mse) & (size(xtrue) ~= size(x))
      error(['Initial x has incompatible size']);
   end
%%
%% nzx = indicator vector of x
%% Compute and store initial value of the objective function
%%
   nzx = spones(x);
   num_nzx = nnz(nzx);
   if (p==1)
      resid = y - A(sparse(x));
   else
      resid = y - A(x);
   end
   f = 0.5*norm(resid,'fro')^2 + tau*L1norm(x);
   objective(1) = f;
   ttime(1)     = 0; 
   if (comp_mse); mse(1) = norm(x-xtrue,'fro'); end
   if strcmp(dir,'sq')
      ups  = 0.5; 
   elseif strcmp(dir,'sd')
      ups  = 0.9; 
   end
   step = 1; 
%%
%% Compute the diagonal of Hessian
%%
   if (chooseH == 2) & (p==1)
      dhess = zeros(n,1);
      for i = 1:n; dhess(i) = norm(Amat(:,i))^2; end
      h = min(max(dhess,1e-10),1e10);
      alpha = -1; 
   else
      xrand = randn(n,p); xrand = xrand/norm(xrand,'fro');
      Ad = A(xrand);
      alpha = norm(Ad,'fro')^2;
      h = alpha*ones(n*p,1);
   end
   timeh = cputime-t0;
   fprintf('  chooseH = %2.0d,  dhess time = %4.2e',chooseH,timeh);  
   if (verbose)
      fprintf('\n  iter   objective     step    nzx  ups    relmaxd    alpha');
      fprintf('\n %3.0f    %10.6e   %3.2f  %5d  %3.2f   %5.2e  %5.2e',...
      0,f,1,num_nzx,ups,1,alpha); 
   end
%%
%% initial gradient   
%%
   g = -AT(resid);
%%
%% Main loop
%%
   for iter = 1:maxiter
      
      if (p==1); 
         xvec = x; gvec = g;
      else
         xvec = x(:); gvec = g(:); 
      end
      if strcmp(dir,'sd')
         [maxd,dvec,nonx] = dirsd(tau,xvec,gvec,h,ups);
      elseif strcmp(dir,'sq')
         [maxd,dvec,nonx] = dirsq(tau,xvec,gvec,h,ups);
      end 
      if (p==1); 
         d = dvec; 
      else 
         d = reshape(dvec,n,p); 
      end
      relmaxd = maxd/max(1,max(max(abs(x)))); 
      if (stopCriterion == 0)
         if (relmaxd < tol); break; end
      end
      if (p==1)
         Ad = A(sparse(d));
      else
         Ad = A(d);
      end
      if (steprule == 0)
         step = findstep(Ad,resid,nonx,xvec,dvec,tau);
      elseif (steprule == 1)
         step = findstep1(step,Ad,resid,nonx,xvec,dvec,tau);
      else
         step = findsteplim2(Ad,resid,nonx,xvec,dvec,tau);
      end
      fprev = f;
      xprev = x;
      nzx_prev = nzx;
      x = x + step*d;
      resid = resid - step*Ad;
      f = 0.5*norm(resid,'fro')^2 + tau*L1norm(x);
      nzx = spones(x);
      num_nzx = nnz(nzx);
      
      gprev = g;
      g = -AT(resid);
      if (chooseH == 0)
          alpha = (g-gprev)'*sparse(d)/(step*d'*sparse(d));
          alpha = min(max(alpha,1e-10),1e10);
          h = alpha*ones(n*p,1);
      end
      
%%
%% Update the threshold for choosing J, based on the current stepsize.
%%    
      if (step > 10)
         ups = max(1e-2,0.80*ups); %% old: ups = max(1e-2,0.50*ups);
         if (chooseH ~= 0)
             alpha = max(alpha/step,1);
             h = alpha*ones(n*p,1);
         end
      elseif (step > 1.0)
         ups = max(1e-2,0.90*ups); %% old: ups = max(1e-2,0.5*ups);
      elseif (step > 0.5)          %% old: not present
         ups = max(1e-2,0.98*ups);       
      elseif (step < 0.1)
         ups = min(0.2,2*ups);     %%  ups = min(0.2,10*ups);
         if (chooseH ~= 0)
             alpha = min(alpha/step,1);
             h = alpha*ones(n*p,1);
         end
      end
%%
%% print results
%%
      objective(iter+1) = f;
      ttime(iter+1) = cputime-t0;
      steps(iter+1) = step;
      if (comp_mse); mse(iter) = norm(x-xtrue,'fro'); end
      if (verbose)
         fprintf('\n %3.0d    %10.6e   %3.2f  %5.0d  %3.2f   %5.2e  %5.2e',...
         iter,f,step,num_nzx,ups,relmaxd,alpha);
      end
      if (stopCriterion == 1)
         num_changes_active = L1norm(nzx-nzx_prev);
         if (num_nzx >= 1)
            criterionActiveSet = num_changes_active / num_nzx;
         else
            criterionActiveSet = 1.0;
         end
         if (criterionActiveSet < tol); break; end
      elseif (stopCriterion == 2) 
         criterionObjective = abs(f-fprev)/(fprev);
         if (criterionObjective < tol); break; end
      elseif (stopCriterion == 3) 
         criterionSolution = step*norm(d,'fro')/max(norm(xprev,'fro'),1);
         if (criterionSolution < tol); break; end
      elseif (stopCriterion == 4)      
         if (f < tol); break; end 
      end
   end 
%%
   if (verbose)
      fprintf('\n  Finished the main algorithm!\n')
      fprintf('  norm(Ax-y,2) = %10.3e\n',norm(resid,'fro'))
      fprintf('  norm(x,1)    = %10.3e\n',L1norm(x))
      fprintf('  Number of non-zero components = %d\n',num_nzx);
      fprintf('  Objective function = %10.3e\n',f);
      fprintf('  CPU time so far   = %10.3e\n', ttime(end));
      fprintf('  relative residual = %3.2e\n',full(relmaxd));
      fprintf('\n');
   end
%%************************************************************************
