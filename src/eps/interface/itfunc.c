/*
      Interface EPS routines that the user calls.
*/

#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Level: advanced

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   This function sets a random initial vector if none has been provided.

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp()
@*/
int EPSSetUp(EPS eps)
{
  int         ierr;
  Vec         v0;
  Mat         A,B;
  PetscTruth  Ah,Bh;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (eps->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!eps->type_name) {
    ierr = EPSSetType(eps,EPSPOWER);CHKERRQ(ierr);
  }

  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  /* Set default problem type */
  if (!eps->problem_type) {
    ierr = SlepcIsHermitian(A,&Ah);CHKERRQ(ierr);
    if (B==PETSC_NULL) {
      if (Ah) { ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr); }
      else    { ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr); }
    }
    else {
      ierr = SlepcIsHermitian(B,&Bh);CHKERRQ(ierr);
      if (Ah && Bh) { ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr); }
      else          { ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr); }
    }
  }
  
  /* Check if the EPS initial vector has been set */
  ierr = EPSGetInitialVector(eps,&v0);CHKERRQ(ierr);
  if (!v0) {
    ierr = MatGetVecs(A,&v0,PETSC_NULL);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(v0);CHKERRQ(ierr);
    eps->vec_initial = v0;
  }
  ierr = STSetVector(eps->OP,v0); CHKERRQ(ierr);

  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);
  ierr = STSetUp(eps->OP); CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve"
/*@
   EPSSolve - Solves the eigensystem.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Options Database:
+   -eps_view - print information about the solver used
.   -eps_view_binary - save the matrices to the default binary file
-   -eps_plot_eigs - plot computed eigenvalues

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSDestroy(), EPSSetTolerances() 
@*/
int EPSSolve(EPS eps) 
{
  int         i,ierr;
  PetscReal   re,im;
  PetscTruth  flg;
  PetscViewer viewer;
  PetscDraw   draw;
  PetscDrawSP drawsp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  ierr = PetscOptionsHasName(eps->prefix,"-eps_view_binary",&flg);CHKERRQ(ierr); 
  if (flg) {
    Mat A,B;
    ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_BINARY_(eps->comm));CHKERRQ(ierr);
    if (B) ierr = MatView(B,PETSC_VIEWER_BINARY_(eps->comm));CHKERRQ(ierr);
  }

  /* reset the convergence flag from the previous solves */
  eps->reason = EPS_CONVERGED_ITERATING;

  if (!eps->setupcalled){ ierr = EPSSetUp(eps);CHKERRQ(ierr); }
  ierr = STResetNumberLinearIterations(eps->OP);
  ierr = PetscLogEventBegin(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);
  ierr = STPreSolve(eps->OP,eps);CHKERRQ(ierr);
  ierr = (*eps->ops->solve)(eps);CHKERRQ(ierr);
  ierr = STPostSolve(eps->OP,eps);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(EPS_Solve,eps,eps->V[0],eps->V[0],0);CHKERRQ(ierr);
  if (!eps->reason) {
    SETERRQ(1,"Internal error, solver returned without setting converged reason");
  }

  /* Map eigenvalues back to the original problem, necessary in some 
  * spectral transformations */
  ierr = (*eps->ops->backtransform)(eps);CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  /* reorder conjugate eigenvalues (positive imaginary first) */
  for (i=0; i<eps->nconv-1; i++) {
    PetscScalar minus = -1.0;
    if (eps->eigi[i] != 0) {
      if (eps->eigi[i] < 0) {
        eps->eigi[i] = -eps->eigi[i];
        eps->eigi[i+1] = -eps->eigi[i+1];
        ierr = VecScale(&minus, eps->V[i+1]); CHKERRQ(ierr);
      }
      i++;
    }
  }
#endif

  /* sort eigenvalues according to eps->which parameter */
  if (eps->nconv > 0) {
    ierr = PetscMalloc(sizeof(int)*eps->nconv, &eps->perm); CHKERRQ(ierr);
    ierr = EPSSortEigenvalues(eps->nconv, eps->eigr, eps->eigi, eps->which, eps->nconv, eps->perm); CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(eps->prefix,"-eps_view",&flg);CHKERRQ(ierr); 
  if (flg && !PetscPreLoadingOn) { ierr = EPSView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  ierr = PetscOptionsHasName(eps->prefix,"-eps_plot_eigs",&flg);CHKERRQ(ierr); 
  if (flg) { 
    ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,"Computed Eigenvalues",
                             PETSC_DECIDE,PETSC_DECIDE,300,300,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
    ierr = PetscDrawSPCreate(draw,1,&drawsp);CHKERRQ(ierr);
    for( i=0; i<eps->nconv; i++ ) {
#if defined(PETSC_USE_COMPLEX)
      re = PetscRealPart(eps->eigr[i]);
      im = PetscImaginaryPart(eps->eigi[i]);
#else
      re = eps->eigr[i];
      im = eps->eigi[i];
#endif
      ierr = PetscDrawSPAddPoint(drawsp,&re,&im);CHKERRQ(ierr);
    }
    ierr = PetscDrawSPDraw(drawsp);CHKERRQ(ierr);
    ierr = PetscDrawSPDestroy(drawsp);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDestroy"
/*@
   EPSDestroy - Destroys the EPS context.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSSolve()
@*/
int EPSDestroy(EPS eps)
{
  int         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (--eps->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(eps);CHKERRQ(ierr);

  ierr = STDestroy(eps->OP);CHKERRQ(ierr);

  if (eps->ops->destroy) {
    ierr = (*eps->ops->destroy)(eps); CHKERRQ(ierr);
  }
  
  if (eps->perm) {
    ierr = PetscFree(eps->perm);CHKERRQ(ierr);
  }

  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial);CHKERRQ(ierr);
  }

  PetscLogObjectDestroy(eps);
  PetscHeaderDestroy(eps);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetTolerances"
/*@
   EPSGetTolerances - Gets the tolerance and maximum
   iteration count used by the default EPS convergence tests. 

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameters:
+  tol - the convergence tolerance
-  maxits - maximum number of iterations

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: EPSSetTolerances()
@*/
int EPSGetTolerances(EPS eps,PetscReal *tol,int *maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (tol)    *tol    = eps->tol;
  if (maxits) *maxits = eps->max_it;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetTolerances"
/*@
   EPSSetTolerances - Sets the tolerance and maximum
   iteration count used by the default EPS convergence testers. 

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
.  tol - the convergence tolerance
-  maxits - maximum number of iterations to use

   Options Database Keys:
+  -eps_tol <tol> - Sets the convergence tolerance
-  -eps_max_it <maxits> - Sets the maximum number of iterations allowed

   Notes:
   Use PETSC_DEFAULT to retain the default value of any of the tolerances.

   Level: intermediate

.seealso: EPSGetTolerances()
@*/
int EPSSetTolerances(EPS eps,PetscReal tol,int maxits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (tol != PETSC_DEFAULT)    eps->tol    = tol;
  if (maxits != PETSC_DEFAULT) eps->max_it = maxits;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetDimensions"
/*@
   EPSGetDimensions - Gets the number of eigenvalues to compute
   and the dimension of the subspace.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameters:
+  nev - number of eigenvalues to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Notes:
   The user can specify PETSC_NULL for any parameter that is not needed.

   Level: intermediate

.seealso: EPSSetDimensions()
@*/
int EPSGetDimensions(EPS eps,int *nev,int *ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if( nev )   *nev = eps->nev;
  if( ncv )   *ncv = eps->ncv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDimensions"
/*@
   EPSSetDimensions - Sets the number of eigenvalues to compute
   and the dimension of the subspace.

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
.  nev - number of eigenvalues to compute
-  ncv - the maximum dimension of the subspace to be used by the solver

   Options Database Keys:
+  -eps_nev <nev> - Sets the number of eigenvalues
-  -eps_ncv <ncv> - Sets the dimension of the subspace

   Notes:
   Use PETSC_DEFAULT to retain the previous value of any parameter.

   Use PETSC_DECIDE for ncv to assign a reasonably good value, which is 
   dependent on the solution method.

   Level: intermediate

.seealso: EPSGetDimensions()
@*/
int EPSSetDimensions(EPS eps,int nev,int ncv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if( nev != PETSC_DEFAULT ) {
    if (nev<1) SETERRQ(1,"Illegal value of nev. Must be > 0");
    eps->nev = nev;
  }
  if( ncv == PETSC_DECIDE ) eps->ncv = 0;
  else if( ncv != PETSC_DEFAULT ) {
    if (ncv<1) SETERRQ(1,"Illegal value of ncv. Must be > 0");
    eps->ncv = ncv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetConverged"
/*@
   EPSGetConverged - Gets the number of converged eigenpairs.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context
  
   Output Parameter:
.  nconv - number of converged eigenpairs 

   Note:
   This function should be called after EPSSolve() has finished.

   Level: beginner

.seealso: EPSSetDimensions()
@*/
int EPSGetConverged(EPS eps,int *nconv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (nconv) *nconv = eps->nconv;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetEigenpair" 
/*@
   EPSGetEigenpair - Gets the i-th solution of the eigenproblem 
   as computed by EPSSolve(). The solution consists in both the eigenvalue
   and the eigenvector (if available).

   Not Collective

   Input Parameters:
+  eps - eigensolver context 
-  i   - index of the solution

   Output Parameters:
+  eigr - real part of eigenvalue
.  eigi - imaginary part of eigenvalue
.  Vr   - real part of eigenvector
-  Vi   - imaginary part of eigenvector

   Notes:
   If the eigenvalue is real, then eigi and Vi are set to zero. In the 
   complex case (e.g. with BOPT=O_complex) the eigenvalue is stored 
   directly in eigr (eigi is set to zero) and the eigenvector Vr (Vi is 
   set to zero).

   The index i should be a value between 0 and nconv (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs()
@*/
int EPSGetEigenpair(EPS eps, int i, PetscScalar *eigr, PetscScalar *eigi, Vec Vr, Vec Vi)
{
  int         ierr, k;
  PetscScalar zero = 0.0, minus = -1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!eps->eigr || !eps->eigi || !eps->V) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  if (eps->dropvectors && (Vr || Vi) ) { 
    SETERRQ(1, "Eigenvectors are not available"); 
  }  

  if (!eps->perm) k = i;
  else k = eps->perm[i];
#ifdef PETSC_USE_COMPLEX
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = 0;
  if (Vr) { ierr = VecCopy(eps->V[k], Vr); CHKERRQ(ierr); }
  if (Vi) { ierr = VecSet(&zero, Vi); CHKERRQ(ierr); }
#else
  if (eigr) *eigr = eps->eigr[k];
  if (eigi) *eigi = eps->eigi[k];
  if (eps->eigi[k] > 0) { /* first value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->V[k], Vr); CHKERRQ(ierr); }
    if (Vi) { ierr = VecCopy(eps->V[k+1], Vi); CHKERRQ(ierr); }
  } else if (eps->eigi[k] < 0) { /* second value of conjugate pair */
    if (Vr) { ierr = VecCopy(eps->V[k-1], Vr); CHKERRQ(ierr); }
    if (Vi) { 
      ierr = VecCopy(eps->V[k], Vi); CHKERRQ(ierr); 
      ierr = VecScale(&minus, Vi); CHKERRQ(ierr); 
    }
  } else { /* real eigenvalue */
    if (Vr) { ierr = VecCopy(eps->V[k], Vr); CHKERRQ(ierr); }
    if (Vi) { ierr = VecSet(&zero, Vi); CHKERRQ(ierr); }
  }
#endif
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetErrorEstimate" 
/*@
   EPSGetErrorEstimate - Returns the error bound associated to the i-th 
   approximate eigenpair.

   Not Collective

   Input Parameter:
+  eps - eigensolver context 
-  i   - index of eigenpair

   Output Parameter:
.  errest - the error estimate

   Level: advanced

.seealso: EPSComputeRelativeError()
@*/
int EPSGetErrorEstimate(EPS eps, int i, PetscReal *errest)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (!eps->eigr || !eps->eigi) { 
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "EPSSolve must be called first"); 
  }
  if (i<0 || i>=eps->nconv) { 
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE, "Argument 2 out of range"); 
  }
  if (eps->perm) i = eps->perm[i];  
  if (errest) *errest = eps->errest[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetST"
/*@
   EPSSetST - Associates a spectral transformation object to the
   eigensolver. 

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  st   - the spectral transformation object

   Note:
   Use EPSGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetST()
@*/
int EPSSetST(EPS eps,ST st)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(st,ST_COOKIE,2);
  PetscCheckSameComm(eps,1,st,2);
  ierr = STDestroy(eps->OP); CHKERRQ(ierr);
  eps->OP = st;
  PetscObjectReference((PetscObject)eps->OP);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetST"
/*@
   EPSGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: EPSSetST()
@*/
int EPSGetST(EPS eps, ST *st)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *st = eps->OP;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetMonitor"
/*@C
   EPSSetMonitor - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the error estimates for each requested eigenpair.
      
   Collective on EPS

   Input Parameters:
+  eps     - eigensolver context obtained from EPSCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
-  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)

   Calling Sequence of monitor:
$     monitor (EPS eps, int its, int nconv, PetscReal* errest, int nest, void *mctx)

+  eps    - eigensolver context obtained from EPSCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  errest - error estimates for each eigenpair
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by EPSSetMonitor()

   Options Database Keys:
+    -eps_monitor        - print error estimates at each iteration
-    -eps_cancelmonitors - cancels all monitors that have been hardwired into
      a code by calls to EPSetMonitor(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   EPSSetMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: EPSDefaultEstimatesMonitor(), EPSClearMonitor()
@*/
int EPSSetMonitor(EPS eps, int (*monitor)(EPS,int,int,PetscReal*,int,void*), void *mctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->numbermonitors >= MAXEPSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many EPS monitors set");
  }
  eps->monitor[eps->numbermonitors]           = monitor;
  eps->monitorcontext[eps->numbermonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetValuesMonitor"
/*@C
   EPSSetValuesMonitor - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the value of approximate eigenvalues.
      
   Collective on EPS

   Input Parameters:
+  eps     - eigensolver context obtained from EPSCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring
-  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)

   Calling Sequence of monitor:
$     monitor (EPS eps, int its, int nconv, PetscScalar* kr, PetscScalar* ki, int nest, void *mctx)

+  eps    - eigensolver context obtained from EPSCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  kr     - real part of each eigenvalue
.  ki     - imaginary part of each eigenvalue
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by EPSSetMonitor()

   Options Database Keys:
+    -eps_monitor_values - print eigenvalue approximations at each iteration
-    -eps_cancelmonitors - cancels all monitors that have been hardwired into
      a code by calls to EPSetValuesMonitor(), but does not cancel those set 
      via the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   EPSSetValuesMonitor() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: EPSDefaultValuesMonitor(), EPSClearMonitor()
@*/
int EPSSetValuesMonitor(EPS eps, int (*monitor)(EPS,int,int,PetscScalar*,PetscScalar*,int,void*), void *mctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->numbervmonitors >= MAXEPSMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many EPS values monitors set");
  }
  eps->vmonitor[eps->numbervmonitors]           = monitor;
  eps->vmonitorcontext[eps->numbervmonitors++]  = (void*)mctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSClearMonitor"
/*@C
   EPSClearMonitor - Clears all monitors for an EPS object.

   Collective on EPS

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Options Database Key:
.    -eps_cancelmonitors - Cancels all monitors that have been hardwired 
      into a code by calls to EPSSetMonitor() or EPSSetValuesMonitor(), 
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: EPSSetMonitor(), EPSSetValuesMonitor()
@*/
int EPSClearMonitor(EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->numbermonitors = 0;
  eps->numbervmonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetMonitorContext"
/*@C
   EPSGetMonitorContext - Gets the estimates monitor context, as set by 
   EPSSetMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: EPSSetMonitor(), EPSDefaultEstimatesMonitor()
@*/
int EPSGetMonitorContext(EPS eps, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *ctx =      (eps->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetValuesMonitorContext"
/*@C
   EPSGetValuesMonitorContext - Gets the values monitor context, as set by 
   EPSSetValuesMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: EPSSetValuesMonitor(), EPSDefaultValuesMonitor()
@*/
int EPSGetValuesMonitorContext(EPS eps, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *ctx =      (eps->vmonitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialVector"
/*@
   EPSSetInitialVector - Sets the initial vector from which the 
   eigensolver starts to iterate.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: EPSGetInitialVector()

@*/
int EPSSetInitialVector(EPS eps,Vec vec)
{
  int ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial); CHKERRQ(ierr);
  }
  eps->vec_initial = vec;
  PetscObjectReference((PetscObject)eps->vec_initial);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInitialVector"
/*@
   EPSGetInitialVector - Gets the initial vector associated with the 
   eigensolver; if the vector was not set it will return a 0 pointer or
   a vector randomly generated by EPSSetUp().

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: EPSSetInitialVector()

@*/
int EPSGetInitialVector(EPS eps,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *vec = eps->vec_initial;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetWhichEigenpairs"
/*@
    EPSSetWhichEigenpairs - Specifies which portion of the spectrum is 
    to be sought.

    Collective on EPS

    Input Parameter:
.   eps - eigensolver context obtained from EPSCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Possible values:
    The parameter 'which' can have one of these values:
    
+     EPS_LARGEST_MAGNITUDE - largest eigenvalues in magnitude (default)
.     EPS_SMALLEST_MAGNITUDE - smallest eigenvalues in magnitude
.     EPS_LARGEST_REAL - largest real parts
.     EPS_SMALLEST_REAL - smallest real parts
.     EPS_LARGEST_IMAGINARY - largest imaginary parts
-     EPS_SMALLEST_IMAGINARY - smallest imaginary parts

    Options Database Keys:
+   -eps_largest_magnitude - Sets largest eigenvalues in magnitude
.   -eps_smallest_magnitude - Sets smallest eigenvalues in magnitude
.   -eps_largest_real - Sets largest real parts
.   -eps_smallest_real - Sets smallest real parts
.   -eps_largest_imaginary - Sets largest imaginary parts in magnitude
-   -eps_smallest_imaginary - Sets smallest imaginary parts in magnitude

    Notes:
    Not all eigensolvers implemented in EPS account for all the possible values
    stated above. Also, some values make sense only for certain types of 
    problems. If SLEPc is compiled for real numbers EPS_LARGEST_IMAGINARY
    and EPS_SMALLEST_IMAGINARY use the absolute value of the imaginary part 
    for eigenvalue selection.     
    
    Level: intermediate

.seealso: EPSGetWhichEigenpairs(), EPSSortEigenvalues()
@*/
int EPSSetWhichEigenpairs(EPS eps,EPSWhich which)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->which = which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetWhichEigenpairs"
/*@
    EPSGetWhichEigenpairs - Returns which portion of the spectrum is to be 
    sought.

    Not Collective

    Input Parameter:
.   eps - eigensolver context obtained from EPSCreate()

    Output Parameter:
.   which - the portion of the spectrum to be sought

    Notes:
    See EPSSetWhichEigenpairs() for possible values of which

    Level: intermediate

.seealso: EPSSetWhichEigenpairs()
@*/
int EPSGetWhichEigenpairs(EPS eps,EPSWhich *which) 
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *which = eps->which;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeExplicitOperator"
/*@
    EPSComputeExplicitOperator - Computes the explicit operator associated
    to the eigenvalue problem with the specified spectral transformation.  

    Collective on EPS

    Input Parameter:
.   eps - the eigenvalue solver context

    Output Parameter:
.   mat - the explicit operator

    Notes:
    This routine builds a matrix containing the explicit operator. For 
    example, in generalized problems with shift-and-invert spectral
    transformation the result would be matrix (A - s B)^-1 B.
    
    This computation is done by applying the operator to columns of the 
    identity matrix.

    Currently, this routine uses a dense matrix format when 1 processor
    is used and a sparse format otherwise.  This routine is costly in general,
    and is recommended for use only with relatively small systems.

    Level: advanced

.seealso: STApply()   
@*/
int EPSComputeExplicitOperator(EPS eps,Mat *mat)
{
  Vec         in,out;
  int         ierr,i,M,m,size,*rows,start,end;
  MPI_Comm    comm;
  PetscScalar *array,zero = 0.0,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(mat,2);
  comm = eps->comm;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecDuplicate(eps->vec_initial,&in);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&out);CHKERRQ(ierr);
  ierr = VecGetSize(in,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(in,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) {rows[i] = start + i;}

  if (size == 1) {
    ierr = MatCreateSeqDense(comm,M,M,PETSC_NULL,mat);CHKERRQ(ierr);
  } else {
    ierr = MatCreateMPIAIJ(comm,m,m,M,M,0,0,0,0,mat);CHKERRQ(ierr);
  }
  
  for (i=0; i<M; i++) {

    ierr = VecSet(&zero,in);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(eps->OP,in,out); CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);

  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOperators"
/*@
   EPSSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on EPS and Mat

   Input Parameters:
+  eps - the eigenproblem solver context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes: 
   To specify a standard eigenproblem, use PETSC_NULL for parameter B.

   Level: beginner

.seealso: EPSSolve(), EPSGetST(), STGetOperators()
@*/
int EPSSetOperators(EPS eps,Mat A,Mat B)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  ierr = STSetOperators(eps->OP,A,B);CHKERRQ(ierr);
  eps->setupcalled = 0;  /* so that next solve call will call setup */

  /* The following call is done in order to check the consistency of the
     problem type with the specified matrices */
  if (eps->problem_type) {
    ierr = EPSSetProblemType(eps,eps->problem_type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeResidualNorm"
/*@
   EPSComputeResidualNorm - Computes the residual norm associated with 
   the i-th converged approximate eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  norm - the residual norm, computed as ||Ax-kBx|| where k is the 
   eigenvalue and x is the eigenvector. 
   If k=0 then the residual norm is computed as ||Ax||.

   Notes:
   The index i should be a value between 0 and nconv (see EPSGetConverged()).
   Eigenpairs are indexed according to the ordering criterion established 
   with EPSSetWhichEigenpairs().

   Level: beginner

.seealso: EPSSolve(), EPSGetConverged(), EPSSetWhichEigenpairs()
@*/
int EPSComputeResidualNorm(EPS eps, int i, PetscReal *norm)
{
  Vec         u, v, w, xr, xi;
  Mat         A, B;
  int         ierr;
  PetscScalar alpha, kr, ki;
  PetscReal   ni, nr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->dropvectors || !eps->V) { SETERRQ(1, "Eigenvectors are not available"); }  
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&v); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&w); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xr); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xi); CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi); CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    ierr = MatMult( A, xr, u ); CHKERRQ(ierr); /* u=A*x */
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      if (eps->isgeneralized) { ierr = MatMult( B, xr, w ); CHKERRQ(ierr); }
      else { ierr = VecCopy( xr, w ); CHKERRQ(ierr); } /* w=B*x */
      alpha = -kr; 
      ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*x-k*B*x */
    }
    ierr = VecNorm( u, NORM_2, norm); CHKERRQ(ierr);  
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = MatMult( A, xr, u ); CHKERRQ(ierr); /* u=A*xr */
    if (eps->isgeneralized) { ierr = MatMult( B, xr, v ); CHKERRQ(ierr); }
    else { ierr = VecCopy( xr, v ); CHKERRQ(ierr); } /* v=B*xr */
    alpha = -kr;
    ierr = VecAXPY( &alpha, v, u ); CHKERRQ(ierr); /* u=A*xr-kr*B*xr */
    if (eps->isgeneralized) { ierr = MatMult( B, xi, w ); CHKERRQ(ierr); }
    else { ierr = VecCopy( xi, w ); CHKERRQ(ierr); } /* w=B*xi */
    alpha = ki;
    ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*xr-kr*B*xr+ki*B*xi */
    ierr = VecNorm( u, NORM_2, &nr ); CHKERRQ(ierr);
    ierr = MatMult( A, xi, u ); CHKERRQ(ierr); /* u=A*xi */
    alpha = -kr;
    ierr = VecAXPY( &alpha, w, u ); CHKERRQ(ierr); /* u=A*xi-kr*B*xi */
    alpha = -ki;
    ierr = VecAXPY( &alpha, v, u ); CHKERRQ(ierr); /* u=A*xi-kr*B*xi-ki*B*xr */
    ierr = VecNorm( u, NORM_2, &ni ); CHKERRQ(ierr);
    *norm = LAlapy2_( &nr, &ni );
  }
#endif

  ierr = VecDestroy(w); CHKERRQ(ierr);
  ierr = VecDestroy(v); CHKERRQ(ierr);
  ierr = VecDestroy(u); CHKERRQ(ierr);
  ierr = VecDestroy(xr); CHKERRQ(ierr);
  ierr = VecDestroy(xi); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSComputeRelativeError"
/*@
   EPSComputeRelativeError - Computes the actual relative error associated 
   with the i-th converged approximate eigenpair.

   Collective on EPS

   Input Parameter:
.  eps - the eigensolver context
.  i   - the solution index

   Output Parameter:
.  error - the relative error, computed as ||Ax-kBx||/||kx|| where k is the 
   eigenvalue and x is the eigenvector. 
   If k=0 the relative error is computed as ||Ax||/||x||.

   Level: beginner

.seealso: EPSSolve(), EPSComputeResidualNorm()
@*/
int EPSComputeRelativeError(EPS eps, int i, PetscReal *error)
{
  Vec         xr, xi, u;
  int         ierr;
  PetscScalar kr, ki, alpha;
  PetscReal   norm, er, ei;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);  
  ierr = EPSComputeResidualNorm(eps,i,&norm); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xr); CHKERRQ(ierr);
  ierr = VecDuplicate(eps->vec_initial,&xi); CHKERRQ(ierr);
  ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi); CHKERRQ(ierr);

#ifndef PETSC_USE_COMPLEX
  if (ki == 0 || 
    PetscAbsScalar(ki) < PetscAbsScalar(kr*PETSC_MACHINE_EPSILON)) {
#endif
    if (PetscAbsScalar(kr) > PETSC_MACHINE_EPSILON) {
      ierr = VecScale(&kr, xr); CHKERRQ(ierr);
    }
    ierr = VecNorm(xr, NORM_2, &er); CHKERRQ(ierr);
    *error = norm / er; 
#ifndef PETSC_USE_COMPLEX
  } else {
    ierr = VecDuplicate(xi, &u); CHKERRQ(ierr);  
    ierr = VecCopy(xi, u); CHKERRQ(ierr);  
    alpha = -ki;
    ierr = VecAXPBY(&kr, &alpha, xr, u); CHKERRQ(ierr);   
    ierr = VecNorm(u, NORM_2, &er); CHKERRQ(ierr);  
    ierr = VecAXPBY(&kr, &ki, xr, xi);  CHKERRQ(ierr);      
    ierr = VecNorm(xi, NORM_2, &ei); CHKERRQ(ierr);  
    ierr = VecDestroy(u); CHKERRQ(ierr);  
    *error = norm / LAlapy2_(&er, &ei);
  }
#endif    
  
  ierr = VecDestroy(xr); CHKERRQ(ierr);
  ierr = VecDestroy(xi); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetProblemType"
/*@
   EPSSetProblemType - Specifies the type of the eigenvalue problem.

   Collective on EPS

   Input Parameters:
+  eps      - the eigensolver context
-  type     - a known type of eigenvalue problem 

   Options Database Keys:
+  -eps_hermitian - Hermitian eigenvalue problem (default)
.  -eps_gen_hermitian - generalized Hermitian eigenvalue problem
.  -eps_non_hermitian - non-Hermitian eigenvalue problem
-  -eps_gen_non_hermitian - generalized non-Hermitian eigenvalue problem 
    
   Note:  
   Normally, the user need not set the EPS type, since it can be determined from
   the information given in the EPSSetOperators call. This routine is reserved
   for special cases such as when a nonsymmetric solver wants to be 
   used in a symmetric problem. 

  Level: advanced

.seealso: EPSSetOperators(), EPSSetType(), EPSType
@*/
int EPSSetProblemType(EPS eps,EPSProblemType type)
{
  int        n,m,ierr;
  Mat        A,B;
  PetscTruth Ah,Bh,inconsistent=PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (type!=EPS_HEP && type!=EPS_GHEP && type!=EPS_NHEP && type!=EPS_GNHEP ) { SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown eigenvalue problem type"); }

  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  if (!A) { SETERRQ(1,"Must call EPSSetOperators() first"); }

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"A is a non-square matrix"); }
  if (B) { 
    ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
    if (m!=n) { SETERRQ(1,"B is a non-square matrix"); }
  }

  eps->problem_type = type;

  ierr = SlepcIsHermitian(A,&Ah);CHKERRQ(ierr);
  if (B) { ierr = SlepcIsHermitian(B,&Bh);CHKERRQ(ierr); }

  if (!B) {
    eps->isgeneralized = PETSC_FALSE;
    if (Ah) eps->ishermitian = PETSC_TRUE;
    else    eps->ishermitian = PETSC_FALSE;
  }
  else {
    eps->isgeneralized = PETSC_TRUE;
    if (Ah && Bh) eps->ishermitian = PETSC_TRUE;
    else          eps->ishermitian = PETSC_FALSE;
  }
 
  switch (type) {
    case EPS_HEP:
      if (eps->isgeneralized || !eps->ishermitian) inconsistent=PETSC_TRUE;
      eps->ishermitian = PETSC_TRUE;
      break;
    case EPS_GHEP:
      /* Note that here we do not consider the case in which A and B are 
         non-hermitian but there exists a linear combination of them which is */
      if (!eps->isgeneralized || !eps->ishermitian) inconsistent=PETSC_TRUE;
      break;
    case EPS_NHEP:
      if (eps->isgeneralized) inconsistent=PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      break;
    case EPS_GNHEP:
      /* If matrix B is not given then an error is issued. An alternative 
         would be to generate an identity matrix. Also in EPS_GHEP above */
      if (!eps->isgeneralized) inconsistent=PETSC_TRUE;
      eps->ishermitian = PETSC_FALSE;
      break;
  }
  if (inconsistent) { SETERRQ(0,"Warning: Inconsistent EPS state"); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetProblemType"
/*@
   EPSGetProblemType - Gets the problem type from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context 

   Output Parameter:
.  type - name of EPS problem type 

   Level: intermediate

.seealso: EPSSetProblemType()
@*/
int EPSGetProblemType(EPS eps,EPSProblemType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *type = eps->problem_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsGeneralized"
/*@
   EPSIsGeneralized - Ask if the EPS object corresponds to a generalized 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
int EPSIsGeneralized(EPS eps,PetscTruth* is)
{
  int  ierr;
  Mat  B;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = STGetOperators(eps->OP,PETSC_NULL,&B);CHKERRQ(ierr);
  if( B ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  if( eps->setupcalled ) {
    if( eps->isgeneralized != *is ) { 
      SETERRQ(0,"Warning: Inconsistent EPS state");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSIsHermitian"
/*@
   EPSIsHermitian - Ask if the EPS object corresponds to a Hermitian 
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

@*/
int EPSIsHermitian(EPS eps,PetscTruth* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if( eps->ishermitian ) *is = PETSC_TRUE;
  else *is = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSReverseProjection"
/*@
   EPSReverseProjection - Compute the operation V=V*S, where the columns of
   V are m of the basis vectors of the EPS object and S is an mxm dense
   matrix.

   Collective on EPS

   Input Parameter:
+  eps - the eigenproblem solver context
.  k - starting column
.  m - dimension of matrix S
-  S - pointer to the values of matrix S

   Level: developer

   Note:
   Matrix S is overwritten.

@*/
int EPSReverseProjection(EPS eps,int k,int m,PetscScalar *S)
{
  int         i,j,n,ierr,lwork;
  PetscScalar *tau,*work,*pV;
  
  PetscFunctionBegin;

  ierr = VecGetLocalSize(eps->vec_initial,&n);CHKERRQ(ierr);
  lwork = n;
  ierr = PetscMalloc(m*sizeof(PetscScalar),&tau);CHKERRQ(ierr);
  ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);

  /* compute the LQ factorization L Q = S */
  LAgelqf_(&m,&m,S,&m,tau,work,&lwork,&ierr);

  /* triangular post-multiplication, V = V L */
  for (i=k;i<k+m;i++) {
    ierr = VecScale(S+(i-k)+m*(i-k),eps->V[i]);CHKERRQ(ierr);
    for (j=i+1;j<k+m;j++) {
      ierr = VecAXPY(S+(j-k)+m*(i-k),eps->V[j],eps->V[i]);CHKERRQ(ierr);
    }
  }

  /* orthogonal post-multiplication, V = V Q */
  ierr = VecGetArray(eps->V[k],&pV);CHKERRQ(ierr);
  LAormlq_("R","N",&n,&m,&m,S,&m,tau,pV,&n,work,&lwork,&ierr,1,1);
  ierr = VecRestoreArray(eps->V[k],&pV);CHKERRQ(ierr);

  ierr = PetscFree(tau);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "EPSSwapEigenpairs"
/*@
   EPSSwapEigenpairs - Swaps all the information internal to the EPS object
   corresponding to eigenpairs which occupy the i-th and j-th positions.

   Collective on EPS

   Input Parameter:
+  eps - the eigenproblem solver context
.  i - first index
-  j - second index

   Level: developer

@*/
int EPSSwapEigenpairs(EPS eps,int i,int j)
{
  int         ierr;
  PetscScalar tscalar;
  PetscReal   treal;
  
  PetscFunctionBegin;
  if (i!=j) {
    ierr = VecSwap(eps->V[i],eps->V[j]);CHKERRQ(ierr);
    tscalar = eps->eigr[i];
    eps->eigr[i] = eps->eigr[j];
    eps->eigr[j] = tscalar;
    tscalar = eps->eigi[i];
    eps->eigi[i] = eps->eigi[j];
    eps->eigi[j] = tscalar;
    treal = eps->errest[i];
    eps->errest[i] = eps->errest[j];
    eps->errest[j] = treal;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBackTransform_Default"
int EPSBackTransform_Default(EPS eps)
{
  ST          st;
  int         ierr,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  for (i=0;i<eps->nconv;i++) {
    ierr = STBackTransform(st,&eps->eigr[i],&eps->eigi[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
