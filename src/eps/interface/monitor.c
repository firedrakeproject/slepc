/*
      EPS routines related to monitors.
*/
#include "src/eps/epsimpl.h"   /*I "slepceps.h" I*/

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
$     monitor (EPS eps, int its, int nconv, PetscScalar *eigr, PetscScalar *eigi, PetscReal* errest, int nest, void *mctx)

+  eps    - eigensolver context obtained from EPSCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
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
PetscErrorCode EPSSetMonitor(EPS eps, int (*monitor)(EPS,int,int,PetscScalar*,PetscScalar*,PetscReal*,int,void*), void *mctx)
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

.seealso: EPSSetMonitor()
@*/
PetscErrorCode EPSClearMonitor(EPS eps)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  eps->numbermonitors = 0;
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
PetscErrorCode EPSGetMonitorContext(EPS eps, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  *ctx =      (eps->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSDefaultMonitor"
/*@C
   EPSDefaultEstimatesMonitor - Print the current approximate values and 
   error estimates at each iteration of the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps    - eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: EPSSetMonitor()
@*/
PetscErrorCode EPSDefaultMonitor(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *dummy)
{
  PetscErrorCode ierr;
  int            i;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(eps->comm);
  ierr = PetscViewerASCIIPrintf(viewer,"%3d EPS nconv=%d Values (Errors)",its,nconv);CHKERRQ(ierr);
  for (i=0;i<nest;i++) {
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",eigr[i]);CHKERRQ(ierr);
    if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",eigi[i]);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",errest[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSLGMonitor"
PetscErrorCode EPSLGMonitor(EPS eps,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,int nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y;
  int            i,n = eps->nev;
#if !defined(PETSC_USE_COMPLEX)
  int            pause;
  PetscDraw      draw1;
  PetscDrawLG    lg1;
#endif

  PetscFunctionBegin;

  if (!viewer) { viewer = PETSC_VIEWER_DRAW_(eps->comm); }

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(eps->tol)-2,0.0);CHKERRQ(ierr);
  }

#if !defined(PETSC_USE_COMPLEX)
  if (eps->ishermitian) {
    ierr = PetscViewerDrawGetDraw(viewer,1,&draw1);CHKERRQ(ierr);
    ierr = PetscViewerDrawGetDrawLG(viewer,1,&lg1);CHKERRQ(ierr);
    if (!its) {
      ierr = PetscDrawSetTitle(draw1,"Approximate eigenvalues");CHKERRQ(ierr);
      ierr = PetscDrawSetDoubleBuffer(draw1);CHKERRQ(ierr);
      ierr = PetscDrawLGSetDimension(lg1,n);CHKERRQ(ierr);
      ierr = PetscDrawLGReset(lg1);CHKERRQ(ierr);
      ierr = PetscDrawLGSetLimits(lg1,0,1.0,1.e20,-1.e20);CHKERRQ(ierr);
    }
  }
#endif

  ierr = PetscMalloc(sizeof(PetscReal)*n,&x);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal) its;
    if (errest[i] > 0.0) y[i] = log10(errest[i]); else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  if (eps->ishermitian) {
    ierr = PetscDrawLGAddPoint(lg1,x,eps->eigr);CHKERRQ(ierr);
    ierr = PetscDrawGetPause(draw1,&pause);CHKERRQ(ierr);
    ierr = PetscDrawSetPause(draw1,0);CHKERRQ(ierr);    
    ierr = PetscDrawLGDraw(lg1);CHKERRQ(ierr);
    ierr = PetscDrawSetPause(draw1,pause);CHKERRQ(ierr);    
  }
#endif  
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
} 
