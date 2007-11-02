/*
      SVD routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "src/svd/svdimpl.h"   /*I "slepcsvd.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorSet"
/*@C
   SVDMonitorSet - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the error estimates for each requested singular triplet.
      
   Collective on SVD

   Input Parameters:
+  svd     - singular value solver context obtained from SVDCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring)
-  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)

   Calling Sequence of monitor:
$     monitor (SVD svd, int its, int nconv, PetscReal *sigma, PetscReal* errest, int nest, void *mctx)

+  svd    - singular value solver context obtained from SVDCreate()
.  its    - iteration number
.  nconv  - number of converged singular triplets
.  sigma  - singular values
.  errest - relative error estimates for each singular triplet
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by SVDMonitorSet()

   Options Database Keys:
+    -svd_monitor        - print error estimates at each iteration
.    -svd_monitor_draw   - sets line graph monitor
-    -svd_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to SVDMonitorSet(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   SVDMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: SVDMonitorDefault(), SVDMonitorCancel()
@*/
PetscErrorCode SVDMonitorSet(SVD svd,PetscErrorCode (*monitor)(SVD,int,int,PetscReal*,PetscReal*,int,void*),
                             void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  if (svd->numbermonitors >= MAXSVDMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many SVD monitors set");
  }
  svd->monitor[svd->numbermonitors]           = monitor;
  svd->monitorcontext[svd->numbermonitors]    = (void*)mctx;
  svd->monitordestroy[svd->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorCancel"
/*@
   SVDMonitorCancel - Clears all monitors for an SVD object.

   Collective on SVD

   Input Parameters:
.  svd - singular value solver context obtained from SVDCreate()

   Options Database Key:
.    -svd_monitor_cancel - Cancels all monitors that have been hardwired 
      into a code by calls to SVDMonitorSet(),
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: SVDMonitorCancel()
@*/
PetscErrorCode SVDMonitorCancel(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  for (i=0; i<svd->numbermonitors; i++) {
    if (svd->monitordestroy[i]) {
      ierr = (*svd->monitordestroy[i])(svd->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  svd->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDGetMonitorContext"
/*@C
   SVDGetMonitorContext - Gets the monitor context, as set by 
   SVDMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  svd - singular value solver context obtained from SVDCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: SVDMonitorSet(), SVDMonitorDefault()
@*/
PetscErrorCode SVDGetMonitorContext(SVD svd, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_COOKIE,1);
  *ctx =      (svd->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorDefault"
/*@C
   SVDDefaultMonitor - Print the current approximate values and 
   error estimates at each iteration of the singular value solver.

   Collective on SVD

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorDefault(SVD svd,int its,int nconv,PetscReal *sigma,PetscReal *errest,int nest,void *dummy)
{
  PetscErrorCode ierr;
  int            i;
  PetscViewer    viewer = (PetscViewer) dummy;

  PetscFunctionBegin;
  if (its) {
    if (!viewer) viewer = PETSC_VIEWER_STDOUT_(((PetscObject)svd)->comm);
    ierr = PetscViewerASCIIPrintf(viewer,"%3d SVD nconv=%d Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
      ierr = PetscViewerASCIIPrintf(viewer," %g (%10.8e)",sigma[i],errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorLG"
PetscErrorCode SVDMonitorLG(SVD svd,int its,int nconv,PetscReal *sigma,PetscReal *errest,int nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y;
  int            i,n = svd->nsv;
  int            p;
  PetscDraw      draw1;
  PetscDrawLG    lg1;

  PetscFunctionBegin;

  if (!viewer) { viewer = PETSC_VIEWER_DRAW_(((PetscObject)svd)->comm); }

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,1,&draw1);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,1,&lg1);CHKERRQ(ierr);

  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(svd->tol)-2,0.0);CHKERRQ(ierr);

    ierr = PetscDrawSetTitle(draw1,"Approximate singular values");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg1,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg1,0,1.0,1.e20,-1.e20);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(sizeof(PetscReal)*n,&x);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal) its;
    if (errest[i] > 0.0) y[i] = log10(errest[i]); else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);

  ierr = PetscDrawLGAddPoint(lg1,x,svd->sigma);CHKERRQ(ierr);
  ierr = PetscDrawGetPause(draw1,&p);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw1,0);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg1);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw1,p);CHKERRQ(ierr);
    
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

