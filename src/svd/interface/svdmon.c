/*
      SVD routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

   This file is part of SLEPc.
      
   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY 
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS 
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for 
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <private/svdimpl.h>   /*I "slepcsvd.h" I*/

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
$     monitor (SVD svd, PetscInt its, PetscInt nconv, PetscReal *sigma, PetscReal* errest, PetscInt nest, void *mctx)

+  svd    - singular value solver context obtained from SVDCreate()
.  its    - iteration number
.  nconv  - number of converged singular triplets
.  sigma  - singular values
.  errest - relative error estimates for each singular triplet
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by SVDMonitorSet()

   Options Database Keys:
+    -svd_monitor          - print only the first error estimate
.    -svd_monitor_all      - print error estimates at each iteration
.    -svd_monitor_conv     - print the singular value approximations only when
      convergence has been reached
.    -svd_monitor_draw     - sets line graph monitor for the first unconverged
      approximate singular value
.    -svd_monitor_draw_all - sets line graph monitor for all unconverged
      approximate singular value
-    -svd_monitor_cancel   - cancels all monitors that have been hardwired into
      a code by calls to SVDMonitorSet(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   SVDMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: SVDMonitorFirst(), SVDMonitorAll(), SVDMonitorLG(), SVDMonitorLGAll(), SVDMonitorCancel()
@*/
PetscErrorCode SVDMonitorSet(SVD svd,PetscErrorCode (*monitor)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*),
                             void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->numbermonitors >= MAXSVDMONITORS) {
    SETERRQ(((PetscObject)svd)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many SVD monitors set");
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

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorCancel(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
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

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDGetMonitorContext(SVD svd, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  *ctx =      (svd->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorAll"
/*@C
   SVDMonitorAll - Print the current approximate values and 
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
PetscErrorCode SVDMonitorAll(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (its) {
    if (!dummy) {ierr = PetscViewerASCIIMonitorCreate(((PetscObject)svd)->comm,"stdout",0,&viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3d SVD nconv=%d Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
      ierr = PetscViewerASCIIMonitorPrintf(viewer," %g (%10.8e)",sigma[i],errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"\n");CHKERRQ(ierr);
    if (!dummy) {ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorFirst"
/*@C
   SVDMonitorFirst - Print the first unconverged approximate values and 
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
PetscErrorCode SVDMonitorFirst(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (its && nconv<nest) {
    if (!dummy) {ierr = PetscViewerASCIIMonitorCreate(((PetscObject)svd)->comm,"stdout",0,&viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3d SVD nconv=%d first unconverged value (error)",its,nconv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIMonitorPrintf(viewer," %g (%10.8e)\n",sigma[nconv],errest[nconv]);CHKERRQ(ierr);
    if (!dummy) {ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorConverged"
/*@C
   SVDMonitorConverged - Print the approximate values and error estimates as they converge.

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
PetscErrorCode SVDMonitorConverged(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  SVDMONITOR_CONV *ctx = (SVDMONITOR_CONV*) dummy;

  PetscFunctionBegin;
  if (!its) {
    ctx->oldnconv = 0;
  } else {
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIMonitorPrintf(ctx->viewer,"%3d SVD converged value (error) #%d",its,i);CHKERRQ(ierr);
      ierr = PetscViewerASCIIMonitorPrintf(ctx->viewer," %g (%10.8e)\n",sigma[i],errest[i]);CHKERRQ(ierr);
    }
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDMonitorDestroy_Converged(SVDMONITOR_CONV *ctx)
{
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = PetscViewerASCIIMonitorDestroy(ctx->viewer);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorLG"
PetscErrorCode SVDMonitorLG(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      x,y,p;
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
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(svd->tol)-2,0.0);CHKERRQ(ierr);

    ierr = PetscDrawSetTitle(draw1,"Approximate singular values");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg1,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg1,0,1.0,1.e20,-1.e20);CHKERRQ(ierr);
  }

  x = (PetscReal) its;
  if (errest[nconv] > 0.0) y = log10(errest[nconv]); else y = 0.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);

  ierr = PetscDrawLGAddPoint(lg1,&x,svd->sigma);CHKERRQ(ierr);
  ierr = PetscDrawGetPause(draw1,&p);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw1,0);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg1);CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw1,p);CHKERRQ(ierr);
    
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "SVDMonitorLGAll"
PetscErrorCode SVDMonitorLGAll(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y,p;
  int            n = PetscMin(svd->nsv,255);
  PetscInt       i;
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
    if (i < nest && errest[i] > 0.0) y[i] = log10(errest[i]);
    else y[i] = 0.0;
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
