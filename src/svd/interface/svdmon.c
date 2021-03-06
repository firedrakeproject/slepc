/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SVD routines related to monitors
*/

#include <slepc/private/svdimpl.h>   /*I "slepcsvd.h" I*/
#include <petscdraw.h>

PetscErrorCode SVDMonitorLGCreate(MPI_Comm comm,const char host[],const char label[],const char metric[],PetscInt l,const char *names[],int x,int y,int m,int n,PetscDrawLG *lgctx)
{
  PetscDraw      draw;
  PetscDrawAxis  axis;
  PetscDrawLG    lg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,host,label,x,y,m,n,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetFromOptions(draw);CHKERRQ(ierr);
  ierr = PetscDrawLGCreate(draw,l,&lg);CHKERRQ(ierr);
  if (names) { ierr = PetscDrawLGSetLegend(lg,names);CHKERRQ(ierr); }
  ierr = PetscDrawLGSetFromOptions(lg);CHKERRQ(ierr);
  ierr = PetscDrawLGGetAxis(lg,&axis);CHKERRQ(ierr);
  ierr = PetscDrawAxisSetLabels(axis,"Convergence","Iteration",metric);CHKERRQ(ierr);
  ierr = PetscDrawDestroy(&draw);CHKERRQ(ierr);
  *lgctx = lg;
  PetscFunctionReturn(0);
}

/*
   Runs the user provided monitor routines, if any.
*/
PetscErrorCode SVDMonitor(SVD svd,PetscInt it,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest)
{
  PetscErrorCode ierr;
  PetscInt       i,n = svd->numbermonitors;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = (*svd->monitor[i])(svd,it,nconv,sigma,errest,nest,svd->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorSet - Sets an ADDITIONAL function to be called at every
   iteration to monitor the error estimates for each requested singular triplet.

   Logically Collective on svd

   Input Parameters:
+  svd     - singular value solver context obtained from SVDCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context (may be NULL)

   Calling Sequence of monitor:
$   monitor(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,void *mctx)

+  svd    - singular value solver context obtained from SVDCreate()
.  its    - iteration number
.  nconv  - number of converged singular triplets
.  sigma  - singular values
.  errest - relative error estimates for each singular triplet
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by SVDMonitorSet()

   Options Database Keys:
+    -svd_monitor        - print only the first error estimate
.    -svd_monitor_all    - print error estimates at each iteration
.    -svd_monitor_conv   - print the singular value approximations only when
      convergence has been reached
.    -svd_monitor draw::draw_lg - sets line graph monitor for the first unconverged
      approximate singular value
.    -svd_monitor_all draw::draw_lg - sets line graph monitor for all unconverged
      approximate singular values
.    -svd_monitor_conv draw::draw_lg - sets line graph monitor for convergence history
-    -svd_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to SVDMonitorSet(), but does not cancel those set via
      the options database.

   Notes:
   Several different monitoring routines may be set by calling
   SVDMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: SVDMonitorFirst(), SVDMonitorAll(), SVDMonitorCancel()
@*/
PetscErrorCode SVDMonitorSet(SVD svd,PetscErrorCode (*monitor)(SVD,PetscInt,PetscInt,PetscReal*,PetscReal*,PetscInt,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  if (svd->numbermonitors >= MAXSVDMONITORS) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_ARG_OUTOFRANGE,"Too many SVD monitors set");
  svd->monitor[svd->numbermonitors]           = monitor;
  svd->monitorcontext[svd->numbermonitors]    = (void*)mctx;
  svd->monitordestroy[svd->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

/*@
   SVDMonitorCancel - Clears all monitors for an SVD object.

   Logically Collective on svd

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
      ierr = (*svd->monitordestroy[i])(&svd->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  svd->numbermonitors = 0;
  PetscFunctionReturn(0);
}

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
PetscErrorCode SVDGetMonitorContext(SVD svd,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  *ctx = svd->monitorcontext[0];
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorFirst - Print the first unconverged approximate value and
   error estimate at each iteration of the singular value solver.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor - activates SVDMonitorFirst()

   Level: intermediate

.seealso: SVDMonitorSet(), SVDMonitorAll(), SVDMonitorConverged()
@*/
PetscErrorCode SVDMonitorFirst(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  if (its==1 && ((PetscObject)svd)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Singular value approximations and residual norms for %s solve.\n",((PetscObject)svd)->prefix);CHKERRQ(ierr);
  }
  if (nconv<nest) {
    ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D SVD nconv=%D first unconverged value (error)",its,nconv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," %g (%10.8e)\n",(double)sigma[nconv],(double)errest[nconv]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorAll - Print the current approximate values and
   error estimates at each iteration of the singular value solver.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor_all - activates SVDMonitorAll()

   Level: intermediate

.seealso: SVDMonitorSet(), SVDMonitorFirst(), SVDMonitorConverged()
@*/
PetscErrorCode SVDMonitorAll(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscViewer    viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
  if (its==1 && ((PetscObject)svd)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Singular value approximations and residual norms for %s solve.\n",((PetscObject)svd)->prefix);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%3D SVD nconv=%D Values (Errors)",its,nconv);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<nest;i++) {
    ierr = PetscViewerASCIIPrintf(viewer," %g (%10.8e)",(double)sigma[i],(double)errest[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorConverged - Print the approximate values and
   error estimates as they converge.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor_conv - activates SVDMonitorConverged()

   Level: intermediate

.seealso: SVDMonitorSet(), SVDMonitorFirst(), SVDMonitorAll()
@*/
PetscErrorCode SVDMonitorConverged(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscViewer    viewer;
  SlepcConvMon   ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  ctx = (SlepcConvMon)vf->data;
  if (its==1 && ((PetscObject)svd)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Convergence history for %s solve.\n",((PetscObject)svd)->prefix);CHKERRQ(ierr);
  }
  if (its==1) ctx->oldnconv = 0;
  if (ctx->oldnconv!=nconv) {
    ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%3D SVD converged value (error) #%D",its,i);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer," %g (%10.8e)\n",(double)sigma[i],(double)errest[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)svd)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode SVDMonitorConvergedCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;
  SlepcConvMon   mctx;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  ierr = PetscNew(&mctx);CHKERRQ(ierr);
  mctx->ctx = ctx;
  (*vf)->data = (void*)mctx;
  PetscFunctionReturn(0);
}

PetscErrorCode SVDMonitorConvergedDestroy(PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*vf) PetscFunctionReturn(0);
  ierr = PetscFree((*vf)->data);
  ierr = PetscViewerDestroy(&(*vf)->viewer);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(&(*vf)->lg);CHKERRQ(ierr);
  ierr = PetscFree(*vf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorFirstDrawLG - Plots the error estimate of the first unconverged
   approximation at each iteration of the singular value solver.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor draw::draw_lg - activates SVDMonitorFirstDrawLG()

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorFirstDrawLG(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  PetscDrawLG    lg;
  PetscReal      x,y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  lg = vf->lg;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,PetscLog10Real(svd->tol)-2,0.0);CHKERRQ(ierr);
  }
  if (nconv<nest) {
    x = (PetscReal)its;
    if (errest[nconv] > 0.0) y = PetscLog10Real(errest[nconv]);
    else y = 0.0;
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
    if (its <= 20 || !(its % 5) || svd->reason) {
      ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
      ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorFirstDrawLGCreate - Creates the plotter for the first error estimate.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorFirstDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = SVDMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"First Error Estimate","Log Error Estimate",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorAllDrawLG - Plots the error estimate of all unconverged
   approximations at each iteration of the singular value solver.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor_all draw::draw_lg - activates SVDMonitorAllDrawLG()

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorAllDrawLG(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  PetscDrawLG    lg;
  PetscInt       i,n = PetscMin(svd->nsv,255);
  PetscReal      *x,*y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  lg = vf->lg;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,PetscLog10Real(svd->tol)-2,0.0);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(n,&x,n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal)its;
    if (i < nest && errest[i] > 0.0) y[i] = PetscLog10Real(errest[i]);
    else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
  if (its <= 20 || !(its % 5) || svd->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscFree2(x,y);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorAllDrawLGCreate - Creates the plotter for all the error estimates.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorAllDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = SVDMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"All Error Estimates","Log Error Estimate",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorConvergedDrawLG - Plots the number of converged eigenvalues
   at each iteration of the singular value solver.

   Collective on svd

   Input Parameters:
+  svd    - singular value solver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged singular triplets so far
.  sigma  - singular values
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -svd_monitor_conv draw::draw_lg - activates SVDMonitorConvergedDrawLG()

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorConvergedDrawLG(SVD svd,PetscInt its,PetscInt nconv,PetscReal *sigma,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode   ierr;
  PetscViewer      viewer;
  PetscDrawLG      lg;
  PetscReal        x,y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(svd,SVD_CLASSID,1);
  PetscValidPointer(vf,7);
  viewer = vf->viewer;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,7);
  lg = vf->lg;
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,0,svd->nsv);CHKERRQ(ierr);
  }
  x = (PetscReal)its;
  y = (PetscReal)svd->nconv;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (its <= 20 || !(its % 5) || svd->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   SVDMonitorConvergedDrawLGCreate - Creates the plotter for the convergence history.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: SVDMonitorSet()
@*/
PetscErrorCode SVDMonitorConvergedDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;
  SlepcConvMon   mctx;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  ierr = PetscNew(&mctx);CHKERRQ(ierr);
  mctx->ctx = ctx;
  (*vf)->data = (void*)mctx;
  ierr = SVDMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"Convergence History","Number Converged",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

