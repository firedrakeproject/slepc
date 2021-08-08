/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   PEP routines related to monitors
*/

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/
#include <petscdraw.h>

PetscErrorCode PEPMonitorLGCreate(MPI_Comm comm,const char host[],const char label[],const char metric[],PetscInt l,const char *names[],int x,int y,int m,int n,PetscDrawLG *lgctx)
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
PetscErrorCode PEPMonitor(PEP pep,PetscInt it,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest)
{
  PetscErrorCode ierr;
  PetscInt       i,n = pep->numbermonitors;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = (*pep->monitor[i])(pep,it,nconv,eigr,eigi,errest,nest,pep->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorSet - Sets an ADDITIONAL function to be called at every
   iteration to monitor the error estimates for each requested eigenpair.

   Logically Collective on pep

   Input Parameters:
+  pep     - eigensolver context obtained from PEPCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context (may be NULL)

   Calling Sequence of monitor:
$   monitor(PEP pep,int its,int nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal* errest,int nest,void *mctx)

+  pep    - polynomial eigensolver context obtained from PEPCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - relative error estimates for each eigenpair
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by PEPMonitorSet()

   Options Database Keys:
+    -pep_monitor        - print only the first error estimate
.    -pep_monitor_all    - print error estimates at each iteration
.    -pep_monitor_conv   - print the eigenvalue approximations only when
      convergence has been reached
.    -pep_monitor draw::draw_lg - sets line graph monitor for the first unconverged
      approximate eigenvalue
.    -pep_monitor_all draw::draw_lg - sets line graph monitor for all unconverged
      approximate eigenvalues
.    -pep_monitor_conv draw::draw_lg - sets line graph monitor for convergence history
-    -pep_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to PEPMonitorSet(), but does not cancel those set via
      the options database.

   Notes:
   Several different monitoring routines may be set by calling
   PEPMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: PEPMonitorFirst(), PEPMonitorAll(), PEPMonitorCancel()
@*/
PetscErrorCode PEPMonitorSet(PEP pep,PetscErrorCode (*monitor)(PEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (pep->numbermonitors >= MAXPEPMONITORS) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_OUTOFRANGE,"Too many PEP monitors set");
  pep->monitor[pep->numbermonitors]           = monitor;
  pep->monitorcontext[pep->numbermonitors]    = (void*)mctx;
  pep->monitordestroy[pep->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

/*@
   PEPMonitorCancel - Clears all monitors for a PEP object.

   Logically Collective on pep

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Options Database Key:
.    -pep_monitor_cancel - Cancels all monitors that have been hardwired
      into a code by calls to PEPMonitorSet(),
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorCancel(PEP pep)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  for (i=0; i<pep->numbermonitors; i++) {
    if (pep->monitordestroy[i]) {
      ierr = (*pep->monitordestroy[i])(&pep->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  pep->numbermonitors = 0;
  PetscFunctionReturn(0);
}

/*@C
   PEPGetMonitorContext - Gets the monitor context, as set by
   PEPMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: PEPMonitorSet(), PEPDefaultMonitor()
@*/
PetscErrorCode PEPGetMonitorContext(PEP pep,void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  *(void**)ctx = pep->monitorcontext[0];
  PetscFunctionReturn(0);
}

/*
   Helper function to compute eigenvalue that must be viewed in monitor
 */
static PetscErrorCode PEPMonitorGetTrueEig(PEP pep,PetscScalar *er,PetscScalar *ei)
{
  PetscErrorCode ierr;
  PetscBool      flg,sinv;

  PetscFunctionBegin;
  ierr = STBackTransform(pep->st,1,er,ei);CHKERRQ(ierr);
  if (pep->sfactor!=1.0) {
    ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (flg && sinv) {
      *er /= pep->sfactor; *ei /= pep->sfactor;
    } else {
      *er *= pep->sfactor; *ei *= pep->sfactor;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorFirst - Print the first unconverged approximate value and
   error estimate at each iteration of the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor - activates PEPMonitorFirst()

   Level: intermediate

.seealso: PEPMonitorSet(), PEPMonitorAll(), PEPMonitorConverged()
@*/
PetscErrorCode PEPMonitorFirst(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscScalar    er,ei;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  if (its==1 && ((PetscObject)pep)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Eigenvalue approximations and residual norms for %s solve.\n",((PetscObject)pep)->prefix);CHKERRQ(ierr);
  }
  if (nconv<nest) {
    ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP nconv=%D first unconverged value (error)",its,nconv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    er = eigr[nconv]; ei = eigi[nconv];
    ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
    if (ei!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[nconv]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorAll - Print the current approximate values and
   error estimates at each iteration of the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor_all - activates PEPMonitorAll()

   Level: intermediate

.seealso: PEPMonitorSet(), PEPMonitorFirst(), PEPMonitorConverged()
@*/
PetscErrorCode PEPMonitorAll(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    er,ei;
  PetscViewer    viewer = vf->viewer;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
  if (its==1 && ((PetscObject)pep)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Eigenvalue approximations and residual norms for %s solve.\n",((PetscObject)pep)->prefix);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP nconv=%D Values (Errors)",its,nconv);CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
  for (i=0;i<nest;i++) {
    er = eigr[i]; ei = eigi[i];
    ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
    if (ei!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",(double)errest[i]);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorConverged - Print the approximate values and
   error estimates as they converge.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor_conv - activates PEPMonitorConverged()

   Level: intermediate

.seealso: PEPMonitorSet(), PEPMonitorFirst(), PEPMonitorAll()
@*/
PetscErrorCode PEPMonitorConverged(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    er,ei;
  PetscViewer    viewer = vf->viewer;
  SlepcConvMon   ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  ctx = (SlepcConvMon)vf->data;
  if (its==1 && ((PetscObject)pep)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Convergence history for %s solve.\n",((PetscObject)pep)->prefix);CHKERRQ(ierr);
  }
  if (its==1) ctx->oldnconv = 0;
  if (ctx->oldnconv!=nconv) {
    ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP converged value (error) #%D",its,i);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      er = eigr[i]; ei = eigi[i];
      ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
      if (ei!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PEPMonitorConvergedCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
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

PetscErrorCode PEPMonitorConvergedDestroy(PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*vf) PetscFunctionReturn(0);
  ierr = PetscFree((*vf)->data);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&(*vf)->viewer);CHKERRQ(ierr);
  ierr = PetscDrawLGDestroy(&(*vf)->lg);CHKERRQ(ierr);
  ierr = PetscFree(*vf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorFirstDrawLG - Plots the error estimate of the first unconverged
   approximation at each iteration of the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor draw::draw_lg - activates PEPMonitorFirstDrawLG()

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorFirstDrawLG(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;
  PetscDrawLG    lg = vf->lg;
  PetscReal      x,y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,PetscLog10Real(pep->tol)-2,0.0);CHKERRQ(ierr);
  }
  if (nconv<nest) {
    x = (PetscReal)its;
    if (errest[nconv] > 0.0) y = PetscLog10Real(errest[nconv]);
    else y = 0.0;
    ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
    if (its <= 20 || !(its % 5) || pep->reason) {
      ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
      ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorFirstDrawLGCreate - Creates the plotter for the first error estimate.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorFirstDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = PEPMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"First Error Estimate","Log Error Estimate",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorAllDrawLG - Plots the error estimate of all unconverged
   approximations at each iteration of the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor_all draw::draw_lg - activates PEPMonitorAllDrawLG()

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorAllDrawLG(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = vf->viewer;
  PetscDrawLG    lg = vf->lg;
  PetscInt       i,n = PetscMin(pep->nev,255);
  PetscReal      *x,*y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,PetscLog10Real(pep->tol)-2,0.0);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(n,&x,n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal)its;
    if (i < nest && errest[i] > 0.0) y[i] = PetscLog10Real(errest[i]);
    else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);
  if (its <= 20 || !(its % 5) || pep->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscFree2(x,y);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorAllDrawLGCreate - Creates the plotter for all the error estimates.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorAllDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  (*vf)->data = ctx;
  ierr = PEPMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"All Error Estimates","Log Error Estimate",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorConvergedDrawLG - Plots the number of converged eigenvalues
   at each iteration of the polynomial eigensolver.

   Collective on pep

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  vf     - viewer and format for monitoring

   Options Database Key:
.  -pep_monitor_conv draw::draw_lg - activates PEPMonitorConvergedDrawLG()

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorConvergedDrawLG(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,PetscViewerAndFormat *vf)
{
  PetscErrorCode   ierr;
  PetscViewer      viewer = vf->viewer;
  PetscDrawLG      lg = vf->lg;
  PetscReal        x,y;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,8);
  PetscValidHeaderSpecific(lg,PETSC_DRAWLG_CLASSID,8);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  if (its==1) {
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,1,1,0,pep->nev);CHKERRQ(ierr);
  }
  x = (PetscReal)its;
  y = (PetscReal)pep->nconv;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  if (its <= 20 || !(its % 5) || pep->reason) {
    ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSave(lg);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PEPMonitorConvergedDrawLGCreate - Creates the plotter for the convergence history.

   Collective on viewer

   Input Parameters:
+  viewer - the viewer
.  format - the viewer format
-  ctx    - an optional user context

   Output Parameter:
.  vf     - the viewer and format context

   Level: intermediate

.seealso: PEPMonitorSet()
@*/
PetscErrorCode PEPMonitorConvergedDrawLGCreate(PetscViewer viewer,PetscViewerFormat format,void *ctx,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;
  SlepcConvMon   mctx;

  PetscFunctionBegin;
  ierr = PetscViewerAndFormatCreate(viewer,format,vf);CHKERRQ(ierr);
  ierr = PetscNew(&mctx);CHKERRQ(ierr);
  mctx->ctx = ctx;
  (*vf)->data = (void*)mctx;
  ierr = PEPMonitorLGCreate(PetscObjectComm((PetscObject)viewer),NULL,"Convergence History","Number Converged",1,NULL,PETSC_DECIDE,PETSC_DECIDE,400,300,&(*vf)->lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

