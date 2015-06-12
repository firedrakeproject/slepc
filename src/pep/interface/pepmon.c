/*
      PEP routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2015, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/pepimpl.h>      /*I "slepcpep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "PEPMonitor"
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

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorSet"
/*@C
   PEPMonitorSet - Sets an ADDITIONAL function to be called at every
   iteration to monitor the error estimates for each requested eigenpair.

   Logically Collective on PEP

   Input Parameters:
+  pep     - eigensolver context obtained from PEPCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context (may be NULL)

   Calling Sequence of monitor:
$     monitor (PEP pep, int its, int nconv, PetscScalar *eigr, PetscScalar *eigi, PetscReal* errest, int nest, void *mctx)

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
.    -pep_monitor_lg     - sets line graph monitor for the first unconverged
      approximate eigenvalue
.    -pep_monitor_lg_all - sets line graph monitor for all unconverged
      approximate eigenvalues
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

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorCancel"
/*@
   PEPMonitorCancel - Clears all monitors for a PEP object.

   Logically Collective on PEP

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

#undef __FUNCT__
#define __FUNCT__ "PEPGetMonitorContext"
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
PetscErrorCode PEPGetMonitorContext(PEP pep,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  *ctx = pep->monitorcontext[0];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorGetTrueEig"
/*
   Helper function to compute eigenvalue that must be viewed in monitor
 */
static PetscErrorCode PEPMonitorGetTrueEig(PEP pep,PetscScalar *er,PetscScalar *ei)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = STGetTransform(pep->st,&flg);CHKERRQ(ierr);
  if (flg) {
    *er *= pep->sfactor; *ei *= pep->sfactor;
  }
  ierr = STBackTransform(pep->st,1,er,ei);CHKERRQ(ierr);
  if (!flg) {
    *er *= pep->sfactor; *ei *= pep->sfactor;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorAll"
/*@C
   PEPMonitorAll - Print the current approximate values and
   error estimates at each iteration of the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be NULL)

   Level: intermediate

.seealso: PEPMonitorSet(), PEPMonitorFirst(), PEPMonitorConverged()
@*/
PetscErrorCode PEPMonitorAll(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    er,ei;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));

  PetscFunctionBegin;
  if (its) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP nconv=%D Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
      er = eigr[i]; ei = eigi[i];
      ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",(double)errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorFirst"
/*@C
   PEPMonitorFirst - Print the first unconverged approximate value and
   error estimate at each iteration of the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be NULL)

   Level: intermediate

.seealso: PEPMonitorSet(), PEPMonitorAll(), PEPMonitorConverged()
@*/
PetscErrorCode PEPMonitorFirst(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscScalar    er,ei;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));

  PetscFunctionBegin;
  if (its && nconv<nest) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP nconv=%D first unconverged value (error)",its,nconv);CHKERRQ(ierr);
    er = eigr[nconv]; ei = eigi[nconv];
    ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
    if (eigi[nconv]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[nconv]);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorConverged"
/*@C
   PEPMonitorConverged - Print the approximate values and
   error estimates as they converge.

   Collective on PEP

   Input Parameters:
+  pep    - polynomial eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context

   Level: intermediate

   Note:
   The monitor context must contain a struct with a PetscViewer and a
   PetscInt. In Fortran, pass a PETSC_NULL_OBJECT.

.seealso: PEPMonitorSet(), PEPMonitorFirst(), PEPMonitorAll()
@*/
PetscErrorCode PEPMonitorConverged(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscScalar      er,ei;
  PetscViewer      viewer;
  SlepcConvMonitor ctx = (SlepcConvMonitor)monctx;

  PetscFunctionBegin;
  if (!monctx) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONG,"Must provide a context for PEPMonitorConverged");
  if (!its) {
    ctx->oldnconv = 0;
  } else {
    viewer = ctx->viewer? ctx->viewer: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%3D PEP converged value (error) #%D",its,i);CHKERRQ(ierr);
      er = eigr[i]; ei = eigi[i];
      ierr = PEPMonitorGetTrueEig(pep,&er,&ei);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(er),(double)PetscImaginaryPart(er));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %g",(double)er);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+gi",(double)ei);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)pep)->tablevel);CHKERRQ(ierr);
    }
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorLG"
PetscErrorCode PEPMonitorLG(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer)monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)pep));
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(pep->tol)-2,0.0);CHKERRQ(ierr);
  }

  x = (PetscReal)its;
  if (errest[nconv] > 0.0) y = log10(errest[nconv]); else y = 0.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPMonitorLGAll"
PetscErrorCode PEPMonitorLGAll(PEP pep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer)monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y;
  PetscInt       i,n = PetscMin(pep->nev,255);

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)pep));
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(pep->tol)-2,0.0);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(n,&x,n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal)its;
    if (i < nest && errest[i] > 0.0) y[i] = log10(errest[i]);
    else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscFree2(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

