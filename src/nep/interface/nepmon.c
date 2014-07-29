/*
      NEP routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/nepimpl.h>      /*I "slepcnep.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "NEPMonitor"
/*
   Runs the user provided monitor routines, if any.
*/
PetscErrorCode NEPMonitor(NEP nep,PetscInt it,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest)
{
  PetscErrorCode ierr;
  PetscInt       i,n = nep->numbermonitors;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = (*nep->monitor[i])(nep,it,nconv,eig,errest,nest,nep->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorSet"
/*@C
   NEPMonitorSet - Sets an ADDITIONAL function to be called at every
   iteration to monitor the error estimates for each requested eigenpair.

   Logically Collective on NEP

   Input Parameters:
+  nep     - eigensolver context obtained from NEPCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
             (may be NULL)

   Calling Sequence of monitor:
$     monitor (NEP nep, int its, int nconv, PetscScalar *eig, PetscReal* errest, int nest, void *mctx)

+  nep    - nonlinear eigensolver context obtained from NEPCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  eig    - eigenvalues
.  errest - error estimates for each eigenpair
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by NEPMonitorSet()

   Options Database Keys:
+    -nep_monitor        - print only the first error estimate
.    -nep_monitor_all    - print error estimates at each iteration
.    -nep_monitor_conv   - print the eigenvalue approximations only when
      convergence has been reached
.    -nep_monitor_lg     - sets line graph monitor for the first unconverged
      approximate eigenvalue
.    -nep_monitor_lg_all - sets line graph monitor for all unconverged
      approximate eigenvalues
-    -nep_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to NEPMonitorSet(), but does not cancel those set via
      the options database.

   Notes:
   Several different monitoring routines may be set by calling
   NEPMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: NEPMonitorFirst(), NEPMonitorAll(), NEPMonitorCancel()
@*/
PetscErrorCode NEPMonitorSet(NEP nep,PetscErrorCode (*monitor)(NEP,PetscInt,PetscInt,PetscScalar*,PetscReal*,PetscInt,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nep->numbermonitors >= MAXNEPMONITORS) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_OUTOFRANGE,"Too many NEP monitors set");
  nep->monitor[nep->numbermonitors]           = monitor;
  nep->monitorcontext[nep->numbermonitors]    = (void*)mctx;
  nep->monitordestroy[nep->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorCancel"
/*@
   NEPMonitorCancel - Clears all monitors for a NEP object.

   Logically Collective on NEP

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Options Database Key:
.    -nep_monitor_cancel - Cancels all monitors that have been hardwired
      into a code by calls to NEPMonitorSet(),
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: NEPMonitorSet()
@*/
PetscErrorCode NEPMonitorCancel(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  for (i=0; i<nep->numbermonitors; i++) {
    if (nep->monitordestroy[i]) {
      ierr = (*nep->monitordestroy[i])(&nep->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  nep->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetMonitorContext"
/*@C
   NEPGetMonitorContext - Gets the monitor context, as set by
   NEPMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: NEPMonitorSet(), NEPDefaultMonitor()
@*/
PetscErrorCode NEPGetMonitorContext(NEP nep,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  *ctx = nep->monitorcontext[0];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorAll"
/*@C
   NEPMonitorAll - Print the current approximate values and
   error estimates at each iteration of the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep    - nonlinear eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eig    - eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be NULL)

   Level: intermediate

.seealso: NEPMonitorSet(), NEPMonitorFirst(), NEPMonitorConverged()
@*/
PetscErrorCode NEPMonitorAll(NEP nep,PetscInt its,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));

  PetscFunctionBegin;
  if (its) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D NEP nconv=%D Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(eig[i]),(double)PetscImaginaryPart(eig[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %g",(double)eig[i]);CHKERRQ(ierr);
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",(double)errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorFirst"
/*@C
   NEPMonitorFirst - Print the first unconverged approximate value and
   error estimate at each iteration of the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep    - nonlinear eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eig    - eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be NULL)

   Level: intermediate

.seealso: NEPMonitorSet(), NEPMonitorAll(), NEPMonitorConverged()
@*/
PetscErrorCode NEPMonitorFirst(NEP nep,PetscInt its,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));

  PetscFunctionBegin;
  if (its && nconv<nest) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D NEP nconv=%D first unconverged value (error)",its,nconv);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(eig[nconv]),(double)PetscImaginaryPart(eig[nconv]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %g",(double)eig[nconv]);CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[nconv]);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorConverged"
/*@C
   NEPMonitorConverged - Print the approximate values and
   error estimates as they converge.

   Collective on NEP

   Input Parameters:
+  nep    - nonlinear eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eig    - eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context

   Level: intermediate

   Note:
   The monitor context must contain a struct with a PetscViewer and a
   PetscInt. In Fortran, pass a PETSC_NULL_OBJECT.

.seealso: NEPMonitorSet(), NEPMonitorFirst(), NEPMonitorAll()
@*/
PetscErrorCode NEPMonitorConverged(NEP nep,PetscInt its,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscViewer      viewer;
  SlepcConvMonitor ctx = (SlepcConvMonitor)monctx;

  PetscFunctionBegin;
  if (!monctx) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_WRONG,"Must provide a context for NEPMonitorConverged");
  if (!its) {
    ctx->oldnconv = 0;
  } else {
    viewer = ctx->viewer? ctx->viewer: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%3D NEP converged value (error) #%D",its,i);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %g%+gi",(double)PetscRealPart(eig[i]),(double)PetscImaginaryPart(eig[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %g",(double)eig[i]);CHKERRQ(ierr);
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)nep)->tablevel);CHKERRQ(ierr);
    }
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorLG"
PetscErrorCode NEPMonitorLG(NEP nep,PetscInt its,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer)monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)nep));
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,PetscLog10Real(nep->rtol)-2,0.0);CHKERRQ(ierr);
  }

  x = (PetscReal)its;
  if (errest[nconv] > 0.0) y = PetscLog10Real(errest[nconv]); else y = 0.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPMonitorLGAll"
PetscErrorCode NEPMonitorLGAll(NEP nep,PetscInt its,PetscInt nconv,PetscScalar *eig,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer)monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y;
  PetscInt       i,n = PetscMin(nep->nev,255);

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)nep));
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,PetscLog10Real(nep->rtol)-2,0.0);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(n,&x,n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal)its;
    if (i < nest && errest[i] > 0.0) y[i] = PetscLog10Real(errest[i]);
    else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscFree2(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

