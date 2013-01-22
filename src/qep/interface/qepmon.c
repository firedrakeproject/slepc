/*
      QEP routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/qepimpl.h>      /*I "slepcqep.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitor"
/*
   Runs the user provided monitor routines, if any.
*/
PetscErrorCode QEPMonitor(QEP qep,PetscInt it,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest)
{
  PetscErrorCode ierr;
  PetscInt       i,n = qep->numbermonitors;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = (*qep->monitor[i])(qep,it,nconv,eigr,eigi,errest,nest,qep->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorSet"
/*@C
   QEPMonitorSet - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the error estimates for each requested eigenpair.
      
   Logically Collective on QEP

   Input Parameters:
+  qep     - eigensolver context obtained from QEPCreate()
.  monitor - pointer to function (if this is PETSC_NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use PETSC_NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context
          (may be PETSC_NULL)

   Calling Sequence of monitor:
$     monitor (QEP qep, int its, int nconv, PetscScalar *eigr, PetscScalar *eigi, PetscReal* errest, int nest, void *mctx)

+  qep    - quadratic eigensolver context obtained from QEPCreate()
.  its    - iteration number
.  nconv  - number of converged eigenpairs
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - relative error estimates for each eigenpair
.  nest   - number of error estimates
-  mctx   - optional monitoring context, as set by QEPMonitorSet()

   Options Database Keys:
+    -qep_monitor          - print only the first error estimate
.    -qep_monitor_all      - print error estimates at each iteration
.    -qep_monitor_conv     - print the eigenvalue approximations only when
      convergence has been reached
.    -qep_monitor_draw     - sets line graph monitor for the first unconverged
      approximate eigenvalue
.    -qep_monitor_draw_all - sets line graph monitor for all unconverged
      approximate eigenvalue
-    -qep_monitor_cancel   - cancels all monitors that have been hardwired into
      a code by calls to QEPMonitorSet(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   QEPMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: QEPMonitorFirst(), QEPMonitorAll(), QEPMonitorCancel()
@*/
PetscErrorCode QEPMonitorSet(QEP qep,PetscErrorCode (*monitor)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  if (qep->numbermonitors >= MAXQEPMONITORS) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Too many QEP monitors set");
  qep->monitor[qep->numbermonitors]           = monitor;
  qep->monitorcontext[qep->numbermonitors]    = (void*)mctx;
  qep->monitordestroy[qep->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorCancel"
/*@
   QEPMonitorCancel - Clears all monitors for a QEP object.

   Logically Collective on QEP

   Input Parameters:
.  qep - eigensolver context obtained from QEPCreate()

   Options Database Key:
.    -qep_monitor_cancel - Cancels all monitors that have been hardwired 
      into a code by calls to QEPMonitorSet(),
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: QEPMonitorSet()
@*/
PetscErrorCode QEPMonitorCancel(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  for (i=0; i<qep->numbermonitors; i++) {
    if (qep->monitordestroy[i]) {
      ierr = (*qep->monitordestroy[i])(&qep->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  qep->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetMonitorContext"
/*@C
   QEPGetMonitorContext - Gets the monitor context, as set by 
   QEPMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: QEPMonitorSet(), QEPDefaultMonitor()
@*/
PetscErrorCode QEPGetMonitorContext(QEP qep,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_CLASSID,1);
  *ctx =      (qep->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorAll"
/*@C
   QEPMonitorAll - Print the current approximate values and 
   error estimates at each iteration of the quadratic eigensolver.

   Collective on QEP

   Input Parameters:
+  qep    - quadratic eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be PETSC_NULL)

   Level: intermediate

.seealso: QEPMonitorSet(), QEPMonitorFirst(), QEPMonitorConverged()
@*/
PetscErrorCode QEPMonitorAll(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);

  PetscFunctionBegin;
  if (its) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D QEP nconv=%D Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %G%+Gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %G",eigr[i]);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+Gi",eigi[i]);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)",(double)errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorFirst"
/*@C
   QEPMonitorFirst - Print the first unconverged approximate value and 
   error estimate at each iteration of the quadratic eigensolver.

   Collective on QEP

   Input Parameters:
+  qep    - quadratic eigensolver context
.  its    - iteration number
.  nconv  - number of converged eigenpairs so far
.  eigr   - real part of the eigenvalues
.  eigi   - imaginary part of the eigenvalues
.  errest - error estimates
.  nest   - number of error estimates to display
-  monctx - monitor context (contains viewer, can be PETSC_NULL)

   Level: intermediate

.seealso: QEPMonitorSet(), QEPMonitorAll(), QEPMonitorConverged()
@*/
PetscErrorCode QEPMonitorFirst(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);

  PetscFunctionBegin;
  if (its && nconv<nest) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%3D QEP nconv=%D first unconverged value (error)",its,nconv);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer," %G%+Gi",PetscRealPart(eigr[nconv]),PetscImaginaryPart(eigr[nconv]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer," %G",eigr[nconv]);CHKERRQ(ierr);
    if (eigi[nconv]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+Gi",eigi[nconv]);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[nconv]);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorConverged"
/*@C
   QEPMonitorConverged - Print the approximate values and 
   error estimates as they converge.

   Collective on QEP

   Input Parameters:
+  qep    - quadratic eigensolver context
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

.seealso: QEPMonitorSet(), QEPMonitorFirst(), QEPMonitorAll()
@*/
PetscErrorCode QEPMonitorConverged(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscErrorCode   ierr;
  PetscInt         i;
  PetscViewer      viewer;
  SlepcConvMonitor ctx = (SlepcConvMonitor)monctx;

  PetscFunctionBegin;
  if (!monctx) SETERRQ(((PetscObject)qep)->comm,PETSC_ERR_ARG_WRONG,"Must provide a context for QEPMonitorConverged");
  if (!its) {
    ctx->oldnconv = 0;
  } else {
    viewer = ctx->viewer? ctx->viewer: PETSC_VIEWER_STDOUT_(((PetscObject)qep)->comm);
    for (i=ctx->oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%3D QEP converged value (error) #%D",its,i);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer," %G%+Gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer," %G",eigr[i]);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIPrintf(viewer,"%+Gi",eigi[i]);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIPrintf(viewer," (%10.8e)\n",(double)errest[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)qep)->tablevel);CHKERRQ(ierr);
    }
    ctx->oldnconv = nconv;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorLG"
PetscErrorCode QEPMonitorLG(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(((PetscObject)qep)->comm);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(qep->tol)-2,0.0);CHKERRQ(ierr);
  }

  x = (PetscReal) its;
  if (errest[nconv] > 0.0) y = log10(errest[nconv]); else y = 0.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorLGAll"
PetscErrorCode QEPMonitorLGAll(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer) monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      *x,*y;
  PetscInt       i,n = PetscMin(qep->nev,255);

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(((PetscObject)qep)->comm);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimates");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,n);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,log10(qep->tol)-2,0.0);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(sizeof(PetscReal)*n,&x);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscReal)*n,&y);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    x[i] = (PetscReal) its;
    if (i < nest && errest[i] > 0.0) y[i] = log10(errest[i]);
    else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
} 

