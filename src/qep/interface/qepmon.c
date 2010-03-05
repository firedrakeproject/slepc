/*
      QEP routines related to monitors.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h"   /*I "slepcqep.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorSet"
/*@C
   QEPMonitorSet - Sets an ADDITIONAL function to be called at every 
   iteration to monitor the error estimates for each requested eigenpair.
      
   Collective on QEP

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
+    -qep_monitor        - print error estimates at each iteration
.    -qep_monitor_first  - print only the first error estimate
.    -qep_monitor_conv   - print the eigenvalue approximations only when
      convergence has been reached
.    -qep_monitor_draw   - sets line graph monitor
-    -qep_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to QEPMonitorSet(), but does not cancel those set via
      the options database.

   Notes:  
   Several different monitoring routines may be set by calling
   QEPMonitorSet() multiple times; all will be called in the 
   order in which they were set.

   Level: intermediate

.seealso: QEPMonitorDefault(), QEPMonitorCancel()
@*/
PetscErrorCode QEPMonitorSet(QEP qep,PetscErrorCode (*monitor)(QEP,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*),
                             void *mctx,PetscErrorCode (*monitordestroy)(void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (qep->numbermonitors >= MAXQEPMONITORS) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Too many QEP monitors set");
  }
  qep->monitor[qep->numbermonitors]           = monitor;
  qep->monitorcontext[qep->numbermonitors]    = (void*)mctx;
  qep->monitordestroy[qep->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorCancel"
/*@
   QEPMonitorCancel - Clears all monitors for a QEP object.

   Collective on QEP

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
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  for (i=0; i<qep->numbermonitors; i++) {
    if (qep->monitordestroy[i]) {
      ierr = (*qep->monitordestroy[i])(qep->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  qep->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetMonitorContext"
/*@C
   QEPGetMonitorContext - Gets the monitor context, as set by 
   QEPSetMonitor() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  qep - eigensolver context obtained from QEPCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: QEPSetMonitor(), QEPDefaultMonitor()
@*/
PetscErrorCode QEPGetMonitorContext(QEP qep, void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  *ctx =      (qep->monitorcontext[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPMonitorDefault"
/*@C
   QEPMonitorDefault - Print the current approximate values and 
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
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: QEPMonitorSet()
@*/
PetscErrorCode QEPMonitorDefault(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode          ierr;
  PetscInt                i;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (its) {
    if (!dummy) {ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,"stdout",0,&viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3d QEP nconv=%d Values (Errors)",its,nconv);CHKERRQ(ierr);
    for (i=0;i<nest;i++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIMonitorPrintf(viewer," %g%+gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIMonitorPrintf(viewer," %g",eigr[i]);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIMonitorPrintf(viewer,"%+gi",eigi[i]);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIMonitorPrintf(viewer," (%10.8e)",errest[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"\n");CHKERRQ(ierr);
    if (!dummy) {ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);}
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
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: QEPMonitorSet()
@*/
PetscErrorCode QEPMonitorFirst(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode          ierr;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (its && nconv<nest) {
    if (!dummy) {ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,"stdout",0,&viewer);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3d QEP nconv=%d first unconverged value (error)",its,nconv);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIMonitorPrintf(viewer," %g%+gi",PetscRealPart(eigr[nconv]),PetscImaginaryPart(eigr[nconv]));CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIMonitorPrintf(viewer," %g",eigr[nconv]);CHKERRQ(ierr);
    if (eigi[nconv]!=0.0) { ierr = PetscViewerASCIIMonitorPrintf(viewer,"%+gi",eigi[nconv]);CHKERRQ(ierr); }
#endif
    ierr = PetscViewerASCIIMonitorPrintf(viewer," (%10.8e)\n",errest[nconv]);CHKERRQ(ierr);
    if (!dummy) {ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);}
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
-  dummy  - unused monitor context 

   Level: intermediate

.seealso: QEPMonitorSet()
@*/
PetscErrorCode QEPMonitorConverged(QEP qep,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *dummy)
{
  PetscErrorCode          ierr;
  static PetscInt         oldnconv;
  PetscInt                i;
  PetscViewerASCIIMonitor viewer = (PetscViewerASCIIMonitor) dummy;

  PetscFunctionBegin;
  if (!its) {
    oldnconv = 0;
  } else {
    if (!dummy) {ierr = PetscViewerASCIIMonitorCreate(((PetscObject)qep)->comm,"stdout",0,&viewer);CHKERRQ(ierr);}
    for (i=oldnconv;i<nconv;i++) {
      ierr = PetscViewerASCIIMonitorPrintf(viewer,"%3d QEP converged value (error) #%d",its,i);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIMonitorPrintf(viewer," %g%+gi",PetscRealPart(eigr[i]),PetscImaginaryPart(eigr[i]));CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIMonitorPrintf(viewer," %g",eigr[i]);CHKERRQ(ierr);
      if (eigi[i]!=0.0) { ierr = PetscViewerASCIIMonitorPrintf(viewer,"%+gi",eigi[i]);CHKERRQ(ierr); }
#endif
      ierr = PetscViewerASCIIMonitorPrintf(viewer," (%10.8e)\n",errest[i]);CHKERRQ(ierr);
    }
    oldnconv = nconv;
    if (!dummy) {ierr = PetscViewerASCIIMonitorDestroy(viewer);CHKERRQ(ierr);}
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
  PetscReal      *x,*y;
  PetscInt       i;
  int            n = qep->nev;

  PetscFunctionBegin;

  if (!viewer) { viewer = PETSC_VIEWER_DRAW_(((PetscObject)qep)->comm); }

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
    if (errest[i] > 0.0) y[i] = log10(errest[i]); else y[i] = 0.0;
  }
  ierr = PetscDrawLGAddPoint(lg,x,y);CHKERRQ(ierr);

  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  ierr = PetscFree(y);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
} 

