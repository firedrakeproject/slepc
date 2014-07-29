/*
      MFN routines related to monitors.

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

#include <slepc-private/mfnimpl.h>   /*I "slepcmfn.h" I*/
#include <petscdraw.h>

#undef __FUNCT__
#define __FUNCT__ "MFNMonitor"
/*
   Runs the user provided monitor routines, if any.
*/
PetscErrorCode MFNMonitor(MFN mfn,PetscInt it,PetscReal errest)
{
  PetscErrorCode ierr;
  PetscInt       i,n = mfn->numbermonitors;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    ierr = (*mfn->monitor[i])(mfn,it,errest,mfn->monitorcontext[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNMonitorSet"
/*@C
   MFNMonitorSet - Sets an ADDITIONAL function to be called at every
   iteration to monitor convergence.

   Logically Collective on MFN

   Input Parameters:
+  mfn     - matrix function context obtained from MFNCreate()
.  monitor - pointer to function (if this is NULL, it turns off monitoring)
.  mctx    - [optional] context for private data for the
             monitor routine (use NULL if no context is desired)
-  monitordestroy - [optional] routine that frees monitor context (may be NULL)

   Calling Sequence of monitor:
$     monitor (MFN mfn, int its, PetscReal errest, void *mctx)

+  mfn    - matrix function context obtained from MFNCreate()
.  its    - iteration number
.  errest - error estimate
-  mctx   - optional monitoring context, as set by MFNMonitorSet()

   Options Database Keys:
+    -mfn_monitor        - print the error estimate
.    -mfn_monitor_lg     - sets line graph monitor for the error estimate
-    -mfn_monitor_cancel - cancels all monitors that have been hardwired into
      a code by calls to MFNMonitorSet(), but does not cancel those set via
      the options database.

   Notes:
   Several different monitoring routines may be set by calling
   MFNMonitorSet() multiple times; all will be called in the
   order in which they were set.

   Level: intermediate

.seealso: MFNMonitorFirst(), MFNMonitorAll(), MFNMonitorCancel()
@*/
PetscErrorCode MFNMonitorSet(MFN mfn,PetscErrorCode (*monitor)(MFN,PetscInt,PetscReal,void*),void *mctx,PetscErrorCode (*monitordestroy)(void**))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (mfn->numbermonitors >= MAXMFNMONITORS) SETERRQ(PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_OUTOFRANGE,"Too many MFN monitors set");
  mfn->monitor[mfn->numbermonitors]           = monitor;
  mfn->monitorcontext[mfn->numbermonitors]    = (void*)mctx;
  mfn->monitordestroy[mfn->numbermonitors++]  = monitordestroy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNMonitorCancel"
/*@
   MFNMonitorCancel - Clears all monitors for an MFN object.

   Logically Collective on MFN

   Input Parameters:
.  mfn - matrix function context obtained from MFNCreate()

   Options Database Key:
.    -mfn_monitor_cancel - Cancels all monitors that have been hardwired
      into a code by calls to MFNMonitorSet(),
      but does not cancel those set via the options database.

   Level: intermediate

.seealso: MFNMonitorSet()
@*/
PetscErrorCode MFNMonitorCancel(MFN mfn)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  for (i=0; i<mfn->numbermonitors; i++) {
    if (mfn->monitordestroy[i]) {
      ierr = (*mfn->monitordestroy[i])(&mfn->monitorcontext[i]);CHKERRQ(ierr);
    }
  }
  mfn->numbermonitors = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetMonitorContext"
/*@C
   MFNGetMonitorContext - Gets the monitor context, as set by
   MFNMonitorSet() for the FIRST monitor only.

   Not Collective

   Input Parameter:
.  mfn - matrix function context obtained from MFNCreate()

   Output Parameter:
.  ctx - monitor context

   Level: intermediate

.seealso: MFNMonitorSet()
@*/
PetscErrorCode MFNGetMonitorContext(MFN mfn,void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  *ctx = mfn->monitorcontext[0];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNMonitorDefault"
/*@C
   MFNMonitorDefault - Print the error estimate of the current approximation at each
   iteration of the matrix function solver.

   Collective on MFN

   Input Parameters:
+  mfn    - matrix function context
.  its    - iteration number
.  errest - error estimate
-  monctx - monitor context (contains viewer, can be NULL)

   Level: intermediate

.seealso: MFNMonitorSet()
@*/
PetscErrorCode MFNMonitorDefault(MFN mfn,PetscInt its,PetscReal errest,void *monctx)
{
  PetscErrorCode ierr;
  PetscViewer    viewer = monctx? (PetscViewer)monctx: PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)mfn));

  PetscFunctionBegin;
  ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)mfn)->tablevel);CHKERRQ(ierr);
  if (its == 0 && ((PetscObject)mfn)->prefix) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Monitor for %s solve.\n",((PetscObject)mfn)->prefix);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"%3D MFN value %14.12e\n",its,(double)errest);CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)mfn)->tablevel);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNMonitorLG"
PetscErrorCode MFNMonitorLG(MFN mfn,PetscInt its,PetscReal errest,void *monctx)
{
  PetscViewer    viewer = (PetscViewer)monctx;
  PetscDraw      draw;
  PetscDrawLG    lg;
  PetscErrorCode ierr;
  PetscReal      x,y;

  PetscFunctionBegin;
  if (!viewer) viewer = PETSC_VIEWER_DRAW_(PetscObjectComm((PetscObject)mfn));
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDrawLG(viewer,0,&lg);CHKERRQ(ierr);
  if (!its) {
    ierr = PetscDrawSetTitle(draw,"Error estimate");CHKERRQ(ierr);
    ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);
    ierr = PetscDrawLGSetDimension(lg,1);CHKERRQ(ierr);
    ierr = PetscDrawLGReset(lg);CHKERRQ(ierr);
    ierr = PetscDrawLGSetLimits(lg,0,1.0,PetscLog10Real(mfn->tol)-2,0.0);CHKERRQ(ierr);
  }
  x = (PetscReal)its;
  if (errest>0.0) y = PetscLog10Real(errest); else y = 0.0;
  ierr = PetscDrawLGAddPoint(lg,&x,&y);CHKERRQ(ierr);
  ierr = PetscDrawLGDraw(lg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

