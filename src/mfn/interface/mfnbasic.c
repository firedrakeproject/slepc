/*
     The basic MFN routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2016, Universitat Politecnica de Valencia, Spain

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

#include <slepc/private/mfnimpl.h>      /*I "slepcmfn.h" I*/

PetscFunctionList MFNList = 0;
PetscBool         MFNRegisterAllCalled = PETSC_FALSE;
PetscClassId      MFN_CLASSID = 0;
PetscLogEvent     MFN_SetUp = 0,MFN_Solve = 0;

#undef __FUNCT__
#define __FUNCT__ "MFNView"
/*@C
   MFNView - Prints the MFN data structure.

   Collective on MFN

   Input Parameters:
+  mfn - the matrix function solver context
-  viewer - optional visualization context

   Options Database Key:
.  -mfn_view -  Calls MFNView() at end of MFNSolve()

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.

   Level: beginner

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode MFNView(MFN mfn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)mfn));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(mfn,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)mfn,viewer);CHKERRQ(ierr);
    if (mfn->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*mfn->ops->view)(mfn,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",mfn->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",mfn->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)mfn->tol);CHKERRQ(ierr);
  } else {
    if (mfn->ops->view) {
      ierr = (*mfn->ops->view)(mfn,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  if (!mfn->V) { ierr = MFNGetFN(mfn,&mfn->fn);CHKERRQ(ierr); }
  ierr = FNView(mfn->fn,viewer);CHKERRQ(ierr);
  if (!mfn->V) { ierr = MFNGetBV(mfn,&mfn->V);CHKERRQ(ierr); }
  ierr = BVView(mfn->V,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNReasonView"
/*@C
   MFNReasonView - Displays the reason an MFN solve converged or diverged.

   Collective on MFN

   Parameter:
+  mfn - the matrix function context
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -mfn_converged_reason - print reason for convergence, and number of iterations

   Level: intermediate

.seealso: MFNSetTolerances(), MFNGetIterationNumber()
@*/
PetscErrorCode MFNReasonView(MFN mfn,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isAscii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerASCIIAddTab(viewer,((PetscObject)mfn)->tablevel);CHKERRQ(ierr);
    if (mfn->reason > 0) {
      ierr = PetscViewerASCIIPrintf(viewer,"%s Matrix function solve converged due to %s; iterations %D\n",((PetscObject)mfn)->prefix?((PetscObject)mfn)->prefix:"",MFNConvergedReasons[mfn->reason],mfn->its);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"%s Matrix function solve did not converge due to %s; iterations %D\n",((PetscObject)mfn)->prefix?((PetscObject)mfn)->prefix:"",MFNConvergedReasons[mfn->reason],mfn->its);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISubtractTab(viewer,((PetscObject)mfn)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNReasonViewFromOptions"
/*@
   MFNReasonViewFromOptions - Processes command line options to determine if/how
   the MFN converged reason is to be viewed. 

   Collective on MFN

   Input Parameters:
.  mfn - the matrix function context

   Level: developer
@*/
PetscErrorCode MFNReasonViewFromOptions(MFN mfn)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr = PetscOptionsGetViewer(PetscObjectComm((PetscObject)mfn),((PetscObject)mfn)->prefix,"-mfn_converged_reason",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = MFNReasonView(mfn,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNCreate"
/*@
   MFNCreate - Creates the default MFN context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  mfn - location to put the MFN context

   Note:
   The default MFN type is MFNKRYLOV

   Level: beginner

.seealso: MFNSetUp(), MFNSolve(), MFNDestroy(), MFN
@*/
PetscErrorCode MFNCreate(MPI_Comm comm,MFN *outmfn)
{
  PetscErrorCode ierr;
  MFN            mfn;

  PetscFunctionBegin;
  PetscValidPointer(outmfn,2);
  *outmfn = 0;
  ierr = MFNInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(mfn,MFN_CLASSID,"MFN","Matrix Function","MFN",comm,MFNDestroy,MFNView);CHKERRQ(ierr);

  mfn->A               = NULL;
  mfn->fn              = NULL;
  mfn->max_it          = 0;
  mfn->ncv             = 0;
  mfn->tol             = PETSC_DEFAULT;
  mfn->errorifnotconverged = PETSC_FALSE;

  mfn->numbermonitors  = 0;

  mfn->V               = NULL;
  mfn->nwork           = 0;
  mfn->work            = NULL;
  mfn->data            = NULL;

  mfn->its             = 0;
  mfn->nv              = 0;
  mfn->errest          = 0;
  mfn->setupcalled     = 0;
  mfn->reason          = MFN_CONVERGED_ITERATING;

  *outmfn = mfn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetType"
/*@C
   MFNSetType - Selects the particular solver to be used in the MFN object.

   Logically Collective on MFN

   Input Parameters:
+  mfn  - the matrix function context
-  type - a known method

   Options Database Key:
.  -mfn_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepcmfn.h" for available methods. The default
   is MFNKRYLOV

   Normally, it is best to use the MFNSetFromOptions() command and
   then set the MFN type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The MFNSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: MFNType
@*/
PetscErrorCode MFNSetType(MFN mfn,MFNType type)
{
  PetscErrorCode ierr,(*r)(MFN);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)mfn,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(MFNList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)mfn),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown MFN type given: %s",type);

  if (mfn->ops->destroy) { ierr = (*mfn->ops->destroy)(mfn);CHKERRQ(ierr); }
  ierr = PetscMemzero(mfn->ops,sizeof(struct _MFNOps));CHKERRQ(ierr);

  mfn->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)mfn,type);CHKERRQ(ierr);
  ierr = (*r)(mfn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetType"
/*@C
   MFNGetType - Gets the MFN type as a string from the MFN object.

   Not Collective

   Input Parameter:
.  mfn - the matrix function context

   Output Parameter:
.  name - name of MFN method

   Level: intermediate

.seealso: MFNSetType()
@*/
PetscErrorCode MFNGetType(MFN mfn,MFNType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)mfn)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNRegister"
/*@C
   MFNRegister - Adds a method to the matrix function solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   MFNRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
    MFNRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MFNSetType(mfn,"my_solver")
   or at runtime via the option
$     -mfn_type my_solver

   Level: advanced

.seealso: MFNRegisterAll()
@*/
PetscErrorCode MFNRegister(const char *name,PetscErrorCode (*function)(MFN))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&MFNList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNReset"
/*@
   MFNReset - Resets the MFN context to the setupcalled=0 state and removes any
   allocated objects.

   Collective on MFN

   Input Parameter:
.  mfn - matrix function context obtained from MFNCreate()

   Level: advanced

.seealso: MFNDestroy()
@*/
PetscErrorCode MFNReset(MFN mfn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  if (mfn->ops->reset) { ierr = (mfn->ops->reset)(mfn);CHKERRQ(ierr); }
  mfn->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNDestroy"
/*@
   MFNDestroy - Destroys the MFN context.

   Collective on MFN

   Input Parameter:
.  mfn - matrix function context obtained from MFNCreate()

   Level: beginner

.seealso: MFNCreate(), MFNSetUp(), MFNSolve()
@*/
PetscErrorCode MFNDestroy(MFN *mfn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mfn) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*mfn,MFN_CLASSID,1);
  if (--((PetscObject)(*mfn))->refct > 0) { *mfn = 0; PetscFunctionReturn(0); }
  ierr = MFNReset(*mfn);CHKERRQ(ierr);
  if ((*mfn)->ops->destroy) { ierr = (*(*mfn)->ops->destroy)(*mfn);CHKERRQ(ierr); }
  ierr = MatDestroy(&(*mfn)->A);CHKERRQ(ierr);
  ierr = BVDestroy(&(*mfn)->V);CHKERRQ(ierr);
  ierr = FNDestroy(&(*mfn)->fn);CHKERRQ(ierr);
  ierr = MFNMonitorCancel(*mfn);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(mfn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetBV"
/*@
   MFNSetBV - Associates a basis vectors object to the matrix function solver.

   Collective on MFN

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  bv  - the basis vectors object

   Note:
   Use MFNGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: MFNGetBV()
@*/
PetscErrorCode MFNSetBV(MFN mfn,BV bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(mfn,1,bv,2);
  ierr = PetscObjectReference((PetscObject)bv);CHKERRQ(ierr);
  ierr = BVDestroy(&mfn->V);CHKERRQ(ierr);
  mfn->V = bv;
  ierr = PetscLogObjectParent((PetscObject)mfn,(PetscObject)mfn->V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetBV"
/*@
   MFNGetBV - Obtain the basis vectors object associated to the matrix
   function solver.

   Not Collective

   Input Parameters:
.  mfn - matrix function context obtained from MFNCreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: MFNSetBV()
@*/
PetscErrorCode MFNGetBV(MFN mfn,BV *bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!mfn->V) {
    ierr = BVCreate(PetscObjectComm((PetscObject)mfn),&mfn->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mfn,(PetscObject)mfn->V);CHKERRQ(ierr);
  }
  *bv = mfn->V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetFN"
/*@
   MFNSetFN - Specifies the function to be computed.

   Collective on MFN

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  fn  - the math function object

   Note:
   Use MFNGetFN() to retrieve the math function context (for example,
   to free it at the end of the computations).

   Level: beginner

.seealso: MFNGetFN()
@*/
PetscErrorCode MFNSetFN(MFN mfn,FN fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(fn,FN_CLASSID,2);
  PetscCheckSameComm(mfn,1,fn,2);
  ierr = PetscObjectReference((PetscObject)fn);CHKERRQ(ierr);
  ierr = FNDestroy(&mfn->fn);CHKERRQ(ierr);
  mfn->fn = fn;
  ierr = PetscLogObjectParent((PetscObject)mfn,(PetscObject)mfn->fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetFN"
/*@
   MFNGetFN - Obtain the math function object associated to the MFN object.

   Not Collective

   Input Parameters:
.  mfn - matrix function context obtained from MFNCreate()

   Output Parameter:
.  fn - math function context

   Level: beginner

.seealso: MFNSetFN()
@*/
PetscErrorCode MFNGetFN(MFN mfn,FN *fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(fn,2);
  if (!mfn->fn) {
    ierr = FNCreate(PetscObjectComm((PetscObject)mfn),&mfn->fn);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)mfn,(PetscObject)mfn->fn);CHKERRQ(ierr);
  }
  *fn = mfn->fn;
  PetscFunctionReturn(0);
}

