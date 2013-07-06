/*
     The basic MFN routines, Create, View, etc. are here.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/mfnimpl.h>      /*I "slepcmfn.h" I*/

PetscFunctionList MFNList = 0;
PetscBool         MFNRegisterAllCalled = PETSC_FALSE;
PetscClassId      MFN_CLASSID = 0;
PetscLogEvent     MFN_SetUp = 0,MFN_Solve = 0;
static PetscBool  MFNPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "MFNFinalizePackage"
/*@C
  MFNFinalizePackage - This function destroys everything in the SLEPc interface
  to the MFN package. It is called from SlepcFinalize().

  Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode MFNFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&MFNList);CHKERRQ(ierr);
  MFNPackageInitialized = PETSC_FALSE;
  MFNRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNInitializePackage"
/*@C
  MFNInitializePackage - This function initializes everything in the MFN package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to MFNCreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode MFNInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MFNPackageInitialized) PetscFunctionReturn(0);
  MFNPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Matrix Function",&MFN_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MFNRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MFNSetUp",MFN_CLASSID,&MFN_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MFNSolve",MFN_CLASSID,&MFN_Solve);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"mfn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(MFN_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"mfn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(MFN_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(MFNFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  const char     *fun;
  char           str[50];
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
    if (mfn->function) {
      switch (mfn->function) {
        case SLEPC_FUNCTION_EXP: fun = "exponential"; break;
        default: SETERRQ(PetscObjectComm((PetscObject)mfn),1,"Wrong value of mfn->function");
      }
    } else fun = "not yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  function: %s\n",fun);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",mfn->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",mfn->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %G\n",mfn->tol);CHKERRQ(ierr);
    ierr = SlepcSNPrintfScalar(str,50,mfn->sfactor,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  scaling factor: %s\n",str);CHKERRQ(ierr);
  } else {
    if (mfn->ops->view) {
      ierr = (*mfn->ops->view)(mfn,viewer);CHKERRQ(ierr);
    }
  }
  if (!mfn->ip) { ierr = MFNGetIP(mfn,&mfn->ip);CHKERRQ(ierr); }
  ierr = IPView(mfn->ip,viewer);CHKERRQ(ierr);
  if (!mfn->ds) { ierr = MFNGetDS(mfn,&mfn->ds);CHKERRQ(ierr); }
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = DSView(mfn->ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNCreate"
/*@C
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
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = MFNInitializePackage();CHKERRQ(ierr);
#endif

  ierr = SlepcHeaderCreate(mfn,_p_MFN,struct _MFNOps,MFN_CLASSID,"MFN","Matrix Function","MFN",comm,MFNDestroy,MFNView);CHKERRQ(ierr);

  mfn->max_it          = 0;
  mfn->ncv             = 0;
  mfn->allocated_ncv   = 0;
  mfn->tol             = PETSC_DEFAULT;
  mfn->function        = (SlepcFunction)0;
  mfn->sfactor         = 1.0;

  mfn->A               = 0;
  mfn->V               = 0;
  mfn->t               = 0;
  mfn->errest          = 0;
  mfn->ip              = 0;
  mfn->ds              = 0;
  mfn->rand            = 0;
  mfn->data            = 0;
  mfn->its             = 0;

  mfn->nwork           = 0;
  mfn->work            = 0;
  mfn->setupcalled     = 0;
  mfn->reason          = MFN_CONVERGED_ITERATING;
  mfn->numbermonitors  = 0;

  ierr = PetscRandomCreate(comm,&mfn->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(mfn->rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(mfn,mfn->rand);CHKERRQ(ierr);
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
  if (mfn->ip) { ierr = IPReset(mfn->ip);CHKERRQ(ierr); }
  if (mfn->ds) { ierr = DSReset(mfn->ds);CHKERRQ(ierr); }
  ierr = VecDestroy(&mfn->t);CHKERRQ(ierr);
  mfn->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNDestroy"
/*@C
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
  ierr = IPDestroy(&(*mfn)->ip);CHKERRQ(ierr);
  ierr = DSDestroy(&(*mfn)->ds);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*mfn)->rand);CHKERRQ(ierr);
  ierr = MFNMonitorCancel(*mfn);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(mfn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetIP"
/*@
   MFNSetIP - Associates an inner product object to the matrix function solver.

   Collective on MFN

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  ip  - the inner product object

   Note:
   Use MFNGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: MFNGetIP()
@*/
PetscErrorCode MFNSetIP(MFN mfn,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(ip,IP_CLASSID,2);
  PetscCheckSameComm(mfn,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(&mfn->ip);CHKERRQ(ierr);
  mfn->ip = ip;
  ierr = PetscLogObjectParent(mfn,mfn->ip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetIP"
/*@C
   MFNGetIP - Obtain the inner product object associated to the eigensolver object.

   Not Collective

   Input Parameters:
.  mfn - matrix function context obtained from MFNCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: MFNSetIP()
@*/
PetscErrorCode MFNGetIP(MFN mfn,IP *ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(ip,2);
  if (!mfn->ip) {
    ierr = IPCreate(PetscObjectComm((PetscObject)mfn),&mfn->ip);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mfn,mfn->ip);CHKERRQ(ierr);
  }
  *ip = mfn->ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNSetDS"
/*@
   MFNSetDS - Associates a direct solver object to the matrix function solver.

   Collective on MFN

   Input Parameters:
+  mfn - matrix function context obtained from MFNCreate()
-  ds  - the direct solver object

   Note:
   Use MFNGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: MFNGetDS()
@*/
PetscErrorCode MFNSetDS(MFN mfn,DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(mfn,1,ds,2);
  ierr = PetscObjectReference((PetscObject)ds);CHKERRQ(ierr);
  ierr = DSDestroy(&mfn->ds);CHKERRQ(ierr);
  mfn->ds = ds;
  ierr = PetscLogObjectParent(mfn,mfn->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MFNGetDS"
/*@C
   MFNGetDS - Obtain the direct solver object associated to the matrix function object.

   Not Collective

   Input Parameters:
.  mfn - matrix function context obtained from MFNCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: MFNSetDS()
@*/
PetscErrorCode MFNGetDS(MFN mfn,DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mfn,MFN_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!mfn->ds) {
    ierr = DSCreate(PetscObjectComm((PetscObject)mfn),&mfn->ds);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(mfn,mfn->ds);CHKERRQ(ierr);
  }
  *ds = mfn->ds;
  PetscFunctionReturn(0);
}

