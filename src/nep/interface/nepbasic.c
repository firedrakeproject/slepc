/*
   Basic NEP routines, Create, View, etc.

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

#include <slepc-private/nepimpl.h>      /*I "slepcnep.h" I*/

PetscFunctionList NEPList = 0;
PetscBool         NEPRegisterAllCalled = PETSC_FALSE;
PetscClassId      NEP_CLASSID = 0;
PetscLogEvent     NEP_SetUp = 0,NEP_Solve = 0,NEP_Dense = 0,NEP_FunctionEval = 0,NEP_JacobianEval = 0;
static PetscBool  NEPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "NEPFinalizePackage"
/*@C
   NEPFinalizePackage - This function destroys everything in the Slepc interface
   to the NEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode NEPFinalizePackage(void)
{
  PetscFunctionBegin;
  NEPPackageInitialized = PETSC_FALSE;
  NEPList               = 0;
  NEPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPInitializePackage"
/*@C
   NEPInitializePackage - This function initializes everything in the NEP package. It is called
   from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to NEPCreate()
   when using static libraries.

   Input Parameter:
.  path - The dynamic library path, or NULL

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode NEPInitializePackage(const char *path)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (NEPPackageInitialized) PetscFunctionReturn(0);
  NEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Nonlinear Eigenvalue Problem solver",&NEP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = NEPRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("NEPSetUp",NEP_CLASSID,&NEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPSolve",NEP_CLASSID,&NEP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPDense",NEP_CLASSID,&NEP_Dense);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPFunctionEval",NEP_CLASSID,&NEP_FunctionEval);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("NEPJacobianEval",NEP_CLASSID,&NEP_JacobianEval);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"nep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(NEP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"nep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(NEP_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(NEPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPView"
/*@C
   NEPView - Prints the NEP data structure.

   Collective on NEP

   Input Parameters:
+  nep - the nonlinear eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -nep_view -  Calls NEPView() at end of NEPSolve()

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
PetscErrorCode NEPView(NEP nep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  char           str[50];
  PetscBool      isascii,isslp;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)nep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(nep,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)nep,viewer,"NEP Object");CHKERRQ(ierr);
    if (nep->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*nep->ops->view)(nep,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (nep->split) {
      ierr = PetscViewerASCIIPrintf(viewer,"  nonlinear operator in split form\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  nonlinear operator from user callbacks\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    ierr = SlepcSNPrintfScalar(str,50,nep->target,PETSC_FALSE);CHKERRQ(ierr);
    if (!nep->which) {
      ierr = PetscViewerASCIIPrintf(viewer,"not yet set\n");CHKERRQ(ierr);
    } else switch (nep->which) {
      case NEP_TARGET_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str);CHKERRQ(ierr);
        break;
      case NEP_TARGET_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str);CHKERRQ(ierr);
        break;
      case NEP_TARGET_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str);CHKERRQ(ierr);
        break;
      case NEP_LARGEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case NEP_LARGEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"largest real parts\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest real parts\n");CHKERRQ(ierr);
        break;
      case NEP_LARGEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n");CHKERRQ(ierr);
        break;
      case NEP_SMALLEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)nep),1,"Wrong value of nep->which");
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",nep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",nep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",nep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",nep->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of function evaluations: %D\n",nep->max_funcs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerances: relative=%G, absolute=%G, solution=%G\n",nep->rtol,nep->abstol,nep->stol);CHKERRQ(ierr);
    if (nep->lag) {
      ierr = PetscViewerASCIIPrintf(viewer,"  updating the preconditioner every %D iterations\n",nep->lag);CHKERRQ(ierr);
    }
    if (nep->cctol) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using a constant tolerance for the linear solver\n");CHKERRQ(ierr);
    }
    if (nep->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(nep->nini));CHKERRQ(ierr);
    }
  } else {
    if (nep->ops->view) {
      ierr = (*nep->ops->view)(nep,viewer);CHKERRQ(ierr);
    }
  }
  if (!nep->ip) { ierr = NEPGetIP(nep,&nep->ip);CHKERRQ(ierr); }
  ierr = IPView(nep->ip,viewer);CHKERRQ(ierr);
  if (!nep->ds) { ierr = NEPGetDS(nep,&nep->ds);CHKERRQ(ierr); }
  ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = DSView(nep->ds,viewer);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)nep,NEPSLP,&isslp);CHKERRQ(ierr);
  if (!isslp) {
    if (!nep->ksp) { ierr = NEPGetKSP(nep,&nep->ksp);CHKERRQ(ierr); }
    ierr = KSPView(nep->ksp,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPCreate"
/*@C
   NEPCreate - Creates the default NEP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  nep - location to put the NEP context

   Level: beginner

.seealso: NEPSetUp(), NEPSolve(), NEPDestroy(), NEP
@*/
PetscErrorCode NEPCreate(MPI_Comm comm,NEP *outnep)
{
  PetscErrorCode ierr;
  NEP            nep;

  PetscFunctionBegin;
  PetscValidPointer(outnep,2);
  *outnep = 0;
  ierr = SlepcHeaderCreate(nep,_p_NEP,struct _NEPOps,NEP_CLASSID,"NEP","Nonlinear Eigenvalue Problem","NEP",comm,NEPDestroy,NEPView);CHKERRQ(ierr);

  nep->max_it          = 0;
  nep->max_funcs       = 0;
  nep->nev             = 1;
  nep->ncv             = 0;
  nep->mpd             = 0;
  nep->lag             = 1;
  nep->nini            = 0;
  nep->allocated_ncv   = 0;
  nep->ip              = 0;
  nep->ds              = 0;
  nep->function        = 0;
  nep->function_pre    = 0;
  nep->jacobian        = 0;
  nep->abstol          = PETSC_DEFAULT;
  nep->rtol            = PETSC_DEFAULT;
  nep->stol            = PETSC_DEFAULT;
  nep->ktol            = 0.1;
  nep->cctol           = PETSC_FALSE;
  nep->ttol            = 0.0;
  nep->which           = (NEPWhich)0;
  nep->computefunction = NULL;
  nep->computejacobian = NULL;
  nep->comparison      = NULL;
  nep->converged       = NEPConvergedDefault;
  nep->convergeddestroy= NULL;
  nep->comparisonctx   = NULL;
  nep->convergedctx    = NULL;
  nep->functionctx     = NULL;
  nep->jacobianctx     = NULL;
  nep->V               = NULL;
  nep->IS              = NULL;
  nep->eigr            = NULL;
  nep->eigi            = NULL;
  nep->errest          = NULL;
  nep->data            = NULL;
  nep->t               = NULL;
  nep->split           = PETSC_FALSE;
  nep->nt              = 0;
  nep->mstr            = DIFFERENT_NONZERO_PATTERN;
  nep->A               = NULL;
  nep->f               = NULL;
  nep->nconv           = 0;
  nep->its             = 0;
  nep->perm            = NULL;
  nep->nfuncs          = 0;
  nep->linits          = 0;
  nep->nwork           = 0;
  nep->work            = NULL;
  nep->setupcalled     = 0;
  nep->reason          = NEP_CONVERGED_ITERATING;
  nep->numbermonitors  = 0;
  nep->trackall        = PETSC_FALSE;
  nep->rand            = 0;

  ierr = PetscRandomCreate(comm,&nep->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(nep->rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(nep,nep->rand);CHKERRQ(ierr);
  *outnep = nep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetType"
/*@C
   NEPSetType - Selects the particular solver to be used in the NEP object.

   Logically Collective on NEP

   Input Parameters:
+  nep      - the nonlinear eigensolver context
-  type     - a known method

   Options Database Key:
.  -nep_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepcnep.h" for available methods.

   Normally, it is best to use the NEPSetFromOptions() command and
   then set the NEP type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The NEPSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: NEPType
@*/
PetscErrorCode NEPSetType(NEP nep,NEPType type)
{
  PetscErrorCode ierr,(*r)(NEP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)nep,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PetscObjectComm((PetscObject)nep),NEPList,type,PETSC_TRUE,(void (**)(void))&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown NEP type given: %s",type);

  if (nep->ops->destroy) { ierr = (*nep->ops->destroy)(nep);CHKERRQ(ierr); }
  ierr = PetscMemzero(nep->ops,sizeof(struct _NEPOps));CHKERRQ(ierr);

  nep->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)nep,type);CHKERRQ(ierr);
  ierr = (*r)(nep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetType"
/*@C
   NEPGetType - Gets the NEP type as a string from the NEP object.

   Not Collective

   Input Parameter:
.  nep - the eigensolver context

   Output Parameter:
.  name - name of NEP method

   Level: intermediate

.seealso: NEPSetType()
@*/
PetscErrorCode NEPGetType(NEP nep,NEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)nep)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRegister"
/*@C
   NEPRegister - Adds a method to the quadratic eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create the solver context
-  routine_create - routine to create the solver context

   Notes:
   NEPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   NEPRegister("my_solver",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     NEPSetType(qep,"my_solver")
   or at runtime via the option
$     -qep_type my_solver

   Level: advanced

.seealso: NEPRegisterDestroy(), NEPRegisterAll()
@*/
PetscErrorCode NEPRegister(const char *sname,const char *path,const char *name,PetscErrorCode (*function)(NEP))
{
  PetscErrorCode ierr;
  char           fullname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  ierr = PetscFunctionListConcat(path,name,fullname);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(PETSC_COMM_WORLD,&NEPList,sname,fullname,(void (*)(void))function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRegisterDestroy"
/*@
   NEPRegisterDestroy - Frees the list of NEP methods that were
   registered by NEPRegister().

   Not Collective

   Level: advanced

.seealso: NEPRegister(), NEPRegisterAll()
@*/
PetscErrorCode NEPRegisterDestroy(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&NEPList);CHKERRQ(ierr);
  NEPRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset"
/*@
   NEPReset - Resets the NEP context to the setupcalled=0 state and removes any
   allocated objects.

   Collective on NEP

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Level: advanced

.seealso: NEPDestroy()
@*/
PetscErrorCode NEPReset(NEP nep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nep->ops->reset) { ierr = (nep->ops->reset)(nep);CHKERRQ(ierr); }
  if (nep->ip) { ierr = IPReset(nep->ip);CHKERRQ(ierr); }
  if (nep->ds) { ierr = DSReset(nep->ds);CHKERRQ(ierr); }
  ierr = VecDestroy(&nep->t);CHKERRQ(ierr);
  ierr = NEPFreeSolution(nep);CHKERRQ(ierr);
  nep->nfuncs      = 0;
  nep->linits      = 0;
  nep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy"
/*@C
   NEPDestroy - Destroys the NEP context.

   Collective on NEP

   Input Parameter:
.  nep - eigensolver context obtained from NEPCreate()

   Level: beginner

.seealso: NEPCreate(), NEPSetUp(), NEPSolve()
@*/
PetscErrorCode NEPDestroy(NEP *nep)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!*nep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*nep,NEP_CLASSID,1);
  if (--((PetscObject)(*nep))->refct > 0) { *nep = 0; PetscFunctionReturn(0); }
  ierr = NEPReset(*nep);CHKERRQ(ierr);
  if ((*nep)->ops->destroy) { ierr = (*(*nep)->ops->destroy)(*nep);CHKERRQ(ierr); }
  ierr = KSPDestroy(&(*nep)->ksp);CHKERRQ(ierr);
  ierr = IPDestroy(&(*nep)->ip);CHKERRQ(ierr);
  ierr = DSDestroy(&(*nep)->ds);CHKERRQ(ierr);
  ierr = MatDestroy(&(*nep)->function);CHKERRQ(ierr);
  ierr = MatDestroy(&(*nep)->function_pre);CHKERRQ(ierr);
  ierr = MatDestroy(&(*nep)->jacobian);CHKERRQ(ierr);
  if ((*nep)->split) {
    ierr = MatDestroyMatrices((*nep)->nt,&(*nep)->A);CHKERRQ(ierr);
    for (i=0;i<(*nep)->nt;i++) {
      ierr = FNDestroy(&(*nep)->f[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*nep)->f);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&(*nep)->rand);CHKERRQ(ierr);
  /* just in case the initial vectors have not been used */
  ierr = SlepcBasisDestroy_Private(&(*nep)->nini,&(*nep)->IS);CHKERRQ(ierr);
  ierr = NEPMonitorCancel(*nep);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(nep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetIP"
/*@
   NEPSetIP - Associates an inner product object to the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  ip  - the inner product object

   Note:
   Use NEPGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetIP()
@*/
PetscErrorCode NEPSetIP(NEP nep,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ip,IP_CLASSID,2);
  PetscCheckSameComm(nep,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(&nep->ip);CHKERRQ(ierr);
  nep->ip = ip;
  ierr = PetscLogObjectParent(nep,nep->ip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetIP"
/*@C
   NEPGetIP - Obtain the inner product object associated
   to the nonlinear eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: NEPSetIP()
@*/
PetscErrorCode NEPGetIP(NEP nep,IP *ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ip,2);
  if (!nep->ip) {
    ierr = IPCreate(PetscObjectComm((PetscObject)nep),&nep->ip);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(nep,nep->ip);CHKERRQ(ierr);
  }
  *ip = nep->ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetDS"
/*@
   NEPSetDS - Associates a direct solver object to the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  ds  - the direct solver object

   Note:
   Use NEPGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetDS()
@*/
PetscErrorCode NEPSetDS(NEP nep,DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(nep,1,ds,2);
  ierr = PetscObjectReference((PetscObject)ds);CHKERRQ(ierr);
  ierr = DSDestroy(&nep->ds);CHKERRQ(ierr);
  nep->ds = ds;
  ierr = PetscLogObjectParent(nep,nep->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetDS"
/*@C
   NEPGetDS - Obtain the direct solver object associated to the
   nonlinear eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: NEPSetDS()
@*/
PetscErrorCode NEPGetDS(NEP nep,DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!nep->ds) {
    ierr = DSCreate(PetscObjectComm((PetscObject)nep),&nep->ds);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(nep,nep->ds);CHKERRQ(ierr);
  }
  *ds = nep->ds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetKSP"
/*@
   NEPSetKSP - Associates a linear solver object to the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  ksp - the linear solver object

   Note:
   Use NEPGetKSP() to retrieve the linear solver context (for example,
   to free it at the end of the computations).

   Level: developer

.seealso: NEPGetKSP()
@*/
PetscErrorCode NEPSetKSP(NEP nep,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(nep,1,ksp,2);
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&nep->ksp);CHKERRQ(ierr);
  nep->ksp = ksp;
  ierr = PetscLogObjectParent(nep,nep->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetKSP"
/*@C
   NEPGetKSP - Obtain the linear solver (KSP) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ksp - linear solver context

   Level: beginner

.seealso: NEPSetKSP()
@*/
PetscErrorCode NEPGetKSP(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!nep->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)nep),&nep->ksp);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(nep->ksp,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(nep->ksp,"nep_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)nep->ksp,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(nep,nep->ksp);CHKERRQ(ierr);
  }
  *ksp = nep->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetTarget"
/*@
   NEPSetTarget - Sets the value of the target.

   Logically Collective on NEP

   Input Parameters:
+  nep    - eigensolver context
-  target - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with NEPSetWhichEigenpairs().

   Level: beginner

.seealso: NEPGetTarget(), NEPSetWhichEigenpairs()
@*/
PetscErrorCode NEPSetTarget(NEP nep,PetscScalar target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(nep,target,2);
  nep->target = target;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetTarget"
/*@
   NEPGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  nep - eigensolver context

   Output Parameter:
.  target - the value of the target

   Level: beginner

   Note:
   If the target was not set by the user, then zero is returned.

.seealso: NEPSetTarget()
@*/
PetscErrorCode NEPGetTarget(NEP nep,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = nep->target;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetFunction"
/*@C
   NEPSetFunction - Sets the function to compute the nonlinear Function T(lambda)
   as well as the location to store the matrix.

   Logically Collective on NEP and Mat

   Input Parameters:
+  nep - the NEP context
.  A   - Function matrix
.  B   - preconditioner matrix (usually same as the Function)
.  fun - Function evaluation routine (if NULL then NEP retains any
         previously set value)
-  ctx - [optional] user-defined context for private data for the Function
         evaluation routine (may be NULL) (if NULL then NEP retains any
         previously set value)

   Notes:
   The routine fun() takes Mat* as the matrix arguments rather than Mat.
   This allows the Function evaluation routine to replace A and/or B with a
   completely new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner

.seealso: NEPGetFunction(), NEPSetJacobian()
@*/
PetscErrorCode NEPSetFunction(NEP nep,Mat A,Mat B,PetscErrorCode (*fun)(NEP,PetscScalar,PetscScalar,Mat*,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(nep,1,A,2);
  if (B) PetscCheckSameComm(nep,1,B,3);
  if (fun) nep->computefunction = fun;
  if (ctx) nep->functionctx     = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&nep->function);CHKERRQ(ierr);
    nep->function = A;
  }
  if (B) {
    ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
    ierr = MatDestroy(&nep->function_pre);CHKERRQ(ierr);
    nep->function_pre = B;
  }
  nep->split = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetFunction"
/*@C
   NEPGetFunction - Returns the Function matrix and optionally the user
   provided context for evaluating the Function.

   Not Collective, but Mat object will be parallel if NEP object is

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  A   - location to stash Function matrix (or NULL)
.  B   - location to stash preconditioner matrix (or NULL)
.  fun - location to put Function function (or NULL)
-  ctx - location to stash Function context (or NULL)

   Level: advanced

.seealso: NEPSetFunction()
@*/
PetscErrorCode NEPGetFunction(NEP nep,Mat *A,Mat *B,PetscErrorCode (**fun)(NEP,PetscScalar,PetscScalar,Mat*,Mat*,MatStructure*,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A)   *A   = nep->function;
  if (B)   *B   = nep->function_pre;
  if (fun) *fun = nep->computefunction;
  if (ctx) *ctx = nep->functionctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetJacobian"
/*@C
   NEPSetJacobian - Sets the function to compute Jacobian T'(lambda) as well
   as the location to store the matrix.

   Logically Collective on NEP and Mat

   Input Parameters:
+  nep - the NEP context
.  A   - Jacobian matrix
.  jac - Jacobian evaluation routine (if NULL then NEP retains any
         previously set value)
-  ctx - [optional] user-defined context for private data for the Jacobian
         evaluation routine (may be NULL) (if NULL then NEP retains any
         previously set value)

   Notes:
   The routine jac() takes Mat* as the matrix arguments rather than Mat.
   This allows the Jacobian evaluation routine to replace A with a
   completely new matrix structure (not just different matrix elements)
   when appropriate, for instance, if the nonzero structure is changing
   throughout the global iterations.

   Level: beginner

.seealso: NEPSetFunction(), NEPGetJacobian()
@*/
PetscErrorCode NEPSetJacobian(NEP nep,Mat A,PetscErrorCode (*jac)(NEP,PetscScalar,PetscScalar,Mat*,MatStructure*,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (A) PetscCheckSameComm(nep,1,A,2);
  if (jac) nep->computejacobian = jac;
  if (ctx) nep->jacobianctx     = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&nep->jacobian);CHKERRQ(ierr);
    nep->jacobian = A;
  }
  nep->split = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetJacobian"
/*@C
   NEPGetJacobian - Returns the Jacobian matrix and optionally the user
   provided context for evaluating the Jacobian.

   Not Collective, but Mat object will be parallel if NEP object is

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  A   - location to stash Jacobian matrix (or NULL)
.  jac - location to put Jacobian function (or NULL)
-  ctx - location to stash Jacobian context (or NULL)

   Level: advanced

.seealso: NEPSetJacobian()
@*/
PetscErrorCode NEPGetJacobian(NEP nep,Mat *A,PetscErrorCode (**jac)(NEP,PetscScalar,PetscScalar,Mat*,MatStructure*,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A)   *A   = nep->jacobian;
  if (jac) *jac = nep->computejacobian;
  if (ctx) *ctx = nep->jacobianctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetSplitOperator"
/*@
   NEPSetSplitOperator - Sets the operator of the nonlinear eigenvalue problem
   in split form.

   Collective on NEP, Mat and FN

   Input Parameters:
+  nep - the nonlinear eigensolver context
.  n   - number of terms in the split form
.  A   - array of matrices
.  f   - array of functions
-  str - structure flag for matrices

   Notes:
   The nonlinear operator is written as T(lambda) = sum_i A_i*f_i(lambda),
   for i=1,...,n. The derivative T'(lambda) can be obtained using the
   derivatives of f_i.

   The structure flag provides information about A_i's nonzero pattern
   (see MatStructure enum). If all matrices have the same pattern, then
   use SAME_NONZERO_PATTERN. If the patterns are different but contained
   in the pattern of the first one, then use SUBSET_NONZERO_PATTERN.
   Otherwise use DIFFERENT_NONZERO_PATTERN.

   This function must be called before NEPSetUp(). If it is called again
   after NEPSetUp() then the NEP object is reset.

   Level: intermediate

.seealso: NEPGetSplitOperatorTerm(), NEPGetSplitOperatorInfo()
 @*/
PetscErrorCode NEPSetSplitOperator(NEP nep,PetscInt n,Mat A[],FN f[],MatStructure str)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidLogicalCollectiveInt(nep,n,2);
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more terms, you have %D",n);
  PetscValidPointer(A,3);
  PetscCheckSameComm(nep,1,*A,3);
  PetscValidPointer(f,4);
  PetscCheckSameComm(nep,1,*f,4);
  if (nep->setupcalled) { ierr = NEPReset(nep);CHKERRQ(ierr); }
  /* clean previously stored information */
  ierr = MatDestroy(&nep->function);CHKERRQ(ierr);
  ierr = MatDestroy(&nep->function_pre);CHKERRQ(ierr);
  ierr = MatDestroy(&nep->jacobian);CHKERRQ(ierr);
  if (nep->split) {
    ierr = MatDestroyMatrices(nep->nt,&nep->A);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNDestroy(&nep->f[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(nep->f);CHKERRQ(ierr);
  }
  /* allocate space and copy matrices and functions */
  ierr = PetscMalloc(n*sizeof(Mat),&nep->A);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(A[i],MAT_CLASSID,3);
    ierr = PetscObjectReference((PetscObject)A[i]);CHKERRQ(ierr);
    nep->A[i] = A[i];
  }
  ierr = PetscMalloc(n*sizeof(FN),&nep->f);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(f[i],FN_CLASSID,4);
    ierr = PetscObjectReference((PetscObject)f[i]);CHKERRQ(ierr);
    nep->f[i] = f[i];
  }
  nep->nt    = n;
  nep->mstr  = str;
  nep->split = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetSplitOperatorTerm"
/*@
   NEPGetSplitOperatorTerm - Gets the matrices and functions associated with
   the nonlinear operator in split form.

   Not collective, though parallel Mats and FNs are returned if the NEP is parallel

   Input Parameter:
+  nep - the nonlinear eigensolver context
-  k   - the index of the requested term (starting in 0)

   Output Parameters:
+  A - the matrix of the requested term
-  f - the function of the requested term

   Level: intermediate

.seealso: NEPSetSplitOperator(), NEPGetSplitOperatorInfo()
@*/
PetscErrorCode NEPGetSplitOperatorTerm(NEP nep,PetscInt k,Mat *A,FN *f)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (k<0 || k>=nep->nt) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %d",nep->nt-1);
  if (A) *A = nep->A[k];
  if (f) *f = nep->f[k];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetSplitOperatorInfo"
/*@
   NEPGetSplitOperatorInfo - Returns the number of terms of the split form of
   the nonlinear operator, as well as the structure flag for matrices.

   Not collective

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  n   - the number of terms passed in NEPSetSplitOperator()
-  str - the matrix structure flag passed in NEPSetSplitOperator()

   Level: intermediate

.seealso: NEPSetSplitOperator(), NEPGetSplitOperatorTerm()
@*/
PetscErrorCode NEPGetSplitOperatorInfo(NEP nep,PetscInt *n,MatStructure *str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (n) *n = nep->nt;
  if (str) *str = nep->mstr;
  PetscFunctionReturn(0);
}

