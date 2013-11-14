/*
     The basic PEP routines, Create, View, etc. are here.

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

#include <slepc-private/pepimpl.h>      /*I "slepcpep.h" I*/

PetscFunctionList PEPList = 0;
PetscBool         PEPRegisterAllCalled = PETSC_FALSE;
PetscClassId      PEP_CLASSID = 0;
PetscLogEvent     PEP_SetUp = 0,PEP_Solve = 0,PEP_Dense = 0;
static PetscBool  PEPPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PEPFinalizePackage"
/*@C
   PEPFinalizePackage - This function destroys everything in the Slepc interface
   to the PEP package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode PEPFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PEPList);CHKERRQ(ierr);
  PEPPackageInitialized = PETSC_FALSE;
  PEPRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPInitializePackage"
/*@C
   PEPInitializePackage - This function initializes everything in the PEP package. It is called
   from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to PEPCreate()
   when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode PEPInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PEPPackageInitialized) PetscFunctionReturn(0);
  PEPPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Polynomial Eigenvalue Problem solver",&PEP_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PEPRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("PEPSetUp",PEP_CLASSID,&PEP_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PEPSolve",PEP_CLASSID,&PEP_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PEPDense",PEP_CLASSID,&PEP_Dense);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"pep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(PEP_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"pep",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(PEP_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(PEPFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPView"
/*@C
   PEPView - Prints the PEP data structure.

   Collective on PEP

   Input Parameters:
+  pep - the polynomial eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -pep_view -  Calls PEPView() at end of PEPSolve()

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
PetscErrorCode PEPView(PEP pep,PetscViewer viewer)
{
  PetscErrorCode ierr;
  const char     *type;
  char           str[50];
  PetscBool      isascii,islinear;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pep,1,viewer,2);

#if defined(PETSC_USE_COMPLEX)
#define HERM "hermitian"
#else
#define HERM "symmetric"
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)pep,viewer);CHKERRQ(ierr);
    if (pep->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*pep->ops->view)(pep,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (pep->problem_type) {
      switch (pep->problem_type) {
        case PEP_GENERAL:    type = "general polynomial eigenvalue problem"; break;
        case PEP_HERMITIAN:  type = HERM " polynomial eigenvalue problem"; break;
        case PEP_GYROSCOPIC: type = "gyroscopic polynomial eigenvalue problem"; break;
        default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of pep->problem_type");
      }
    } else type = "not yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    ierr = SlepcSNPrintfScalar(str,50,pep->target,PETSC_FALSE);CHKERRQ(ierr);
    if (!pep->which) {
      ierr = PetscViewerASCIIPrintf(viewer,"not yet set\n");CHKERRQ(ierr);
    } else switch (pep->which) {
      case PEP_TARGET_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str);CHKERRQ(ierr);
        break;
      case PEP_TARGET_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str);CHKERRQ(ierr);
        break;
      case PEP_TARGET_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str);CHKERRQ(ierr);
        break;
      case PEP_LARGEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case PEP_SMALLEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case PEP_LARGEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"largest real parts\n");CHKERRQ(ierr);
        break;
      case PEP_SMALLEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest real parts\n");CHKERRQ(ierr);
        break;
      case PEP_LARGEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n");CHKERRQ(ierr);
        break;
      case PEP_SMALLEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n");CHKERRQ(ierr);
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)pep),1,"Wrong value of pep->which");
    }
    if (pep->leftvecs) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing left eigenvectors also\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",pep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",pep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",pep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",pep->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %G\n",pep->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  scaling factor: %G\n",pep->sfactor);CHKERRQ(ierr);
    if (pep->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(pep->nini));CHKERRQ(ierr);
    }
    if (pep->ninil) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial left space: %D\n",PetscAbs(pep->ninil));CHKERRQ(ierr);
    }
  } else {
    if (pep->ops->view) {
      ierr = (*pep->ops->view)(pep,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    if (!pep->ip) { ierr = PEPGetIP(pep,&pep->ip);CHKERRQ(ierr); }
    ierr = IPView(pep->ip,viewer);CHKERRQ(ierr);
    if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = DSView(pep->ds,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    if (!pep->st) { ierr = PEPGetST(pep,&pep->st);CHKERRQ(ierr); }
    ierr = STView(pep->st,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPPrintSolution"
/*@
   PEPPrintSolution - Prints the computed eigenvalues.

   Collective on PEP

   Input Parameters:
+  pep - the eigensolver context
-  viewer - optional visualization context

   Options Database Key:
.  -pep_terse - print only minimal information

   Note:
   By default, this function prints a table with eigenvalues and associated
   relative errors. With -pep_terse only the eigenvalues are printed.

   Level: intermediate

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PEPPrintSolution(PEP pep,PetscViewer viewer)
{
  PetscBool      terse,errok,isascii;
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pep));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(pep,1,viewer,2);
  if (!pep->eigr || !pep->eigi || !pep->V) SETERRQ(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_WRONGSTATE,"PEPSolve must be called first");
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscOptionsHasName(NULL,"-pep_terse",&terse);CHKERRQ(ierr);
  if (terse) {
    if (pep->nconv<pep->nev) {
      ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D eigenvalues converged\n\n",pep->nev);CHKERRQ(ierr);
    } else {
      errok = PETSC_TRUE;
      for (i=0;i<pep->nev;i++) {
        ierr = PEPComputeRelativeError(pep,i,&error);CHKERRQ(ierr);
        errok = (errok && error<pep->tol)? PETSC_TRUE: PETSC_FALSE;
      }
      if (errok) {
        ierr = PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:");CHKERRQ(ierr);
        for (i=0;i<=(pep->nev-1)/8;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
          for (j=0;j<PetscMin(8,pep->nev-8*i);j++) {
            ierr = PEPGetEigenpair(pep,8*i+j,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
            re = PetscRealPart(kr);
            im = PetscImaginaryPart(kr);
#else
            re = kr;
            im = ki;
#endif
            if (PetscAbs(re)/PetscAbs(im)<PETSC_SMALL) re = 0.0;
            if (PetscAbs(im)/PetscAbs(re)<PETSC_SMALL) im = 0.0;
            if (im!=0.0) {
              ierr = PetscViewerASCIIPrintf(viewer,"%.5F%+.5Fi",re,im);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer,"%.5F",re);CHKERRQ(ierr);
            }
            if (8*i+j+1<pep->nev) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",pep->nev);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscViewerASCIIPrintf(viewer," Number of converged approximate eigenpairs: %D\n\n",pep->nconv);CHKERRQ(ierr);
    if (pep->nconv>0) {
      ierr = PetscViewerASCIIPrintf(viewer,
           "           k          ||(k^2M+Ck+K)x||/||kx||\n"
           "   ----------------- -------------------------\n");CHKERRQ(ierr);
      for (i=0;i<pep->nconv;i++) {
        ierr = PEPGetEigenpair(pep,i,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
        ierr = PEPComputeRelativeError(pep,i,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        re = PetscRealPart(kr);
        im = PetscImaginaryPart(kr);
#else
        re = kr;
        im = ki;
#endif
        if (im!=0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," % 9F%+9F i     %12G\n",re,im,error);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"   % 12F           %12G\n",re,error);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPCreate"
/*@C
   PEPCreate - Creates the default PEP context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  pep - location to put the PEP context

   Note:
   The default PEP type is PEPLINEAR

   Level: beginner

.seealso: PEPSetUp(), PEPSolve(), PEPDestroy(), PEP
@*/
PetscErrorCode PEPCreate(MPI_Comm comm,PEP *outpep)
{
  PetscErrorCode ierr;
  PEP            pep;

  PetscFunctionBegin;
  PetscValidPointer(outpep,2);
  *outpep = 0;
  ierr = PEPInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(pep,_p_PEP,struct _PEPOps,PEP_CLASSID,"PEP","Polynomial Eigenvalue Problem","PEP",comm,PEPDestroy,PEPView);CHKERRQ(ierr);

  pep->A               = 0;
  pep->nmat            = 0;
  pep->max_it          = 0;
  pep->nev             = 1;
  pep->ncv             = 0;
  pep->mpd             = 0;
  pep->nini            = 0;
  pep->ninil           = 0;
  pep->allocated_ncv   = 0;
  pep->ip              = 0;
  pep->ds              = 0;
  pep->tol             = PETSC_DEFAULT;
  pep->sfactor         = 0.0;
  pep->sfactor_set     = PETSC_FALSE;
  pep->converged       = PEPConvergedDefault;
  pep->convergedctx    = NULL;
  pep->which           = (PEPWhich)0;
  pep->comparison      = NULL;
  pep->comparisonctx   = NULL;
  pep->leftvecs        = PETSC_FALSE;
  pep->problem_type    = (PEPProblemType)0;
  pep->V               = NULL;
  pep->W               = NULL;
  pep->IS              = NULL;
  pep->ISL             = NULL;
  pep->eigr            = NULL;
  pep->eigi            = NULL;
  pep->errest          = NULL;
  pep->data            = NULL;
  pep->t               = NULL;
  pep->nconv           = 0;
  pep->its             = 0;
  pep->perm            = NULL;
  pep->matvecs         = 0;
  pep->linits          = 0;
  pep->nwork           = 0;
  pep->work            = NULL;
  pep->setupcalled     = 0;
  pep->reason          = PEP_CONVERGED_ITERATING;
  pep->numbermonitors  = 0;
  pep->trackall        = PETSC_FALSE;
  pep->rand            = 0;

  ierr = PetscRandomCreate(comm,&pep->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(pep->rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->rand);CHKERRQ(ierr);
  *outpep = pep;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetType"
/*@C
   PEPSetType - Selects the particular solver to be used in the PEP object.

   Logically Collective on PEP

   Input Parameters:
+  pep      - the polynomial eigensolver context
-  type     - a known method

   Options Database Key:
.  -pep_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepcpep.h" for available methods. The default
   is PEPLINEAR.

   Normally, it is best to use the PEPSetFromOptions() command and
   then set the PEP type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The PEPSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: PEPType
@*/
PetscErrorCode PEPSetType(PEP pep,PEPType type)
{
  PetscErrorCode ierr,(*r)(PEP);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)pep,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PEPList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)pep),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PEP type given: %s",type);

  if (pep->ops->destroy) { ierr = (*pep->ops->destroy)(pep);CHKERRQ(ierr); }
  ierr = PetscMemzero(pep->ops,sizeof(struct _PEPOps));CHKERRQ(ierr);

  pep->setupcalled = 0;
  ierr = PetscObjectChangeTypeName((PetscObject)pep,type);CHKERRQ(ierr);
  ierr = (*r)(pep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetType"
/*@C
   PEPGetType - Gets the PEP type as a string from the PEP object.

   Not Collective

   Input Parameter:
.  pep - the eigensolver context

   Output Parameter:
.  name - name of PEP method

   Level: intermediate

.seealso: PEPSetType()
@*/
PetscErrorCode PEPGetType(PEP pep,PEPType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)pep)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPRegister"
/*@C
   PEPRegister - Adds a method to the polynomial eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   PEPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   PEPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PEPSetType(pep,"my_solver")
   or at runtime via the option
$     -pep_type my_solver

   Level: advanced

.seealso: PEPRegisterAll()
@*/
PetscErrorCode PEPRegister(const char *name,PetscErrorCode (*function)(PEP))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&PEPList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPReset"
/*@
   PEPReset - Resets the PEP context to the setupcalled=0 state and removes any
   allocated objects.

   Collective on PEP

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Level: advanced

.seealso: PEPDestroy()
@*/
PetscErrorCode PEPReset(PEP pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (pep->ops->reset) { ierr = (pep->ops->reset)(pep);CHKERRQ(ierr); }
  if (pep->ip) { ierr = IPReset(pep->ip);CHKERRQ(ierr); }
  if (pep->ds) { ierr = DSReset(pep->ds);CHKERRQ(ierr); }
  ierr = VecDestroy(&pep->t);CHKERRQ(ierr);
  ierr = PEPFreeSolution(pep);CHKERRQ(ierr);
  pep->matvecs     = 0;
  pep->linits      = 0;
  pep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPDestroy"
/*@C
   PEPDestroy - Destroys the PEP context.

   Collective on PEP

   Input Parameter:
.  pep - eigensolver context obtained from PEPCreate()

   Level: beginner

.seealso: PEPCreate(), PEPSetUp(), PEPSolve()
@*/
PetscErrorCode PEPDestroy(PEP *pep)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*pep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*pep,PEP_CLASSID,1);
  if (--((PetscObject)(*pep))->refct > 0) { *pep = 0; PetscFunctionReturn(0); }
  ierr = PEPReset(*pep);CHKERRQ(ierr);
  ierr = MatDestroyMatrices((*pep)->nmat,&(*pep)->A);CHKERRQ(ierr);
  if ((*pep)->ops->destroy) { ierr = (*(*pep)->ops->destroy)(*pep);CHKERRQ(ierr); }
  ierr = STDestroy(&(*pep)->st);CHKERRQ(ierr);
  ierr = IPDestroy(&(*pep)->ip);CHKERRQ(ierr);
  ierr = DSDestroy(&(*pep)->ds);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*pep)->rand);CHKERRQ(ierr);
  /* just in case the initial vectors have not been used */
  ierr = SlepcBasisDestroy_Private(&(*pep)->nini,&(*pep)->IS);CHKERRQ(ierr);
  ierr = SlepcBasisDestroy_Private(&(*pep)->ninil,&(*pep)->ISL);CHKERRQ(ierr);
  ierr = PEPMonitorCancel(*pep);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(pep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetIP"
/*@
   PEPSetIP - Associates an inner product object to the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  ip  - the inner product object

   Note:
   Use PEPGetIP() to retrieve the inner product context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetIP()
@*/
PetscErrorCode PEPSetIP(PEP pep,IP ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(ip,IP_CLASSID,2);
  PetscCheckSameComm(pep,1,ip,2);
  ierr = PetscObjectReference((PetscObject)ip);CHKERRQ(ierr);
  ierr = IPDestroy(&pep->ip);CHKERRQ(ierr);
  pep->ip = ip;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ip);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetIP"
/*@C
   PEPGetIP - Obtain the inner product object associated
   to the polynomial eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  ip - inner product context

   Level: advanced

.seealso: PEPSetIP()
@*/
PetscErrorCode PEPGetIP(PEP pep,IP *ip)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(ip,2);
  if (!pep->ip) {
    ierr = IPCreate(PetscObjectComm((PetscObject)pep),&pep->ip);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ip);CHKERRQ(ierr);
  }
  *ip = pep->ip;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetDS"
/*@
   PEPSetDS - Associates a direct solver object to the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  ds  - the direct solver object

   Note:
   Use PEPGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetDS()
@*/
PetscErrorCode PEPSetDS(PEP pep,DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(pep,1,ds,2);
  ierr = PetscObjectReference((PetscObject)ds);CHKERRQ(ierr);
  ierr = DSDestroy(&pep->ds);CHKERRQ(ierr);
  pep->ds = ds;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetDS"
/*@C
   PEPGetDS - Obtain the direct solver object associated to the
   polynomial eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: PEPSetDS()
@*/
PetscErrorCode PEPGetDS(PEP pep,DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!pep->ds) {
    ierr = DSCreate(PetscObjectComm((PetscObject)pep),&pep->ds);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->ds);CHKERRQ(ierr);
  }
  *ds = pep->ds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetST"
/*@
   PEPSetST - Associates a spectral transformation object to the eigensolver.

   Collective on PEP

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  st   - the spectral transformation object

   Note:
   Use PEPGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: developer

.seealso: PEPGetST()
@*/
PetscErrorCode PEPSetST(PEP pep,ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(st,ST_CLASSID,2);
  PetscCheckSameComm(pep,1,st,2);
  ierr = PetscObjectReference((PetscObject)st);CHKERRQ(ierr);
  ierr = STDestroy(&pep->st);CHKERRQ(ierr);
  pep->st = st;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetST"
/*@C
   PEPGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: PEPSetST()
@*/
PetscErrorCode PEPGetST(PEP pep,ST *st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(st,2);
  if (!pep->st) {
    ierr = STCreate(PetscObjectComm((PetscObject)pep),&pep->st);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->st);CHKERRQ(ierr);
  }
  *st = pep->st;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetTarget"
/*@
   PEPSetTarget - Sets the value of the target.

   Logically Collective on PEP

   Input Parameters:
+  pep    - eigensolver context
-  target - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with PEPSetWhichEigenpairs().

   Level: beginner

.seealso: PEPGetTarget(), PEPSetWhichEigenpairs()
@*/
PetscErrorCode PEPSetTarget(PEP pep,PetscScalar target)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pep,target,2);
  pep->target = target;
  if (!pep->st) { ierr = PEPGetST(pep,&pep->st);CHKERRQ(ierr); }
  ierr = STSetDefaultShift(pep->st,target);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetTarget"
/*@
   PEPGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  pep - eigensolver context

   Output Parameter:
.  target - the value of the target

   Level: beginner

   Note:
   If the target was not set by the user, then zero is returned.

.seealso: PEPSetTarget()
@*/
PetscErrorCode PEPGetTarget(PEP pep,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = pep->target;
  PetscFunctionReturn(0);
}

