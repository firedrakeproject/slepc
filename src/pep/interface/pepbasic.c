/*
     The basic PEP routines, Create, View, etc. are here.

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

#include <slepc-private/pepimpl.h>      /*I "slepcpep.h" I*/

PetscFunctionList PEPList = 0;
PetscBool         PEPRegisterAllCalled = PETSC_FALSE;
PetscClassId      PEP_CLASSID = 0;
PetscLogEvent     PEP_SetUp = 0,PEP_Solve = 0,PEP_Refine = 0;

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
  PetscBool      isascii,islinear,istrivial;
  PetscInt       i;

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
    ierr = PetscViewerASCIIPrintf(viewer,"  polynomial represented in %s basis\n",PEPBasisTypes[pep->basis]);CHKERRQ(ierr);
    switch (pep->scale) {
      case PEP_SCALE_NONE:
        break;
      case PEP_SCALE_SCALAR:
        ierr = PetscViewerASCIIPrintf(viewer,"  scalar balancing enabled, with scaling factor=%g\n",(double)pep->sfactor);CHKERRQ(ierr);
        break;
      case PEP_SCALE_DIAGONAL:
        ierr = PetscViewerASCIIPrintf(viewer,"  diagonal balancing enabled, with its=%D and lambda=%g\n",pep->sits,(double)pep->slambda);CHKERRQ(ierr);
        break;
      case PEP_SCALE_BOTH:
        ierr = PetscViewerASCIIPrintf(viewer,"  scalar & diagonal balancing enabled, with scaling factor=%g, its=%D and lambda=%g\n",(double)pep->sfactor,pep->sits,(double)pep->slambda);CHKERRQ(ierr);
        break;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  extraction type: %s\n",PEPExtractTypes[pep->extract]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  iterative refinement: %s%s\n",PEPRefineTypes[pep->refine],pep->schur?", with a Schur complement approach":"");CHKERRQ(ierr);
    if (pep->refine) {
      ierr = PetscViewerASCIIPrintf(viewer,"  refinement stopping criterion: tol=%g, its=%D\n",(double)pep->rtol,pep->rits);CHKERRQ(ierr);
      if (pep->npart>1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  splitting communicator in %D partitions for refinement\n",pep->npart);CHKERRQ(ierr);
      }
    }
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
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",pep->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",pep->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",pep->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",pep->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)pep->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  convergence test: ");CHKERRQ(ierr);
    switch (pep->conv) {
    case PEP_CONV_ABS:
      ierr = PetscViewerASCIIPrintf(viewer,"absolute\n");CHKERRQ(ierr);break;
    case PEP_CONV_EIG:
      ierr = PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue\n");CHKERRQ(ierr);break;
    case PEP_CONV_NORM:
      ierr = PetscViewerASCIIPrintf(viewer,"relative to the matrix norms\n");CHKERRQ(ierr);
      if (pep->nrma) {
        ierr = PetscViewerASCIIPrintf(viewer,"  computed matrix norms: %g",(double)pep->nrma[0]);CHKERRQ(ierr);
        for (i=1;i<pep->nmat;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,", %g",(double)pep->nrma[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      break;
    case PEP_CONV_USER:
      ierr = PetscViewerASCIIPrintf(viewer,"user-defined\n");CHKERRQ(ierr);break;
    }
    if (pep->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(pep->nini));CHKERRQ(ierr);
    }
  } else {
    if (pep->ops->view) {
      ierr = (*pep->ops->view)(pep,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompare((PetscObject)pep,PEPLINEAR,&islinear);CHKERRQ(ierr);
  if (!islinear) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    if (!pep->V) { ierr = PEPGetBV(pep,&pep->V);CHKERRQ(ierr); }
    ierr = BVView(pep->V,viewer);CHKERRQ(ierr);
    if (!pep->rg) { ierr = PEPGetRG(pep,&pep->rg);CHKERRQ(ierr); }
    ierr = RGIsTrivial(pep->rg,&istrivial);CHKERRQ(ierr);
    if (!istrivial) { ierr = RGView(pep->rg,viewer);CHKERRQ(ierr); }
    if (!pep->ds) { ierr = PEPGetDS(pep,&pep->ds);CHKERRQ(ierr); }
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
  PEPCheckSolved(pep,1);
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
        errok = (errok && error<5.0*pep->tol)? PETSC_TRUE: PETSC_FALSE;
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
              ierr = PetscViewerASCIIPrintf(viewer,"%.5f%+.5fi",(double)re,(double)im);CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer,"%.5f",(double)re);CHKERRQ(ierr);
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
           "           k             ||P(k)x||/||kx||\n"
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
          ierr = PetscViewerASCIIPrintf(viewer," % 9f%+9f i     %12g\n",(double)re,(double)im,(double)error);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"   % 12f           %12g\n",(double)re,(double)error);CHKERRQ(ierr);
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
   The default PEP type is PEPTOAR

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

  pep->max_it          = 0;
  pep->nev             = 1;
  pep->ncv             = 0;
  pep->mpd             = 0;
  pep->nini            = 0;
  pep->target          = 0.0;
  pep->tol             = PETSC_DEFAULT;
  pep->conv            = PEP_CONV_NORM;
  pep->which           = (PEPWhich)0;
  pep->basis           = PEP_BASIS_MONOMIAL;
  pep->problem_type    = (PEPProblemType)0;
  pep->scale           = PEP_SCALE_NONE;
  pep->sfactor         = 1.0;
  pep->sits            = 5;
  pep->slambda         = 1.0;
  pep->refine          = PEP_REFINE_NONE;
  pep->npart           = 1;
  pep->rtol            = PETSC_DEFAULT;
  pep->rits            = PETSC_DEFAULT;
  pep->schur           = PETSC_FALSE;
  pep->extract         = PEP_EXTRACT_NORM;
  pep->trackall        = PETSC_FALSE;

  pep->converged       = PEPConvergedNormRelative;
  pep->convergeddestroy= NULL;
  pep->convergedctx    = NULL;
  pep->numbermonitors  = 0;

  pep->st              = NULL;
  pep->ds              = NULL;
  pep->V               = NULL;
  pep->rg              = NULL;
  pep->rand            = NULL;
  pep->A               = NULL;
  pep->nmat            = 0;
  pep->Dl              = NULL;
  pep->Dr              = NULL;
  pep->IS              = NULL;
  pep->eigr            = NULL;
  pep->eigi            = NULL;
  pep->errest          = NULL;
  pep->perm            = NULL;
  pep->pbc             = NULL;
  pep->solvematcoeffs  = NULL;
  pep->nwork           = 0;
  pep->work            = NULL;
  pep->data            = NULL;

  pep->state           = PEP_STATE_INITIAL;
  pep->nconv           = 0;
  pep->its             = 0;
  pep->n               = 0;
  pep->nloc            = 0;
  pep->nrma            = NULL;
  pep->sfactor_set     = PETSC_FALSE;
  pep->reason          = PEP_CONVERGED_ITERATING;

  ierr = PetscNewLog(pep,&pep->sc);CHKERRQ(ierr);
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
   is PEPTOAR.

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

  pep->state = PEP_STATE_INITIAL;
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
   PEPReset - Resets the PEP context to the initial state and removes any
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
  PetscInt       ncols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  if (pep->ops->reset) { ierr = (pep->ops->reset)(pep);CHKERRQ(ierr); }
  if (pep->st) { ierr = STReset(pep->st);CHKERRQ(ierr); }
  if (pep->ds) { ierr = DSReset(pep->ds);CHKERRQ(ierr); }
  if (pep->nmat) {
    ierr = MatDestroyMatrices(pep->nmat,&pep->A);CHKERRQ(ierr);
    ierr = PetscFree3(pep->pbc,pep->solvematcoeffs,pep->nrma);CHKERRQ(ierr);
    pep->nmat = 0;
  }
  ierr = VecDestroy(&pep->Dl);CHKERRQ(ierr);
  ierr = VecDestroy(&pep->Dr);CHKERRQ(ierr);
  ierr = BVGetSizes(pep->V,NULL,NULL,&ncols);CHKERRQ(ierr);
  if (ncols) {
    ierr = PetscFree4(pep->eigr,pep->eigi,pep->errest,pep->perm);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&pep->V);CHKERRQ(ierr);
  ierr = VecDestroyVecs(pep->nwork,&pep->work);CHKERRQ(ierr);
  pep->nwork = 0;
  pep->state = PEP_STATE_INITIAL;
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
  if ((*pep)->ops->destroy) { ierr = (*(*pep)->ops->destroy)(*pep);CHKERRQ(ierr); }
  ierr = STDestroy(&(*pep)->st);CHKERRQ(ierr);
  ierr = RGDestroy(&(*pep)->rg);CHKERRQ(ierr);
  ierr = DSDestroy(&(*pep)->ds);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*pep)->rand);CHKERRQ(ierr);
  ierr = PetscFree((*pep)->sc);CHKERRQ(ierr);
  /* just in case the initial vectors have not been used */
  ierr = SlepcBasisDestroy_Private(&(*pep)->nini,&(*pep)->IS);CHKERRQ(ierr);
  if ((*pep)->convergeddestroy) {
    ierr = (*(*pep)->convergeddestroy)((*pep)->convergedctx);CHKERRQ(ierr);
  }
  ierr = PEPMonitorCancel(*pep);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(pep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetBV"
/*@
   PEPSetBV - Associates a basis vectors object to the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  bv  - the basis vectors object

   Note:
   Use PEPGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetBV()
@*/
PetscErrorCode PEPSetBV(PEP pep,BV bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(pep,1,bv,2);
  ierr = PetscObjectReference((PetscObject)bv);CHKERRQ(ierr);
  ierr = BVDestroy(&pep->V);CHKERRQ(ierr);
  pep->V = bv;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetBV"
/*@C
   PEPGetBV - Obtain the basis vectors object associated to the polynomial
   eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: PEPSetBV()
@*/
PetscErrorCode PEPGetBV(PEP pep,BV *bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!pep->V) {
    ierr = BVCreate(PetscObjectComm((PetscObject)pep),&pep->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->V);CHKERRQ(ierr);
  }
  *bv = pep->V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPSetRG"
/*@
   PEPSetRG - Associates a region object to the polynomial eigensolver.

   Collective on PEP

   Input Parameters:
+  pep - eigensolver context obtained from PEPCreate()
-  rg  - the region object

   Note:
   Use PEPGetRG() to retrieve the region context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: PEPGetRG()
@*/
PetscErrorCode PEPSetRG(PEP pep,RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidHeaderSpecific(rg,RG_CLASSID,2);
  PetscCheckSameComm(pep,1,rg,2);
  ierr = PetscObjectReference((PetscObject)rg);CHKERRQ(ierr);
  ierr = RGDestroy(&pep->rg);CHKERRQ(ierr);
  pep->rg = rg;
  ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PEPGetRG"
/*@C
   PEPGetRG - Obtain the region object associated to the
   polynomial eigensolver object.

   Not Collective

   Input Parameters:
.  pep - eigensolver context obtained from PEPCreate()

   Output Parameter:
.  rg - region context

   Level: advanced

.seealso: PEPSetRG()
@*/
PetscErrorCode PEPGetRG(PEP pep,RG *rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pep,PEP_CLASSID,1);
  PetscValidPointer(rg,2);
  if (!pep->rg) {
    ierr = RGCreate(PetscObjectComm((PetscObject)pep),&pep->rg);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)pep,(PetscObject)pep->rg);CHKERRQ(ierr);
  }
  *rg = pep->rg;
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

   Options Database Key:
.  -pep_target <scalar> - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with PEPSetWhichEigenpairs().

   In the case of complex scalars, a complex value can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -pep_target 1.0+2.0i

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

   Note:
   If the target was not set by the user, then zero is returned.

   Level: beginner

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

