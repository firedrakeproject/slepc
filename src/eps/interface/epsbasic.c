/*
     The basic EPS routines, Create, View, etc. are here.

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

#include <slepc-private/epsimpl.h>      /*I "slepceps.h" I*/

PetscFunctionList EPSList = 0;
PetscBool         EPSRegisterAllCalled = PETSC_FALSE;
PetscClassId      EPS_CLASSID = 0;
PetscLogEvent     EPS_SetUp = 0,EPS_Solve = 0;

#undef __FUNCT__
#define __FUNCT__ "EPSView"
/*@C
   EPSView - Prints the EPS data structure.

   Collective on EPS

   Input Parameters:
+  eps - the eigenproblem solver context
-  viewer - optional visualization context

   Options Database Key:
.  -eps_view -  Calls EPSView() at end of EPSSolve()

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

.seealso: STView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode EPSView(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  const char     *type,*extr,*bal;
  char           str[50];
  PetscBool      isascii,ispower,isexternal,istrivial;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)eps));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(eps,1,viewer,2);

#if defined(PETSC_USE_COMPLEX)
#define HERM "hermitian"
#else
#define HERM "symmetric"
#endif
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)eps,viewer);CHKERRQ(ierr);
    if (eps->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*eps->ops->view)(eps,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (eps->problem_type) {
      switch (eps->problem_type) {
        case EPS_HEP:   type = HERM " eigenvalue problem"; break;
        case EPS_GHEP:  type = "generalized " HERM " eigenvalue problem"; break;
        case EPS_NHEP:  type = "non-" HERM " eigenvalue problem"; break;
        case EPS_GNHEP: type = "generalized non-" HERM " eigenvalue problem"; break;
        case EPS_PGNHEP: type = "generalized non-" HERM " eigenvalue problem with " HERM " positive definite B"; break;
        case EPS_GHIEP: type = "generalized " HERM "-indefinite eigenvalue problem"; break;
        default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->problem_type");
      }
    } else type = "not yet set";
    ierr = PetscViewerASCIIPrintf(viewer,"  problem type: %s\n",type);CHKERRQ(ierr);
    if (eps->extraction) {
      switch (eps->extraction) {
        case EPS_RITZ:             extr = "Rayleigh-Ritz"; break;
        case EPS_HARMONIC:         extr = "harmonic Ritz"; break;
        case EPS_HARMONIC_RELATIVE:extr = "relative harmonic Ritz"; break;
        case EPS_HARMONIC_RIGHT:   extr = "right harmonic Ritz"; break;
        case EPS_HARMONIC_LARGEST: extr = "largest harmonic Ritz"; break;
        case EPS_REFINED:          extr = "refined Ritz"; break;
        case EPS_REFINED_HARMONIC: extr = "refined harmonic Ritz"; break;
        default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->extraction");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  extraction type: %s\n",extr);CHKERRQ(ierr);
    }
    if (!eps->ishermitian && eps->balance!=EPS_BALANCE_NONE) {
      switch (eps->balance) {
        case EPS_BALANCE_ONESIDE:   bal = "one-sided Krylov"; break;
        case EPS_BALANCE_TWOSIDE:   bal = "two-sided Krylov"; break;
        case EPS_BALANCE_USER:      bal = "user-defined matrix"; break;
        default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->balance");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  balancing enabled: %s",bal);CHKERRQ(ierr);
      if (eps->balance==EPS_BALANCE_ONESIDE || eps->balance==EPS_BALANCE_TWOSIDE) {
        ierr = PetscViewerASCIIPrintf(viewer,", with its=%D",eps->balance_its);CHKERRQ(ierr);
      }
      if (eps->balance==EPS_BALANCE_TWOSIDE && eps->balance_cutoff!=0.0) {
        ierr = PetscViewerASCIIPrintf(viewer," and cutoff=%g",(double)eps->balance_cutoff);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  selected portion of the spectrum: ");CHKERRQ(ierr);
    ierr = SlepcSNPrintfScalar(str,50,eps->target,PETSC_FALSE);CHKERRQ(ierr);
    if (!eps->which) {
      ierr = PetscViewerASCIIPrintf(viewer,"not yet set\n");CHKERRQ(ierr);
    } else switch (eps->which) {
      case EPS_WHICH_USER:
        ierr = PetscViewerASCIIPrintf(viewer,"user defined\n");CHKERRQ(ierr);
        break;
      case EPS_TARGET_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (in magnitude)\n",str);CHKERRQ(ierr);
        break;
      case EPS_TARGET_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the real axis)\n",str);CHKERRQ(ierr);
        break;
      case EPS_TARGET_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"closest to target: %s (along the imaginary axis)\n",str);CHKERRQ(ierr);
        break;
      case EPS_LARGEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"largest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case EPS_SMALLEST_MAGNITUDE:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest eigenvalues in magnitude\n");CHKERRQ(ierr);
        break;
      case EPS_LARGEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"largest real parts\n");CHKERRQ(ierr);
        break;
      case EPS_SMALLEST_REAL:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest real parts\n");CHKERRQ(ierr);
        break;
      case EPS_LARGEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"largest imaginary parts\n");CHKERRQ(ierr);
        break;
      case EPS_SMALLEST_IMAGINARY:
        ierr = PetscViewerASCIIPrintf(viewer,"smallest imaginary parts\n");CHKERRQ(ierr);
        break;
      case EPS_ALL:
        ierr = PetscViewerASCIIPrintf(viewer,"all eigenvalues in interval [%g,%g]\n",(double)eps->inta,(double)eps->intb);CHKERRQ(ierr);
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)eps),1,"Wrong value of eps->which");
    }
    if (eps->trueres) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing true residuals explicitly\n");CHKERRQ(ierr);
    }
    if (eps->trackall) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing all residuals (for tracking convergence)\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  number of eigenvalues (nev): %D\n",eps->nev);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of column vectors (ncv): %D\n",eps->ncv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum dimension of projected problem (mpd): %D\n",eps->mpd);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  maximum number of iterations: %D\n",eps->max_it);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance: %g\n",(double)eps->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  convergence test: ");CHKERRQ(ierr);
    switch (eps->conv) {
    case EPS_CONV_ABS:
      ierr = PetscViewerASCIIPrintf(viewer,"absolute\n");CHKERRQ(ierr);break;
    case EPS_CONV_EIG:
      ierr = PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue\n");CHKERRQ(ierr);break;
    case EPS_CONV_NORM:
      ierr = PetscViewerASCIIPrintf(viewer,"relative to the eigenvalue and matrix norms\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  computed matrix norms: norm(A)=%g",(double)eps->nrma);CHKERRQ(ierr);
      if (eps->isgeneralized) {
        ierr = PetscViewerASCIIPrintf(viewer,", norm(B)=%g",(double)eps->nrmb);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      break;
    case EPS_CONV_USER:
      ierr = PetscViewerASCIIPrintf(viewer,"user-defined\n");CHKERRQ(ierr);break;
    }
    if (eps->nini) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided initial space: %D\n",PetscAbs(eps->nini));CHKERRQ(ierr);
    }
    if (eps->nds) {
      ierr = PetscViewerASCIIPrintf(viewer,"  dimension of user-provided deflation space: %D\n",PetscAbs(eps->nds));CHKERRQ(ierr);
    }
  } else {
    if (eps->ops->view) {
      ierr = (*eps->ops->view)(eps,viewer);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectTypeCompareAny((PetscObject)eps,&isexternal,EPSARPACK,EPSBLZPACK,EPSTRLAN,EPSBLOPEX,EPSPRIMME,"");CHKERRQ(ierr);
  if (!isexternal) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
    ierr = BVView(eps->V,viewer);CHKERRQ(ierr);
    if (eps->rg) {
      ierr = RGIsTrivial(eps->rg,&istrivial);CHKERRQ(ierr);
      if (!istrivial) { ierr = RGView(eps->rg,viewer);CHKERRQ(ierr); }
    }
    ierr = PetscObjectTypeCompare((PetscObject)eps,EPSPOWER,&ispower);CHKERRQ(ierr);
    if (!ispower) {
      if (!eps->ds) { ierr = EPSGetDS(eps,&eps->ds);CHKERRQ(ierr); }
      ierr = DSView(eps->ds,viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  if (!eps->st) { ierr = EPSGetST(eps,&eps->st);CHKERRQ(ierr); }
  ierr = STView(eps->st,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSPrintSolution"
/*@
   EPSPrintSolution - Prints the computed eigenvalues.

   Collective on EPS

   Input Parameters:
+  eps - the eigensolver context
-  viewer - optional visualization context

   Options Database Key:
.  -eps_terse - print only minimal information

   Note:
   By default, this function prints a table with eigenvalues and associated
   relative errors. With -eps_terse only the eigenvalues are printed.

   Level: intermediate

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode EPSPrintSolution(EPS eps,PetscViewer viewer)
{
  PetscBool      terse,errok,isascii;
  PetscReal      error,re,im;
  PetscScalar    kr,ki;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)eps));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(eps,1,viewer,2);
  EPSCheckSolved(eps,1);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) PetscFunctionReturn(0);

  ierr = PetscOptionsHasName(NULL,"-eps_terse",&terse);CHKERRQ(ierr);
  if (terse) {
    if (eps->nconv<eps->nev) {
      ierr = PetscViewerASCIIPrintf(viewer," Problem: less than %D eigenvalues converged\n\n",eps->nev);CHKERRQ(ierr);
    } else {
      errok = PETSC_TRUE;
      for (i=0;i<eps->nev;i++) {
        ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);
        errok = (errok && error<5.0*eps->tol)? PETSC_TRUE: PETSC_FALSE;
      }
      if (errok) {
        ierr = PetscViewerASCIIPrintf(viewer," All requested eigenvalues computed up to the required tolerance:");CHKERRQ(ierr);
        for (i=0;i<=(eps->nev-1)/8;i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"\n     ");CHKERRQ(ierr);
          for (j=0;j<PetscMin(8,eps->nev-8*i);j++) {
            ierr = EPSGetEigenpair(eps,8*i+j,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
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
            if (8*i+j+1<eps->nev) { ierr = PetscViewerASCIIPrintf(viewer,", ");CHKERRQ(ierr); }
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer," Problem: some of the first %D relative errors are higher than the tolerance\n\n",eps->nev);CHKERRQ(ierr);
      }
    }
  } else {
    ierr = PetscViewerASCIIPrintf(viewer," Number of converged approximate eigenpairs: %D\n\n",eps->nconv);CHKERRQ(ierr);
    if (eps->nconv>0) {
      ierr = PetscViewerASCIIPrintf(viewer,
           "           k          ||Ax-k%sx||/||kx||\n"
           "   ----------------- ------------------\n",eps->isgeneralized?"B":"");CHKERRQ(ierr);
      for (i=0;i<eps->nconv;i++) {
        ierr = EPSGetEigenpair(eps,i,&kr,&ki,NULL,NULL);CHKERRQ(ierr);
        ierr = EPSComputeRelativeError(eps,i,&error);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        re = PetscRealPart(kr);
        im = PetscImaginaryPart(kr);
#else
        re = kr;
        im = ki;
#endif
        if (im!=0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," % 9f%+9f i %12g\n",(double)re,(double)im,(double)error);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"   % 12f       %12g\n",(double)re,(double)error);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSCreate"
/*@C
   EPSCreate - Creates the default EPS context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  eps - location to put the EPS context

   Note:
   The default EPS type is EPSKRYLOVSCHUR

   Level: beginner

.seealso: EPSSetUp(), EPSSolve(), EPSDestroy(), EPS
@*/
PetscErrorCode EPSCreate(MPI_Comm comm,EPS *outeps)
{
  PetscErrorCode ierr;
  EPS            eps;

  PetscFunctionBegin;
  PetscValidPointer(outeps,2);
  *outeps = 0;
  ierr = EPSInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(eps,_p_EPS,struct _EPSOps,EPS_CLASSID,"EPS","Eigenvalue Problem Solver","EPS",comm,EPSDestroy,EPSView);CHKERRQ(ierr);

  eps->max_it          = 0;
  eps->nev             = 1;
  eps->ncv             = 0;
  eps->mpd             = 0;
  eps->nini            = 0;
  eps->nds             = 0;
  eps->target          = 0.0;
  eps->tol             = PETSC_DEFAULT;
  eps->conv            = EPS_CONV_EIG;
  eps->which           = (EPSWhich)0;
  eps->inta            = 0.0;
  eps->intb            = 0.0;
  eps->problem_type    = (EPSProblemType)0;
  eps->extraction      = EPS_RITZ;
  eps->balance         = EPS_BALANCE_NONE;
  eps->balance_its     = 5;
  eps->balance_cutoff  = 1e-8;
  eps->trueres         = PETSC_FALSE;
  eps->trackall        = PETSC_FALSE;

  eps->converged       = EPSConvergedEigRelative;
  eps->convergeddestroy= NULL;
  eps->arbitrary       = NULL;
  eps->convergedctx    = NULL;
  eps->arbitraryctx    = NULL;
  eps->numbermonitors  = 0;

  eps->st              = NULL;
  eps->ds              = NULL;
  eps->V               = NULL;
  eps->rg              = NULL;
  eps->rand            = NULL;
  eps->D               = NULL;
  eps->IS              = NULL;
  eps->defl            = NULL;
  eps->eigr            = NULL;
  eps->eigi            = NULL;
  eps->errest          = NULL;
  eps->rr              = NULL;
  eps->ri              = NULL;
  eps->perm            = NULL;
  eps->nwork           = 0;
  eps->work            = NULL;
  eps->data            = NULL;

  eps->state           = EPS_STATE_INITIAL;
  eps->nconv           = 0;
  eps->its             = 0;
  eps->nloc            = 0;
  eps->nrma            = 0.0;
  eps->nrmb            = 0.0;
  eps->isgeneralized   = PETSC_FALSE;
  eps->ispositive      = PETSC_FALSE;
  eps->ishermitian     = PETSC_FALSE;
  eps->reason          = EPS_CONVERGED_ITERATING;

  ierr = PetscNewLog(eps,&eps->sc);CHKERRQ(ierr);
  ierr = PetscRandomCreate(comm,&eps->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(eps->rand,0x12345678);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->rand);CHKERRQ(ierr);
  *outeps = eps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetType"
/*@C
   EPSSetType - Selects the particular solver to be used in the EPS object.

   Logically Collective on EPS

   Input Parameters:
+  eps  - the eigensolver context
-  type - a known method

   Options Database Key:
.  -eps_type <method> - Sets the method; use -help for a list
    of available methods

   Notes:
   See "slepc/include/slepceps.h" for available methods. The default
   is EPSKRYLOVSCHUR.

   Normally, it is best to use the EPSSetFromOptions() command and
   then set the EPS type from the options database rather than by using
   this routine.  Using the options database provides the user with
   maximum flexibility in evaluating the different available methods.
   The EPSSetType() routine is provided for those situations where it
   is necessary to set the iterative solver independently of the command
   line or options database.

   Level: intermediate

.seealso: STSetType(), EPSType
@*/
PetscErrorCode EPSSetType(EPS eps,EPSType type)
{
  PetscErrorCode ierr,(*r)(EPS);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)eps,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(EPSList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown EPS type given: %s",type);

  if (eps->ops->destroy) { ierr = (*eps->ops->destroy)(eps);CHKERRQ(ierr); }
  ierr = PetscMemzero(eps->ops,sizeof(struct _EPSOps));CHKERRQ(ierr);

  eps->state = EPS_STATE_INITIAL;
  ierr = PetscObjectChangeTypeName((PetscObject)eps,type);CHKERRQ(ierr);
  ierr = (*r)(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetType"
/*@C
   EPSGetType - Gets the EPS type as a string from the EPS object.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  name - name of EPS method

   Level: intermediate

.seealso: EPSSetType()
@*/
PetscErrorCode EPSGetType(EPS eps,EPSType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)eps)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSRegister"
/*@C
   EPSRegister - Adds a method to the eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   EPSRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
   EPSRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     EPSSetType(eps,"my_solver")
   or at runtime via the option
$     -eps_type my_solver

   Level: advanced

.seealso: EPSRegisterAll()
@*/
PetscErrorCode EPSRegister(const char *name,PetscErrorCode (*function)(EPS))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&EPSList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSReset"
/*@
   EPSReset - Resets the EPS context to the initial state and removes any
   allocated objects.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: advanced

.seealso: EPSDestroy()
@*/
PetscErrorCode EPSReset(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       ncols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  if (eps->ops->reset) { ierr = (eps->ops->reset)(eps);CHKERRQ(ierr); }
  if (eps->st) { ierr = STReset(eps->st);CHKERRQ(ierr); }
  if (eps->ds) { ierr = DSReset(eps->ds);CHKERRQ(ierr); }
  ierr = VecDestroy(&eps->D);CHKERRQ(ierr);
  ierr = BVGetSizes(eps->V,NULL,NULL,&ncols);CHKERRQ(ierr);
  if (ncols) {
    ierr = PetscFree4(eps->eigr,eps->eigi,eps->errest,eps->perm);CHKERRQ(ierr);
    ierr = PetscFree2(eps->rr,eps->ri);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&eps->V);CHKERRQ(ierr);
  ierr = VecDestroyVecs(eps->nwork,&eps->work);CHKERRQ(ierr);
  eps->nwork = 0;
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSDestroy"
/*@C
   EPSDestroy - Destroys the EPS context.

   Collective on EPS

   Input Parameter:
.  eps - eigensolver context obtained from EPSCreate()

   Level: beginner

.seealso: EPSCreate(), EPSSetUp(), EPSSolve()
@*/
PetscErrorCode EPSDestroy(EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*eps) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*eps,EPS_CLASSID,1);
  if (--((PetscObject)(*eps))->refct > 0) { *eps = 0; PetscFunctionReturn(0); }
  ierr = EPSReset(*eps);CHKERRQ(ierr);
  if ((*eps)->ops->destroy) { ierr = (*(*eps)->ops->destroy)(*eps);CHKERRQ(ierr); }
  ierr = STDestroy(&(*eps)->st);CHKERRQ(ierr);
  ierr = RGDestroy(&(*eps)->rg);CHKERRQ(ierr);
  ierr = DSDestroy(&(*eps)->ds);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&(*eps)->rand);CHKERRQ(ierr);
  ierr = PetscFree((*eps)->sc);CHKERRQ(ierr);
  /* just in case the initial vectors have not been used */
  ierr = SlepcBasisDestroy_Private(&(*eps)->nds,&(*eps)->defl);CHKERRQ(ierr);
  ierr = SlepcBasisDestroy_Private(&(*eps)->nini,&(*eps)->IS);CHKERRQ(ierr);
  if ((*eps)->convergeddestroy) {
    ierr = (*(*eps)->convergeddestroy)((*eps)->convergedctx);CHKERRQ(ierr);
  }
  ierr = EPSMonitorCancel(*eps);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetTarget"
/*@
   EPSSetTarget - Sets the value of the target.

   Logically Collective on EPS

   Input Parameters:
+  eps    - eigensolver context
-  target - the value of the target

   Options Database Key:
.  -eps_target <scalar> - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with EPSSetWhichEigenpairs().

   In the case of complex scalars, a complex value can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -eps_target 1.0+2.0i

   Level: beginner

.seealso: EPSGetTarget(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSSetTarget(EPS eps,PetscScalar target)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveScalar(eps,target,2);
  eps->target = target;
  if (!eps->st) { ierr = EPSGetST(eps,&eps->st);CHKERRQ(ierr); }
  ierr = STSetDefaultShift(eps->st,target);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetTarget"
/*@
   EPSGetTarget - Gets the value of the target.

   Not Collective

   Input Parameter:
.  eps - eigensolver context

   Output Parameter:
.  target - the value of the target

   Note:
   If the target was not set by the user, then zero is returned.

   Level: beginner

.seealso: EPSSetTarget()
@*/
PetscErrorCode EPSGetTarget(EPS eps,PetscScalar* target)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidScalarPointer(target,2);
  *target = eps->target;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetInterval"
/*@
   EPSSetInterval - Defines the computational interval for spectrum slicing.

   Logically Collective on EPS

   Input Parameters:
+  eps  - eigensolver context
.  inta - left end of the interval
-  intb - right end of the interval

   Options Database Key:
.  -eps_interval <a,b> - set [a,b] as the interval of interest

   Notes:
   Spectrum slicing is a technique employed for computing all eigenvalues of
   symmetric eigenproblems in a given interval. This function provides the
   interval to be considered. It must be used in combination with EPS_ALL, see
   EPSSetWhichEigenpairs().

   In the command-line option, two values must be provided. For an open interval,
   one can give an infinite, e.g., -eps_interval 1.0,inf or -eps_interval -inf,1.0.
   An open interval in the programmatic interface can be specified with
   PETSC_MAX_REAL and -PETSC_MAX_REAL.

   Level: intermediate

.seealso: EPSGetInterval(), EPSSetWhichEigenpairs()
@*/
PetscErrorCode EPSSetInterval(EPS eps,PetscReal inta,PetscReal intb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,inta,2);
  PetscValidLogicalCollectiveReal(eps,intb,3);
  if (inta>=intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be inta<intb");
  eps->inta = inta;
  eps->intb = intb;
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetInterval"
/*@
   EPSGetInterval - Gets the computational interval for spectrum slicing.

   Not Collective

   Input Parameter:
.  eps - eigensolver context

   Output Parameters:
+  inta - left end of the interval
-  intb - right end of the interval

   Level: intermediate

   Note:
   If the interval was not set by the user, then zeros are returned.

.seealso: EPSSetInterval()
@*/
PetscErrorCode EPSGetInterval(EPS eps,PetscReal* inta,PetscReal* intb)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(inta,2);
  PetscValidPointer(intb,3);
  if (inta) *inta = eps->inta;
  if (intb) *intb = eps->intb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetST"
/*@
   EPSSetST - Associates a spectral transformation object to the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  st   - the spectral transformation object

   Note:
   Use EPSGetST() to retrieve the spectral transformation context (for example,
   to free it at the end of the computations).

   Level: developer

.seealso: EPSGetST()
@*/
PetscErrorCode EPSSetST(EPS eps,ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(st,ST_CLASSID,2);
  PetscCheckSameComm(eps,1,st,2);
  ierr = PetscObjectReference((PetscObject)st);CHKERRQ(ierr);
  ierr = STDestroy(&eps->st);CHKERRQ(ierr);
  eps->st = st;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetST"
/*@C
   EPSGetST - Obtain the spectral transformation (ST) object associated
   to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  st - spectral transformation context

   Level: beginner

.seealso: EPSSetST()
@*/
PetscErrorCode EPSGetST(EPS eps,ST *st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(st,2);
  if (!eps->st) {
    ierr = STCreate(PetscObjectComm((PetscObject)eps),&eps->st);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->st);CHKERRQ(ierr);
  }
  *st = eps->st;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetBV"
/*@
   EPSSetBV - Associates a basis vectors object to the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  V   - the basis vectors object

   Note:
   Use EPSGetBV() to retrieve the basis vectors context (for example,
   to free them at the end of the computations).

   Level: advanced

.seealso: EPSGetBV()
@*/
PetscErrorCode EPSSetBV(EPS eps,BV V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);
  PetscCheckSameComm(eps,1,V,2);
  ierr = PetscObjectReference((PetscObject)V);CHKERRQ(ierr);
  ierr = BVDestroy(&eps->V);CHKERRQ(ierr);
  eps->V = V;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetBV"
/*@C
   EPSGetBV - Obtain the basis vectors object associated to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  V - basis vectors context

   Level: advanced

.seealso: EPSSetBV()
@*/
PetscErrorCode EPSGetBV(EPS eps,BV *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(V,2);
  if (!eps->V) {
    ierr = BVCreate(PetscObjectComm((PetscObject)eps),&eps->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->V);CHKERRQ(ierr);
  }
  *V = eps->V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetRG"
/*@
   EPSSetRG - Associates a region object to the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  rg  - the region object

   Note:
   Use EPSGetRG() to retrieve the region context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetRG()
@*/
PetscErrorCode EPSSetRG(EPS eps,RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(rg,RG_CLASSID,2);
  PetscCheckSameComm(eps,1,rg,2);
  ierr = PetscObjectReference((PetscObject)rg);CHKERRQ(ierr);
  ierr = RGDestroy(&eps->rg);CHKERRQ(ierr);
  eps->rg = rg;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetRG"
/*@C
   EPSGetRG - Obtain the region object associated to the eigensolver.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  rg - region context

   Level: advanced

.seealso: EPSSetRG()
@*/
PetscErrorCode EPSGetRG(EPS eps,RG *rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(rg,2);
  if (!eps->rg) {
    ierr = RGCreate(PetscObjectComm((PetscObject)eps),&eps->rg);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->rg);CHKERRQ(ierr);
  }
  *rg = eps->rg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSSetDS"
/*@
   EPSSetDS - Associates a direct solver object to the eigensolver.

   Collective on EPS

   Input Parameters:
+  eps - eigensolver context obtained from EPSCreate()
-  ds  - the direct solver object

   Note:
   Use EPSGetDS() to retrieve the direct solver context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: EPSGetDS()
@*/
PetscErrorCode EPSSetDS(EPS eps,DS ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidHeaderSpecific(ds,DS_CLASSID,2);
  PetscCheckSameComm(eps,1,ds,2);
  ierr = PetscObjectReference((PetscObject)ds);CHKERRQ(ierr);
  ierr = DSDestroy(&eps->ds);CHKERRQ(ierr);
  eps->ds = ds;
  ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetDS"
/*@C
   EPSGetDS - Obtain the direct solver object associated to the eigensolver object.

   Not Collective

   Input Parameters:
.  eps - eigensolver context obtained from EPSCreate()

   Output Parameter:
.  ds - direct solver context

   Level: advanced

.seealso: EPSSetDS()
@*/
PetscErrorCode EPSGetDS(EPS eps,DS *ds)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(ds,2);
  if (!eps->ds) {
    ierr = DSCreate(PetscObjectComm((PetscObject)eps),&eps->ds);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)eps->ds);CHKERRQ(ierr);
  }
  *ds = eps->ds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSIsGeneralized"
/*@
   EPSIsGeneralized - Ask if the EPS object corresponds to a generalized
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

.seealso: EPSIsHermitian(), EPSIsPositive()
@*/
PetscErrorCode EPSIsGeneralized(EPS eps,PetscBool* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(is,2);
  *is = eps->isgeneralized;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSIsHermitian"
/*@
   EPSIsHermitian - Ask if the EPS object corresponds to a Hermitian
   eigenvalue problem.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

.seealso: EPSIsGeneralized(), EPSIsPositive()
@*/
PetscErrorCode EPSIsHermitian(EPS eps,PetscBool* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(is,2);
  *is = eps->ishermitian;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSIsPositive"
/*@
   EPSIsPositive - Ask if the EPS object corresponds to an eigenvalue
   problem type that requires a positive (semi-) definite matrix B.

   Not collective

   Input Parameter:
.  eps - the eigenproblem solver context

   Output Parameter:
.  is - the answer

   Level: intermediate

.seealso: EPSIsGeneralized(), EPSIsHermitian()
@*/
PetscErrorCode EPSIsPositive(EPS eps,PetscBool* is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidPointer(is,2);
  *is = eps->ispositive;
  PetscFunctionReturn(0);
}

