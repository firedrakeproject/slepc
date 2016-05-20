/*
   Basic NEP routines, Create, View, etc.

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

#include <slepc/private/nepimpl.h>      /*I "slepcnep.h" I*/

PetscFunctionList NEPList = 0;
PetscBool         NEPRegisterAllCalled = PETSC_FALSE;
PetscClassId      NEP_CLASSID = 0;
PetscLogEvent     NEP_SetUp = 0,NEP_Solve = 0,NEP_Refine = 0,NEP_FunctionEval = 0,NEP_JacobianEval = 0,NEP_DerivativesEval = 0;

#undef __FUNCT__
#define __FUNCT__ "NEPCreate"
/*@
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
  ierr = NEPInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(nep,NEP_CLASSID,"NEP","Nonlinear Eigenvalue Problem","NEP",comm,NEPDestroy,NEPView);CHKERRQ(ierr);

  nep->max_it          = 0;
  nep->nev             = 1;
  nep->ncv             = 0;
  nep->mpd             = 0;
  nep->nini            = 0;
  nep->target          = 0.0;
  nep->tol             = PETSC_DEFAULT;
  nep->conv            = NEP_CONV_REL;
  nep->stop            = NEP_STOP_BASIC;
  nep->which           = (NEPWhich)0;
  nep->refine          = NEP_REFINE_NONE;
  nep->npart           = 1;
  nep->rtol            = PETSC_DEFAULT;
  nep->rits            = PETSC_DEFAULT;
  nep->scheme          = (NEPRefineScheme)0;
  nep->trackall        = PETSC_FALSE;

  nep->computefunction = NULL;
  nep->computejacobian = NULL;
  nep->functionctx     = NULL;
  nep->jacobianctx     = NULL;
  nep->computederivatives = NULL;
  nep->derivativesctx  = NULL;
  nep->converged       = NEPConvergedRelative;
  nep->convergeddestroy= NULL;
  nep->stopping        = NEPStoppingBasic;
  nep->stoppingdestroy = NULL;
  nep->convergedctx    = NULL;
  nep->stoppingctx     = NULL;
  nep->numbermonitors  = 0;

  nep->ds              = NULL;
  nep->V               = NULL;
  nep->rg              = NULL;
  nep->function        = NULL;
  nep->function_pre    = NULL;
  nep->jacobian        = NULL;
  nep->derivatives     = NULL;
  nep->A               = NULL;
  nep->f               = NULL;
  nep->nt              = 0;
  nep->mstr            = DIFFERENT_NONZERO_PATTERN;
  nep->IS              = NULL;
  nep->eigr            = NULL;
  nep->eigi            = NULL;
  nep->errest          = NULL;
  nep->perm            = NULL;
  nep->nwork           = 0;
  nep->work            = NULL;
  nep->data            = NULL;

  nep->state           = NEP_STATE_INITIAL;
  nep->nconv           = 0;
  nep->its             = 0;
  nep->n               = 0;
  nep->nloc            = 0;
  nep->nrma            = NULL;
  nep->fui             = (NEPUserInterface)0;
  nep->reason          = NEP_CONVERGED_ITERATING;

  ierr = PetscNewLog(nep,&nep->sc);CHKERRQ(ierr);
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

  ierr = PetscFunctionListFind(NEPList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)nep),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown NEP type given: %s",type);

  if (nep->ops->destroy) { ierr = (*nep->ops->destroy)(nep);CHKERRQ(ierr); }
  ierr = PetscMemzero(nep->ops,sizeof(struct _NEPOps));CHKERRQ(ierr);

  nep->state = NEP_STATE_INITIAL;
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
   NEPRegister - Adds a method to the nonlinear eigenproblem solver package.

   Not Collective

   Input Parameters:
+  name - name of a new user-defined solver
-  function - routine to create the solver context

   Notes:
   NEPRegister() may be called multiple times to add several user-defined solvers.

   Sample usage:
.vb
    NEPRegister("my_solver",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     NEPSetType(nep,"my_solver")
   or at runtime via the option
$     -nep_type my_solver

   Level: advanced

.seealso: NEPRegisterAll()
@*/
PetscErrorCode NEPRegister(const char *name,PetscErrorCode (*function)(NEP))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&NEPList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPReset_Problem"
/*
   NEPReset_Problem - Destroys the problem matrices.
@*/
PetscErrorCode NEPReset_Problem(NEP nep)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  ierr = MatDestroy(&nep->function);CHKERRQ(ierr);
  ierr = MatDestroy(&nep->function_pre);CHKERRQ(ierr);
  ierr = MatDestroy(&nep->jacobian);CHKERRQ(ierr);
  ierr = MatDestroy(&nep->derivatives);CHKERRQ(ierr);
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    ierr = MatDestroyMatrices(nep->nt,&nep->A);CHKERRQ(ierr);
    for (i=0;i<nep->nt;i++) {
      ierr = FNDestroy(&nep->f[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(nep->f);CHKERRQ(ierr);
    ierr = PetscFree(nep->nrma);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "NEPReset"
/*@
   NEPReset - Resets the NEP context to the initial state and removes any
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
  PetscInt       ncols;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (nep->ops->reset) { ierr = (nep->ops->reset)(nep);CHKERRQ(ierr); }
  if (nep->ds) { ierr = DSReset(nep->ds);CHKERRQ(ierr); }
  ierr = NEPReset_Problem(nep);CHKERRQ(ierr);
  ierr = BVGetSizes(nep->V,NULL,NULL,&ncols);CHKERRQ(ierr);
  if (ncols) {
    ierr = PetscFree4(nep->eigr,nep->eigi,nep->errest,nep->perm);CHKERRQ(ierr);
  }
  ierr = BVDestroy(&nep->V);CHKERRQ(ierr);
  ierr = VecDestroyVecs(nep->nwork,&nep->work);CHKERRQ(ierr);
  ierr = KSPDestroy(&nep->refineksp);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&nep->refinesubc);CHKERRQ(ierr);
  nep->nwork = 0;
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPDestroy"
/*@
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

  PetscFunctionBegin;
  if (!*nep) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*nep,NEP_CLASSID,1);
  if (--((PetscObject)(*nep))->refct > 0) { *nep = 0; PetscFunctionReturn(0); }
  ierr = NEPReset(*nep);CHKERRQ(ierr);
  if ((*nep)->ops->destroy) { ierr = (*(*nep)->ops->destroy)(*nep);CHKERRQ(ierr); }
  ierr = RGDestroy(&(*nep)->rg);CHKERRQ(ierr);
  ierr = DSDestroy(&(*nep)->ds);CHKERRQ(ierr);
  ierr = PetscFree((*nep)->sc);CHKERRQ(ierr);
  /* just in case the initial vectors have not been used */
  ierr = SlepcBasisDestroy_Private(&(*nep)->nini,&(*nep)->IS);CHKERRQ(ierr);
  if ((*nep)->convergeddestroy) {
    ierr = (*(*nep)->convergeddestroy)((*nep)->convergedctx);CHKERRQ(ierr);
  }
  ierr = NEPMonitorCancel(*nep);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(nep);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetBV"
/*@
   NEPSetBV - Associates a basis vectors object to the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  bv  - the basis vectors object

   Note:
   Use NEPGetBV() to retrieve the basis vectors context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetBV()
@*/
PetscErrorCode NEPSetBV(NEP nep,BV bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(bv,BV_CLASSID,2);
  PetscCheckSameComm(nep,1,bv,2);
  ierr = PetscObjectReference((PetscObject)bv);CHKERRQ(ierr);
  ierr = BVDestroy(&nep->V);CHKERRQ(ierr);
  nep->V = bv;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetBV"
/*@
   NEPGetBV - Obtain the basis vectors object associated to the nonlinear
   eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  bv - basis vectors context

   Level: advanced

.seealso: NEPSetBV()
@*/
PetscErrorCode NEPGetBV(NEP nep,BV *bv)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(bv,2);
  if (!nep->V) {
    ierr = BVCreate(PetscObjectComm((PetscObject)nep),&nep->V);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->V);CHKERRQ(ierr);
  }
  *bv = nep->V;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetRG"
/*@
   NEPSetRG - Associates a region object to the nonlinear eigensolver.

   Collective on NEP

   Input Parameters:
+  nep - eigensolver context obtained from NEPCreate()
-  rg  - the region object

   Note:
   Use NEPGetRG() to retrieve the region context (for example,
   to free it at the end of the computations).

   Level: advanced

.seealso: NEPGetRG()
@*/
PetscErrorCode NEPSetRG(NEP nep,RG rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(rg,RG_CLASSID,2);
  PetscCheckSameComm(nep,1,rg,2);
  ierr = PetscObjectReference((PetscObject)rg);CHKERRQ(ierr);
  ierr = RGDestroy(&nep->rg);CHKERRQ(ierr);
  nep->rg = rg;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->rg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetRG"
/*@
   NEPGetRG - Obtain the region object associated to the
   nonlinear eigensolver object.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  rg - region context

   Level: advanced

.seealso: NEPSetRG()
@*/
PetscErrorCode NEPGetRG(NEP nep,RG *rg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(rg,2);
  if (!nep->rg) {
    ierr = RGCreate(PetscObjectComm((PetscObject)nep),&nep->rg);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->rg);CHKERRQ(ierr);
  }
  *rg = nep->rg;
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
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->ds);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetDS"
/*@
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
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->ds);CHKERRQ(ierr);
  }
  *ds = nep->ds;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPRefineGetKSP"
/*@
   NEPRefineGetKSP - Obtain the ksp object used by the eigensolver
   object in the refinement phase.

   Not Collective

   Input Parameters:
.  nep - eigensolver context obtained from NEPCreate()

   Output Parameter:
.  ksp - ksp context

   Level: advanced

.seealso: NEPSetRefine()
@*/
PetscErrorCode NEPRefineGetKSP(NEP nep,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!nep->refineksp) {
    if (nep->npart>1) {
      /* Split in subcomunicators */
      ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)nep),&nep->refinesubc);CHKERRQ(ierr);
      ierr = PetscSubcommSetNumber(nep->refinesubc,nep->npart);CHKERRQ(ierr);CHKERRQ(ierr);
      ierr = PetscSubcommSetType(nep->refinesubc,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)nep,sizeof(PetscSubcomm));CHKERRQ(ierr);
    }
    ierr = KSPCreate((nep->npart==1)?PetscObjectComm((PetscObject)nep):PetscSubcommChild(nep->refinesubc),&nep->refineksp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)nep->refineksp);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(*ksp,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(*ksp,"nep_refine_");CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(*ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  *ksp = nep->refineksp;
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

   Options Database Key:
.  -nep_target <scalar> - the value of the target

   Notes:
   The target is a scalar value used to determine the portion of the spectrum
   of interest. It is used in combination with NEPSetWhichEigenpairs().

   In the case of complex scalars, a complex value can be provided in the
   command line with [+/-][realnumber][+/-]realnumberi with no spaces, e.g.
   -nep_target 1.0+2.0i

   Level: intermediate

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

   Note:
   If the target was not set by the user, then zero is returned.

   Level: intermediate

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

   Calling Sequence of fun:
$   fun(NEP nep,PetscScalar lambda,Mat F,Mat P,void *ctx)

+  nep    - the NEP context
.  lambda - the scalar argument where T(.) must be evaluated
.  T      - matrix that will contain T(lambda)
.  P      - (optional) different matrix to build the preconditioner
-  ctx    - (optional) user-defined context, as set by NEPSetFunction()

   Level: beginner

.seealso: NEPGetFunction(), NEPSetJacobian()
@*/
PetscErrorCode NEPSetFunction(NEP nep,Mat A,Mat B,PetscErrorCode (*fun)(NEP,PetscScalar,Mat,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  if (A) PetscCheckSameComm(nep,1,A,2);
  if (B) PetscCheckSameComm(nep,1,B,3);

  if (nep->fui && nep->fui!=NEP_USER_INTERFACE_CALLBACK) {  /* clean previous user info */
    ierr = NEPReset_Problem(nep);CHKERRQ(ierr);
  }

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
  nep->fui = NEP_USER_INTERFACE_CALLBACK;
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
PetscErrorCode NEPGetFunction(NEP nep,Mat *A,Mat *B,PetscErrorCode (**fun)(NEP,PetscScalar,Mat,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckCallback(nep,1);
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

   Calling Sequence of jac:
$   jac(NEP nep,PetscScalar lambda,Mat J,void *ctx)

+  nep    - the NEP context
.  lambda - the scalar argument where T'(.) must be evaluated
.  J      - matrix that will contain T'(lambda)
-  ctx    - (optional) user-defined context, as set by NEPSetJacobian()

   Level: beginner

.seealso: NEPSetFunction(), NEPGetJacobian()
@*/
PetscErrorCode NEPSetJacobian(NEP nep,Mat A,PetscErrorCode (*jac)(NEP,PetscScalar,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (A) PetscCheckSameComm(nep,1,A,2);

  if (nep->fui && nep->fui!=NEP_USER_INTERFACE_CALLBACK) {  /* clean previous user info */
    ierr = NEPReset_Problem(nep);CHKERRQ(ierr);
  }

  if (jac) nep->computejacobian = jac;
  if (ctx) nep->jacobianctx     = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&nep->jacobian);CHKERRQ(ierr);
    nep->jacobian = A;
  }
  nep->fui = NEP_USER_INTERFACE_CALLBACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetJacobian"
/*@C
   NEPGetJacobian - Returns the Jacobian matrix and optionally the user
   provided routine and context for evaluating the Jacobian.

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
PetscErrorCode NEPGetJacobian(NEP nep,Mat *A,PetscErrorCode (**jac)(NEP,PetscScalar,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckCallback(nep,1);
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

   Level: beginner

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
  if (nep->state) { ierr = NEPReset(nep);CHKERRQ(ierr); }
  /* clean previously stored information */
  ierr = NEPReset_Problem(nep);CHKERRQ(ierr);
  /* allocate space and copy matrices and functions */
  ierr = PetscMalloc1(n,&nep->A);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,n*sizeof(Mat));CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(A[i],MAT_CLASSID,3);
    ierr = PetscObjectReference((PetscObject)A[i]);CHKERRQ(ierr);
    nep->A[i] = A[i];
  }
  ierr = PetscMalloc1(n,&nep->f);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,n*sizeof(FN));CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(f[i],FN_CLASSID,4);
    ierr = PetscObjectReference((PetscObject)f[i]);CHKERRQ(ierr);
    nep->f[i] = f[i];
  }
  ierr = PetscCalloc1(n,&nep->nrma);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)nep,n*sizeof(PetscReal));CHKERRQ(ierr);
  nep->nt   = n;
  nep->mstr = str;
  nep->fui  = NEP_USER_INTERFACE_SPLIT;
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
  NEPCheckSplit(nep,1);
  if (k<0 || k>=nep->nt) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",nep->nt-1);
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
  NEPCheckSplit(nep,1);
  if (n)   *n = nep->nt;
  if (str) *str = nep->mstr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPSetDerivatives"
/*@C
   NEPSetDerivatives - Sets the function to compute the k-th derivative T^(k)(lambda)
   for any value of k (including 0), as well as the location to store the matrix.

   Logically Collective on NEP and Mat

   Input Parameters:
+  nep - the NEP context
.  A   - the matrix to store the computed derivative
.  der - routing to evaluate the k-th derivative (if NULL then NEP retains any
         previously set value)
-  ctx - [optional] user-defined context for private data for the derivatives
         evaluation routine (may be NULL) (if NULL then NEP retains any
         previously set value)

   Level: beginner

.seealso: NEPSetFunction(), NEPGetDerivatives()
@*/
PetscErrorCode NEPSetDerivatives(NEP nep,Mat A,PetscErrorCode (*der)(NEP,PetscScalar,PetscInt,Mat,void*),void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  if (A) PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  if (A) PetscCheckSameComm(nep,1,A,2);

  if (nep->fui && nep->fui!=NEP_USER_INTERFACE_DERIVATIVES) {  /* clean previous user info */
    ierr = NEPReset_Problem(nep);CHKERRQ(ierr);
  }

  if (der) nep->computederivatives = der;
  if (ctx) nep->derivativesctx     = ctx;
  if (A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ierr = MatDestroy(&nep->derivatives);CHKERRQ(ierr);
    nep->derivatives = A;
  }
  nep->fui = NEP_USER_INTERFACE_DERIVATIVES;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NEPGetDerivatives"
/*@C
   NEPGetDerivatives - Returns the derivatives matrix and optionally the user
   provided routine and context for evaluating the derivatives.

   Not Collective, but Mat object will be parallel if NEP object is

   Input Parameter:
.  nep - the nonlinear eigensolver context

   Output Parameters:
+  A   - location to stash the derivatives matrix (or NULL)
.  der - location to put derivatives function (or NULL)
-  ctx - location to stash derivatives context (or NULL)

   Level: advanced

.seealso: NEPSetDerivatives()
@*/
PetscErrorCode NEPGetDerivatives(NEP nep,Mat *A,PetscErrorCode (**der)(NEP,PetscScalar,PetscInt,Mat,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  NEPCheckDerivatives(nep,1);
  if (A)   *A   = nep->derivatives;
  if (der) *der = nep->computederivatives;
  if (ctx) *ctx = nep->derivativesctx;
  PetscFunctionReturn(0);
}

