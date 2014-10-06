/*
     Basic routines

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

#include <slepc-private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

PetscFunctionList FNList = 0;
PetscBool         FNRegisterAllCalled = PETSC_FALSE;
PetscClassId      FN_CLASSID = 0;
PetscLogEvent     FN_Evaluate = 0;
static PetscBool  FNPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "FNFinalizePackage"
/*@C
   FNFinalizePackage - This function destroys everything in the Slepc interface
   to the FN package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode FNFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&FNList);CHKERRQ(ierr);
  FNPackageInitialized = PETSC_FALSE;
  FNRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNInitializePackage"
/*@C
  FNInitializePackage - This function initializes everything in the FN package.
  It is called from PetscDLLibraryRegister() when using dynamic libraries, and
  on the first call to FNCreate() when using static libraries.

  Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode FNInitializePackage(void)
{
  char             logList[256];
  char             *className;
  PetscBool        opt;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (FNPackageInitialized) PetscFunctionReturn(0);
  FNPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Math function",&FN_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = FNRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("FNEvaluate",FN_CLASSID,&FN_Evaluate);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"fn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(FN_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,"-log_summary_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"fn",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(FN_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(FNFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNCreate"
/*@C
   FNCreate - Creates an FN context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  newfn - location to put the FN context

   Level: beginner

.seealso: FNDestroy(), FN
@*/
PetscErrorCode FNCreate(MPI_Comm comm,FN *newfn)
{
  FN             fn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(newfn,2);
  *newfn = 0;
  ierr = FNInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(fn,_p_FN,struct _FNOps,FN_CLASSID,"FN","Math Function","FN",comm,FNDestroy,FNView);CHKERRQ(ierr);
  fn->na       = 0;
  fn->alpha    = NULL;
  fn->nb       = 0;
  fn->beta     = NULL;

  *newfn = fn;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetOptionsPrefix"
/*@C
   FNSetOptionsPrefix - Sets the prefix used for searching for all
   FN options in the database.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  prefix - the prefix string to prepend to all FN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: FNAppendOptionsPrefix()
@*/
PetscErrorCode FNSetOptionsPrefix(FN fn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNAppendOptionsPrefix"
/*@C
   FNAppendOptionsPrefix - Appends to the prefix used for searching for all
   FN options in the database.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  prefix - the prefix string to prepend to all FN option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: FNSetOptionsPrefix()
@*/
PetscErrorCode FNAppendOptionsPrefix(FN fn,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetOptionsPrefix"
/*@C
   FNGetOptionsPrefix - Gets the prefix used for searching for all
   FN options in the database.

   Not Collective

   Input Parameters:
.  fn - the math function context

   Output Parameters:
.  prefix - pointer to the prefix string used is returned

   Notes: On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: FNSetOptionsPrefix(), FNAppendOptionsPrefix()
@*/
PetscErrorCode FNGetOptionsPrefix(FN fn,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)fn,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetType"
/*@C
   FNSetType - Selects the type for the FN object.

   Logically Collective on FN

   Input Parameter:
+  fn   - the math function context
-  type - a known type

   Notes:
   The default is FNRATIONAL, which includes polynomials as a particular
   case as well as simple functions such as f(x)=x and f(x)=constant.

   Level: intermediate

.seealso: FNGetType()
@*/
PetscErrorCode FNSetType(FN fn,FNType type)
{
  PetscErrorCode ierr,(*r)(FN);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)fn,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(FNList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested FN type %s",type);

  ierr = PetscMemzero(fn->ops,sizeof(struct _FNOps));CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)fn,type);CHKERRQ(ierr);
  ierr = (*r)(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetType"
/*@C
   FNGetType - Gets the FN type name (as a string) from the FN context.

   Not Collective

   Input Parameter:
.  fn - the math function context

   Output Parameter:
.  name - name of the math function

   Level: intermediate

.seealso: FNSetType()
@*/
PetscErrorCode FNGetType(FN fn,FNType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)fn)->type_name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetParameters"
/*@
   FNSetParameters - Sets the parameters that define the matematical function.

   Logically Collective on FN

   Input Parameters:
+  fn    - the math function context
.  na    - number of parameters in the first group
.  alpha - first group of parameters (array of scalar values)
.  nb    - number of parameters in the second group
-  beta  - second group of parameters (array of scalar values)

   Notes:
   In a rational function r(x) = p(x)/q(x), where p(x) and q(x) are polynomials,
   the parameters alpha and beta represent the coefficients of p(x) and q(x),
   respectively. Hence, p(x) is of degree na-1 and q(x) of degree nb-1.
   If nb is zero, then the function is assumed to be polynomial, r(x) = p(x).

   In other functions the parameters have other meanings.

   In polynomials, high order coefficients are stored in the first positions
   of the array, e.g. to represent x^2-3 use {1,0,-3}.

   Level: intermediate

.seealso: FNGetParameters()
@*/
PetscErrorCode FNSetParameters(FN fn,PetscInt na,PetscScalar *alpha,PetscInt nb,PetscScalar *beta)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveInt(fn,na,2);
  if (na<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument na cannot be negative");
  if (na) PetscValidPointer(alpha,3);
  PetscValidLogicalCollectiveInt(fn,nb,4);
  if (nb<0) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_OUTOFRANGE,"Argument nb cannot be negative");
  if (nb) PetscValidPointer(beta,5);
  fn->na = na;
  ierr = PetscFree(fn->alpha);CHKERRQ(ierr);
  if (na) {
    ierr = PetscMalloc1(na,&fn->alpha);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)fn,na*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<na;i++) fn->alpha[i] = alpha[i];
  }
  fn->nb = nb;
  ierr = PetscFree(fn->beta);CHKERRQ(ierr);
  if (nb) {
    ierr = PetscMalloc1(nb,&fn->beta);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)fn,nb*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<nb;i++) fn->beta[i] = beta[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNGetParameters"
/*@
   FNGetParameters - Returns the parameters that define the matematical function.

   Not Collective

   Input Parameter:
.  fn    - the math function context

   Output Parameters:
+  na    - number of parameters in the first group
.  alpha - first group of parameters (array of scalar values, length na)
.  nb    - number of parameters in the second group
-  beta  - second group of parameters (array of scalar values, length nb)

   Notes:
   The values passed by user with FNSetParameters() are returned (or null
   pointers otherwise).
   The alpha and beta arrays should be freed by the user when no longer needed.

   Level: intermediate

.seealso: FNSetParameters()
@*/
PetscErrorCode FNGetParameters(FN fn,PetscInt *na,PetscScalar *alpha[],PetscInt *nb,PetscScalar *beta[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (na) *na = fn->na;
  if (alpha) {
    if (!fn->na) *alpha = NULL;
    else {
      ierr = PetscMalloc1(fn->na,alpha);CHKERRQ(ierr);
      for (i=0;i<fn->na;i++) (*alpha)[i] = fn->alpha[i];
    }
  }
  if (nb) *nb = fn->nb;
  if (beta) {
    if (!fn->nb) *beta = NULL;
    else {
      ierr = PetscMalloc1(fn->nb,beta);CHKERRQ(ierr);
      for (i=0;i<fn->nb;i++) (*beta)[i] = fn->beta[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunction"
/*@
   FNEvaluateFunction - Computes the value of the function f(x) for a given x.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  x  - the value where the function must be evaluated

   Output Parameter:
.  y  - the result of f(x)

   Level: intermediate

.seealso: FNEvaluateDerivative(), FNEvaluateFunctionMat()
@*/
PetscErrorCode FNEvaluateFunction(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveScalar(fn,x,2);
  PetscValidType(fn,1);
  PetscValidPointer(y,3);
  ierr = PetscLogEventBegin(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  ierr = (*fn->ops->evaluatefunction)(fn,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateDerivative"
/*@
   FNEvaluateDerivative - Computes the value of the derivative f'(x) for a given x.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  x  - the value where the derivative must be evaluated

   Output Parameter:
.  y  - the result of f'(x)

   Level: intermediate

.seealso: FNEvaluateFunction()
@*/
PetscErrorCode FNEvaluateDerivative(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidLogicalCollectiveScalar(fn,x,2);
  PetscValidType(fn,1);
  PetscValidPointer(y,3);
  ierr = PetscLogEventBegin(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  ierr = (*fn->ops->evaluatederivative)(fn,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat_Sym_Default"
/*
   FNEvaluateFunctionMat_Sym_Default - given a symmetric matrix A,
   compute the matrix function as f(A)=Q*f(D)*Q' where the spectral
   decomposition of A is A=Q*D*Q' 
*/
static PetscErrorCode FNEvaluateFunctionMat_Sym_Default(FN fn,Mat A,Mat B)
{
#if defined(PETSC_MISSING_LAPACK_SYEV)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYEV - Lapack routines are unavailable");
#else
  PetscErrorCode ierr;
  PetscInt       i,j,m;
  PetscBLASInt   n,ld,lwork,info;
  PetscScalar    *As,*Bs,*Q,*W,*work,a,x,y,one=1.0,zero=0.0;
  PetscReal      *eig;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&As);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&Bs);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(m,&n);CHKERRQ(ierr);
  ld = n;

  /* workspace query and memory allocation */
  lwork = -1;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,As,&ld,eig,&a,&lwork,NULL,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
  ierr = PetscMalloc5(m,&eig,m*m,&Q,m*m,&W,lwork,&work,PetscMax(1,3*m-2),&rwork);CHKERRQ(ierr);
#else
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,As,&ld,eig,&a,&lwork,&info));
  ierr = PetscBLASIntCast((PetscInt)PetscRealPart(a),&lwork);CHKERRQ(ierr);
  ierr = PetscMalloc4(m,&eig,m*m,&Q,m*m,&W,lwork,&work);CHKERRQ(ierr);
#endif

  /* compute eigendecomposition */
  PetscStackCallBLAS("LAPACKlacpy",LAPACKlacpy_("L",&n,&n,As,&ld,Q,&ld));
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,Q,&ld,eig,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","L",&n,Q,&ld,eig,work,&lwork,&info));
#endif
  if (info) SETERRQ1(PetscObjectComm((PetscObject)fn),PETSC_ERR_LIB,"Error in Lapack xSYEV %i",info);

  /* W = f(Lambda)*Q' */
  for (i=0;i<n;i++) {
    x = eig[i];
    ierr = (*fn->ops->evaluatefunction)(fn,x,&y);CHKERRQ(ierr);  /* y = f(x) */
    for (j=0;j<n;j++) W[i+j*ld] = Q[j+i*ld]*y;
  }
  /* Bs = Q*W */
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&one,Q,&ld,W,&ld,&zero,Bs,&ld));
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree5(eig,Q,W,work,rwork);CHKERRQ(ierr);
#else
  ierr = PetscFree4(eig,Q,W,work);CHKERRQ(ierr);
#endif
  ierr = MatDenseRestoreArray(A,&As);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&Bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__
#define __FUNCT__ "FNEvaluateFunctionMat"
/*@
   FNEvaluateFunctionMat - Computes the value of the function f(A) for a given
   matrix A, where the result is also a matrix.

   Logically Collective on FN

   Input Parameters:
+  fn - the math function context
-  A  - matrix on which the function must be evaluated

   Output Parameter:
.  B  - matrix resulting from evaluating f(A)

   Notes:
   The matrix A must be a sequential dense Mat, with all entries equal on
   all processes (otherwise each process will compute different results).
   Matrix B must also be a sequential dense Mat. Both matrices must be
   square with the same dimensions.

   If A is known to be real symmetric or complex Hermitian then it is
   recommended to set the appropriate flag with MatSetOption(), so that
   a different algorithm that exploits symmetry is used.

   Level: advanced

.seealso: FNEvaluateFunction()
@*/
PetscErrorCode FNEvaluateFunctionMat(FN fn,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscBool      match,set,flg,symm=PETSC_FALSE;
  PetscInt       m,n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscValidHeaderSpecific(B,MAT_CLASSID,3);
  PetscValidType(fn,1);
  PetscValidType(A,2);
  PetscValidType(B,3);
  if (A==B) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_WRONG,"A and B arguments must be different");
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_SUP,"Mat A must be of type seqdense");
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&match);CHKERRQ(ierr);
  if (!match) SETERRQ(PetscObjectComm((PetscObject)fn),PETSC_ERR_SUP,"Mat B must be of type seqdense");
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ,"Mat A is not square (has %D rows, %D cols)",m,n);
  ierr = MatGetSize(B,&m,&n);CHKERRQ(ierr);
  if (m!=n) SETERRQ2(PetscObjectComm((PetscObject)fn),PETSC_ERR_ARG_SIZ,"Mat B is not square (has %D rows, %D cols)",m,n);

  /* check symmetry of A */
  ierr = MatIsHermitianKnown(A,&set,&flg);CHKERRQ(ierr);
  symm = set? flg: PETSC_FALSE;

  ierr = PetscLogEventBegin(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  if (symm) {
    if (fn->ops->evaluatefunctionmatsym) {
      ierr = (*fn->ops->evaluatefunctionmatsym)(fn,A,B);CHKERRQ(ierr);
    } else {
      ierr = FNEvaluateFunctionMat_Sym_Default(fn,A,B);CHKERRQ(ierr);
    }
  } else {
    if (fn->ops->evaluatefunctionmat) {
      ierr = (*fn->ops->evaluatefunctionmat)(fn,A,B);CHKERRQ(ierr);
    } else SETERRQ1(PetscObjectComm((PetscObject)fn),PETSC_ERR_SUP,"Matrix functions not implemented in FN type %s",((PetscObject)fn)->type_name);
  }
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  ierr = PetscLogEventEnd(FN_Evaluate,fn,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNSetFromOptions"
/*@
   FNSetFromOptions - Sets FN options from the options database.

   Collective on FN

   Input Parameters:
.  fn - the math function context

   Notes:
   To see all options, run your program with the -help option.

   Level: beginner
@*/
PetscErrorCode FNSetFromOptions(FN fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (!FNRegisterAllCalled) { ierr = FNRegisterAll();CHKERRQ(ierr); }
  /* Set default type (we do not allow changing it with -fn_type) */
  if (!((PetscObject)fn)->type_name) {
    ierr = FNSetType(fn,FNRATIONAL);CHKERRQ(ierr);
  }
  ierr = PetscObjectOptionsBegin((PetscObject)fn);CHKERRQ(ierr);
    ierr = PetscObjectProcessOptionsHandlers((PetscObject)fn);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNView"
/*@C
   FNView - Prints the FN data structure.

   Collective on FN

   Input Parameters:
+  fn - the math function context
-  viewer - optional visualization context

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
@*/
PetscErrorCode FNView(FN fn,PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fn,FN_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)fn));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(fn,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)fn,viewer);CHKERRQ(ierr);
    if (fn->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*fn->ops->view)(fn,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNDestroy"
/*@C
   FNDestroy - Destroys FN context that was created with FNCreate().

   Collective on FN

   Input Parameter:
.  fn - the math function context

   Level: beginner

.seealso: FNCreate()
@*/
PetscErrorCode FNDestroy(FN *fn)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*fn) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*fn,FN_CLASSID,1);
  if (--((PetscObject)(*fn))->refct > 0) { *fn = 0; PetscFunctionReturn(0); }
  ierr = PetscFree((*fn)->alpha);CHKERRQ(ierr);
  ierr = PetscFree((*fn)->beta);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(fn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FNRegister"
/*@C
   FNRegister - See Adds a mathematical function to the FN package.

   Not collective

   Input Parameters:
+  name - name of a new user-defined FN
-  function - routine to create context

   Notes:
   FNRegister() may be called multiple times to add several user-defined inner products.

   Level: advanced

.seealso: FNRegisterAll()
@*/
PetscErrorCode FNRegister(const char *name,PetscErrorCode (*function)(FN))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&FNList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode FNCreate_Rational(FN);
PETSC_EXTERN PetscErrorCode FNCreate_Exp(FN);

#undef __FUNCT__
#define __FUNCT__ "FNRegisterAll"
/*@C
   FNRegisterAll - Registers all of the math functions in the FN package.

   Not Collective

   Level: advanced
@*/
PetscErrorCode FNRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  FNRegisterAllCalled = PETSC_TRUE;
  ierr = FNRegister(FNRATIONAL,FNCreate_Rational);CHKERRQ(ierr);
  ierr = FNRegister(FNEXP,FNCreate_Exp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

