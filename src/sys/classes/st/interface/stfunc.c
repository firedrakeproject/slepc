/*
    The ST (spectral transformation) interface routines, callable by users.

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

#include <slepc/private/stimpl.h>            /*I "slepcst.h" I*/

PetscClassId     ST_CLASSID = 0;
PetscLogEvent    ST_SetUp = 0,ST_Apply = 0,ST_ApplyTranspose = 0,ST_MatSetUp = 0,ST_MatMult = 0,ST_MatMultTranspose = 0,ST_MatSolve = 0,ST_MatSolveTranspose = 0;
static PetscBool STPackageInitialized = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "STFinalizePackage"
/*@C
   STFinalizePackage - This function destroys everything in the Slepc interface
   to the ST package. It is called from SlepcFinalize().

   Level: developer

.seealso: SlepcFinalize()
@*/
PetscErrorCode STFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&STList);CHKERRQ(ierr);
  STPackageInitialized = PETSC_FALSE;
  STRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STInitializePackage"
/*@C
   STInitializePackage - This function initializes everything in the ST package.
   It is called from PetscDLLibraryRegister() when using dynamic libraries, and
   on the first call to STCreate() when using static libraries.

   Level: developer

.seealso: SlepcInitialize()
@*/
PetscErrorCode STInitializePackage(void)
{
  char           logList[256];
  char           *className;
  PetscBool      opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (STPackageInitialized) PetscFunctionReturn(0);
  STPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Spectral Transform",&ST_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = STRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("STSetUp",ST_CLASSID,&ST_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STApply",ST_CLASSID,&ST_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STApplyTranspose",ST_CLASSID,&ST_ApplyTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STMatSetUp",ST_CLASSID,&ST_MatSetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STMatMult",ST_CLASSID,&ST_MatMult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STMatMultTranspose",ST_CLASSID,&ST_MatMultTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STMatSolve",ST_CLASSID,&ST_MatSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("STMatSolveTranspose",ST_CLASSID,&ST_MatSolveTranspose);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"st",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(ST_CLASSID);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,256,&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList,"st",&className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(ST_CLASSID);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(STFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STReset"
/*@
   STReset - Resets the ST context and removes any allocated objects.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: advanced

.seealso: STDestroy()
@*/
PetscErrorCode STReset(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (st->ops->reset) { ierr = (*st->ops->reset)(st);CHKERRQ(ierr); }
  if (st->ksp) { ierr = KSPReset(st->ksp);CHKERRQ(ierr); }
  ierr = MatDestroyMatrices(PetscMax(2,st->nmat),&st->T);CHKERRQ(ierr);
  ierr = VecDestroy(&st->w);CHKERRQ(ierr);
  ierr = VecDestroy(&st->wb);CHKERRQ(ierr);
  st->state = ST_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STDestroy"
/*@
   STDestroy - Destroys ST context that was created with STCreate().

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: beginner

.seealso: STCreate(), STSetUp()
@*/
PetscErrorCode STDestroy(ST *st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*st) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*st,ST_CLASSID,1);
  if (--((PetscObject)(*st))->refct > 0) { *st = 0; PetscFunctionReturn(0); }
  ierr = STReset(*st);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(PetscMax(2,(*st)->nmat),&(*st)->A);CHKERRQ(ierr);
  ierr = PetscFree((*st)->Astate);CHKERRQ(ierr);
  if ((*st)->ops->destroy) { ierr = (*(*st)->ops->destroy)(*st);CHKERRQ(ierr); }
  ierr = MatDestroy(&(*st)->P);CHKERRQ(ierr);
  ierr = VecDestroy(&(*st)->D);CHKERRQ(ierr);
  ierr = KSPDestroy(&(*st)->ksp);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(st);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCreate"
/*@
   STCreate - Creates a spectral transformation context.

   Collective on MPI_Comm

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  st - location to put the spectral transformation context

   Level: beginner

.seealso: STSetUp(), STApply(), STDestroy(), ST
@*/
PetscErrorCode STCreate(MPI_Comm comm,ST *newst)
{
  PetscErrorCode ierr;
  ST             st;

  PetscFunctionBegin;
  PetscValidPointer(newst,2);
  *newst = 0;
  ierr = STInitializePackage();CHKERRQ(ierr);
  ierr = SlepcHeaderCreate(st,ST_CLASSID,"ST","Spectral Transformation","ST",comm,STDestroy,STView);CHKERRQ(ierr);

  st->A            = NULL;
  st->Astate       = NULL;
  st->T            = NULL;
  st->P            = NULL;
  st->nmat         = 0;
  st->sigma        = 0.0;
  st->sigma_set    = PETSC_FALSE;
  st->defsigma     = 0.0;
  st->shift_matrix = ST_MATMODE_COPY;
  st->str          = DIFFERENT_NONZERO_PATTERN;
  st->transform    = PETSC_FALSE;

  st->ksp          = NULL;
  st->w            = NULL;
  st->D            = NULL;
  st->wb           = NULL;
  st->data         = NULL;
  st->state        = ST_STATE_INITIAL;

  *newst = st;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetOperators"
/*@
   STSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on ST and Mat

   Input Parameters:
+  st - the spectral transformation context
.  n  - number of matrices in array A
-  A  - the array of matrices associated with the eigensystem

   Notes:
   It must be called before STSetUp(). If it is called again after STSetUp() then
   the ST object is reset.

   Level: intermediate

.seealso: STGetOperators(), STGetNumMatrices(), STSetUp(), STReset()
 @*/
PetscErrorCode STSetOperators(ST st,PetscInt n,Mat A[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscBool      same=PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,n,2);
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have one or more matrices, you have %D",n);
  PetscValidPointer(A,3);
  PetscCheckSameComm(st,1,*A,3);
  if (st->state) {
    if (n!=st->nmat) same = PETSC_FALSE;
    for (i=0;same&&i<n;i++) {
      if (A[i]!=st->A[i]) same = PETSC_FALSE;
    }
    if (!same) { ierr = STReset(st);CHKERRQ(ierr); }
  } else same = PETSC_FALSE;
  if (!same) {
    ierr = MatDestroyMatrices(PetscMax(2,st->nmat),&st->A);CHKERRQ(ierr);
    ierr = PetscCalloc1(PetscMax(2,n),&st->A);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)st,PetscMax(2,n)*sizeof(Mat));CHKERRQ(ierr);
    ierr = PetscFree(st->Astate);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax(2,n),&st->Astate);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)st,PetscMax(2,n)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  for (i=0;i<n;i++) {
    PetscValidHeaderSpecific(A[i],MAT_CLASSID,3);
    ierr = PetscObjectReference((PetscObject)A[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&st->A[i]);CHKERRQ(ierr);
    st->A[i] = A[i];
    st->Astate[i] = ((PetscObject)A[i])->state;
  }
  if (n==1) {
    st->A[1] = NULL;
    st->Astate[1] = 0;
  }
  st->nmat = n;
  if (same) st->state = ST_STATE_UPDATED;
  else st->state = ST_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetOperators"
/*@
   STGetOperators - Gets the matrices associated with the original eigensystem.

   Not collective, though parallel Mats are returned if the ST is parallel

   Input Parameter:
+  st - the spectral transformation context
-  k  - the index of the requested matrix (starting in 0)

   Output Parameters:
.  A - the requested matrix

   Level: intermediate

.seealso: STSetOperators(), STGetNumMatrices()
@*/
PetscErrorCode STGetOperators(ST st,PetscInt k,Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,k,2);
  PetscValidPointer(A,3);
  STCheckMatrices(st,1);
  if (k<0 || k>=st->nmat) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",st->nmat-1);
  if (((PetscObject)st->A[k])->state!=st->Astate[k]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot retrieve original matrices (have been modified)");
  *A = st->A[k];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetTOperators"
/*@
   STGetTOperators - Gets the matrices associated with the transformed eigensystem.

   Not collective, though parallel Mats are returned if the ST is parallel

   Input Parameter:
+  st - the spectral transformation context
-  k  - the index of the requested matrix (starting in 0)

   Output Parameters:
.  T - the requested matrix

   Level: developer

.seealso: STGetOperators(), STGetNumMatrices()
@*/
PetscErrorCode STGetTOperators(ST st,PetscInt k,Mat *T)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,k,2);
  PetscValidPointer(T,3);
  STCheckMatrices(st,1);
  if (k<0 || k>=st->nmat) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",st->nmat-1);
  if (!st->T) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_POINTER,"There are no transformed matrices");
  *T = st->T[k];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetNumMatrices"
/*@
   STGetNumMatrices - Returns the number of matrices stored in the ST.

   Not collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameters:
.  n - the number of matrices passed in STSetOperators()

   Level: intermediate

.seealso: STSetOperators()
@*/
PetscErrorCode STGetNumMatrices(ST st,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(n,2);
  *n = st->nmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetShift"
/*@
   STSetShift - Sets the shift associated with the spectral transformation.

   Logically Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  shift - the value of the shift

   Notes:
   In some spectral transformations, changing the shift may have associated
   a lot of work, for example recomputing a factorization.

   This function is normally not directly called by users, since the shift is
   indirectly set by EPSSetTarget().

   Level: intermediate

.seealso: EPSSetTarget(), STGetShift(), STSetDefaultShift()
@*/
PetscErrorCode STSetShift(ST st,PetscScalar shift)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,shift,2);
  if (st->state==ST_STATE_SETUP && st->sigma != shift) {
    if (st->ops->setshift) {
      ierr = (*st->ops->setshift)(st,shift);CHKERRQ(ierr);
    }
  }
  st->sigma = shift;
  st->sigma_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetShift"
/*@
   STGetShift - Gets the shift associated with the spectral transformation.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  shift - the value of the shift

   Level: intermediate

.seealso: STSetShift()
@*/
PetscErrorCode STGetShift(ST st,PetscScalar* shift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidScalarPointer(shift,2);
  *shift = st->sigma;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetDefaultShift"
/*@
   STSetDefaultShift - Sets the value of the shift that should be employed if
   the user did not specify one.

   Logically Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  defaultshift - the default value of the shift

   Level: developer

.seealso: STSetShift()
@*/
PetscErrorCode STSetDefaultShift(ST st,PetscScalar defaultshift)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,defaultshift,2);
  st->defsigma = defaultshift;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STScaleShift"
/*@
   STScaleShift - Multiply the shift with a given factor.

   Logically Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  factor - the scaling factor

   Note:
   This function does not update the transformation matrices, as opposed to
   STSetShift().

   Level: developer

.seealso: STSetShift()
@*/
PetscErrorCode STScaleShift(ST st,PetscScalar factor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveScalar(st,factor,2);
  st->sigma *= factor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetBalanceMatrix"
/*@
   STSetBalanceMatrix - Sets the diagonal matrix to be used for balancing.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  D  - the diagonal matrix (represented as a vector)

   Notes:
   If this matrix is set, STApply will effectively apply D*OP*D^{-1}.

   Balancing is usually set via EPSSetBalance, but the advanced user may use
   this function to bypass the usual balancing methods.

   Level: developer

.seealso: EPSSetBalance(), STApply(), STGetBalanceMatrix()
@*/
PetscErrorCode STSetBalanceMatrix(ST st,Vec D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(D,VEC_CLASSID,2);
  PetscCheckSameComm(st,1,D,2);
  ierr = PetscObjectReference((PetscObject)D);CHKERRQ(ierr);
  ierr = VecDestroy(&st->D);CHKERRQ(ierr);
  st->D = D;
  st->state = ST_STATE_INITIAL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetBalanceMatrix"
/*@
   STGetBalanceMatrix - Gets the balance matrix used by the spectral transformation.

   Not collective, but vector is shared by all processors that share the ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  D  - the diagonal matrix (represented as a vector)

   Note:
   If the matrix was not set, a null pointer will be returned.

   Level: developer

.seealso: STSetBalanceMatrix()
@*/
PetscErrorCode STGetBalanceMatrix(ST st,Vec *D)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(D,2);
  *D = st->D;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatCreateVecs"
/*@C
   STMatCreateVecs - Get vector(s) compatible with the ST matrices.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Output Parameters:
+  right - (optional) vector that the matrix can be multiplied against
-  left  - (optional) vector that the matrix vector product can be stored in

   Level: developer
@*/
PetscErrorCode STMatCreateVecs(ST st,Vec *right,Vec *left)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STCheckMatrices(st,1);
  ierr = MatCreateVecs(st->A[0],right,left);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatGetSize"
/*@
   STMatGetSize - Returns the number of rows and columns of the ST matrices.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameters:
+  m - the number of global rows
-  n - the number of global columns

   Level: developer
@*/
PetscErrorCode STMatGetSize(ST st,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STCheckMatrices(st,1);
  ierr = MatGetSize(st->A[0],m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatGetLocalSize"
/*@
   STMatGetLocalSize - Returns the number of local rows and columns of the ST matrices.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameters:
+  m - the number of local rows
-  n - the number of local columns

   Level: developer
@*/
PetscErrorCode STMatGetLocalSize(ST st,PetscInt *m,PetscInt *n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  STCheckMatrices(st,1);
  ierr = MatGetLocalSize(st->A[0],m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "STSetOptionsPrefix"
/*@C
   STSetOptionsPrefix - Sets the prefix used for searching for all
   ST options in the database.

   Logically Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  prefix - the prefix string to prepend to all ST option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: STAppendOptionsPrefix(), STGetOptionsPrefix()
@*/
PetscErrorCode STSetOptionsPrefix(ST st,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOptionsPrefix(st->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)st,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STAppendOptionsPrefix"
/*@C
   STAppendOptionsPrefix - Appends to the prefix used for searching for all
   ST options in the database.

   Logically Collective on ST

   Input Parameters:
+  st     - the spectral transformation context
-  prefix - the prefix string to prepend to all ST option requests

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the
   hyphen.

   Level: advanced

.seealso: STSetOptionsPrefix(), STGetOptionsPrefix()
@*/
PetscErrorCode STAppendOptionsPrefix(ST st,const char *prefix)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)st,prefix);CHKERRQ(ierr);
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPSetOptionsPrefix(st->ksp,((PetscObject)st)->prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetOptionsPrefix"
/*@C
   STGetOptionsPrefix - Gets the prefix used for searching for all
   ST options in the database.

   Not Collective

   Input Parameters:
.  st - the spectral transformation context

   Output Parameters:
.  prefix - pointer to the prefix string used, is returned

   Note:
   On the Fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: STSetOptionsPrefix(), STAppendOptionsPrefix()
@*/
PetscErrorCode STGetOptionsPrefix(ST st,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(prefix,2);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)st,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STView"
/*@C
   STView - Prints the ST data structure.

   Collective on ST

   Input Parameters:
+  st - the ST context
-  viewer - optional visualization context

   Note:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their
         data to the first processor to print.

   The user can open an alternative visualization contexts with
   PetscViewerASCIIOpen() (output to a specified file).

   Level: beginner

.seealso: EPSView(), PetscViewerASCIIOpen()
@*/
PetscErrorCode STView(ST st,PetscViewer viewer)
{
  PetscErrorCode ierr;
  STType         cstr;
  const char*    pat;
  char           str[50];
  PetscBool      isascii,isstring,flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)st));
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(st,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)st,viewer);CHKERRQ(ierr);
    if (st->ops->view) {
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    ierr = SlepcSNPrintfScalar(str,50,st->sigma,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  shift: %s\n",str);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of matrices: %D\n",st->nmat);CHKERRQ(ierr);
    switch (st->shift_matrix) {
    case ST_MATMODE_COPY:
      break;
    case ST_MATMODE_INPLACE:
      ierr = PetscViewerASCIIPrintf(viewer,"  shifting the matrix and unshifting at exit\n");CHKERRQ(ierr);
      break;
    case ST_MATMODE_SHELL:
      ierr = PetscViewerASCIIPrintf(viewer,"  using a shell matrix\n");CHKERRQ(ierr);
      break;
    }
    if (st->nmat>1 && st->shift_matrix != ST_MATMODE_SHELL) {
      switch (st->str) {
        case SAME_NONZERO_PATTERN:      pat = "same nonzero pattern";break;
        case DIFFERENT_NONZERO_PATTERN: pat = "different nonzero pattern";break;
        case SUBSET_NONZERO_PATTERN:    pat = "subset nonzero pattern";break;
        default: SETERRQ(PetscObjectComm((PetscObject)st),1,"Wrong structure flag");
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  all matrices have %s\n",pat);CHKERRQ(ierr);
    }
    if (st->transform && st->nmat>2) {
      ierr = PetscViewerASCIIPrintf(viewer,"  computing transformed matrices\n");CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = STGetType(st,&cstr);CHKERRQ(ierr);
    ierr = PetscViewerStringSPrintf(viewer," %-7.7s",cstr);CHKERRQ(ierr);
    if (st->ops->view) { ierr = (*st->ops->view)(st,viewer);CHKERRQ(ierr); }
  }
  ierr = PetscObjectTypeCompare((PetscObject)st,STSHIFT,&flg);CHKERRQ(ierr);
  if (st->nmat>1 || !flg) {
    if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(st->ksp,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STRegister"
/*@C
   STRegister - Adds a method to the spectral transformation package.

   Not collective

   Input Parameters:
+  name - name of a new user-defined transformation
-  function - routine to create method context

   Notes:
   STRegister() may be called multiple times to add several user-defined
   spectral transformations.

   Sample usage:
.vb
    STRegister("my_transform",MyTransformCreate);
.ve

   Then, your spectral transform can be chosen with the procedural interface via
$     STSetType(st,"my_transform")
   or at runtime via the option
$     -st_type my_transform

   Level: advanced

.seealso: STRegisterAll()
@*/
PetscErrorCode STRegister(const char *name,PetscErrorCode (*function)(ST))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListAdd(&STList,name,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

