/*
    The ST (spectral transformation) interface routines related to the
    KSP object associated to it.

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

#undef __FUNCT__
#define __FUNCT__ "STMatMult"
/*@
   STMatMult - Computes the matrix-vector product y = T[k] x, where T[k] is
   the k-th matrix of the spectral transformation.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
.  k  - index of matrix to use
-  x  - the vector to be multiplied

   Output Parameter:
.  y - the result

   Level: developer

.seealso: STMatMultTranspose()
@*/
PetscErrorCode STMatMult(ST st,PetscInt k,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,k,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  STCheckMatrices(st,1);
  if (k<0 || k>=PetscMax(2,st->nmat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",st->nmat);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecLocked(y,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_MatMult,st,x,y,0);CHKERRQ(ierr);
  if (!st->T[k]) {
    /* T[k]=NULL means identity matrix */
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMult(st->T[k],x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ST_MatMult,st,x,y,0);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatMultTranspose"
/*@
   STMatMultTranspose - Computes the matrix-vector product y = T[k]' x, where T[k] is
   the k-th matrix of the spectral transformation.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
.  k  - index of matrix to use
-  x  - the vector to be multiplied

   Output Parameter:
.  y - the result

   Level: developer

.seealso: STMatMult()
@*/
PetscErrorCode STMatMultTranspose(ST st,PetscInt k,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidLogicalCollectiveInt(st,k,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidHeaderSpecific(y,VEC_CLASSID,4);
  STCheckMatrices(st,1);
  if (k<0 || k>=PetscMax(2,st->nmat)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"k must be between 0 and %D",st->nmat);
  if (x == y) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and y must be different vectors");
  VecLocked(y,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }
  ierr = VecLockPush(x);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_MatMultTranspose,st,x,y,0);CHKERRQ(ierr);
  if (!st->T[k]) {
    /* T[k]=NULL means identity matrix */
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(st->T[k],x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(ST_MatMultTranspose,st,x,y,0);CHKERRQ(ierr);
  ierr = VecLockPop(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatSolve"
/*@
   STMatSolve - Solves P x = b, where P is the preconditioner matrix of
   the spectral transformation, using a KSP object stored internally.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  b  - right hand side vector

   Output Parameter:
.  x - computed solution

   Level: developer

.seealso: STMatSolveTranspose()
@*/
PetscErrorCode STMatSolve(ST st,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       its;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  STCheckMatrices(st,1);
  if (x == b) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  VecLocked(x,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }
  ierr = VecLockPush(b);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_MatSolve,st,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)st,&flg,STPRECOND,STSHELL,"");CHKERRQ(ierr);
  if (!flg && !st->P) {
    /* P=NULL means identity matrix */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPSolve(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);
  ierr = PetscInfo1(st,"Linear solve iterations=%D\n",its);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_MatSolve,st,b,x,0);CHKERRQ(ierr);
  ierr = VecLockPop(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatSolveTranspose"
/*@
   STMatSolveTranspose - Solves P' x = b, where P is the preconditioner matrix of
   the spectral transformation, using a KSP object stored internally.

   Collective on ST

   Input Parameters:
.  st - the spectral transformation context
.  b  - right hand side vector

   Output Parameter:
.  x - computed solution

   Level: developer

.seealso: STMatSolve()
@*/
PetscErrorCode STMatSolveTranspose(ST st,Vec b,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       its;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(b,VEC_CLASSID,2);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  STCheckMatrices(st,1);
  if (x == b) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_ARG_IDN,"x and b must be different vectors");
  VecLocked(x,3);

  if (st->state!=ST_STATE_SETUP) { ierr = STSetUp(st);CHKERRQ(ierr); }
  ierr = VecLockPush(b);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(ST_MatSolveTranspose,st,b,x,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)st,&flg,STPRECOND,STSHELL,"");CHKERRQ(ierr);
  if (!flg && !st->P) {
    /* P=NULL means identity matrix */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPSolveTranspose(st->ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(st->ksp,&its);CHKERRQ(ierr);
  ierr = PetscInfo1(st,"Linear solve iterations=%D\n",its);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_MatSolveTranspose,st,b,x,0);CHKERRQ(ierr);
  ierr = VecLockPop(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STMatSetHermitian"
/*
   STMatSetHermitian - Sets the Hermitian flag to the ST matrix.

   Input Parameters:
.  st - the spectral transformation context
.  M  - matrix
*/
PetscErrorCode STMatSetHermitian(ST st,Mat M)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  PetscBool      set,aherm,mherm;
  PetscInt       i;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  mherm = PETSC_FALSE;
  for (i=0;i<st->nmat;i++) {
    ierr = MatIsHermitianKnown(st->A[i],&set,&aherm);CHKERRQ(ierr);
    if (!set) aherm = PETSC_FALSE;
    mherm = (mherm && aherm)? PETSC_TRUE: PETSC_FALSE;
    if (PetscRealPart(st->sigma)==0.0) break;
  }
  mherm = (mherm && PetscImaginaryPart(st->sigma)==0.0)? PETSC_TRUE: PETSC_FALSE;
  ierr = MatSetOption(M,MAT_HERMITIAN,mherm);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckFactorPackage"
PetscErrorCode STCheckFactorPackage(ST st)
{
  PetscErrorCode         ierr;
  PC                     pc;
  PetscMPIInt            size;
  PetscBool              flg;
  const MatSolverPackage stype;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)st),&size);CHKERRQ(ierr);
  if (size==1) PetscFunctionReturn(0);
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = PCFactorGetMatSolverPackage(pc,&stype);CHKERRQ(ierr);
  if (stype) {   /* currently selected PC is a factorization */
    ierr = PetscStrcmp(stype,MATSOLVERPETSC,&flg);CHKERRQ(ierr);
    if (flg) SETERRQ(PetscObjectComm((PetscObject)st),PETSC_ERR_SUP,"You chose to solve linear systems with a factorization, but in parallel runs you need to select an external package; see the users guide for details");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STSetKSP"
/*@
   STSetKSP - Sets the KSP object associated with the spectral
   transformation.

   Collective on ST

   Input Parameters:
+  st   - the spectral transformation context
-  ksp  - the linear system context

   Level: advanced
@*/
PetscErrorCode STSetKSP(ST st,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(st,1,ksp,2);
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&st->ksp);CHKERRQ(ierr);
  st->ksp = ksp;
  ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)st->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STGetKSP"
/*@
   STGetKSP - Gets the KSP object associated with the spectral
   transformation.

   Not Collective

   Input Parameter:
.  st - the spectral transformation context

   Output Parameter:
.  ksp  - the linear system context

   Level: intermediate
@*/
PetscErrorCode STGetKSP(ST st,KSP* ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidPointer(ksp,2);
  if (!st->ksp) {
    ierr = KSPCreate(PetscObjectComm((PetscObject)st),&st->ksp);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(st->ksp,((PetscObject)st)->prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(st->ksp,"st_");CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)st->ksp,(PetscObject)st,1);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)st,(PetscObject)st->ksp);CHKERRQ(ierr);
    ierr = KSPSetTolerances(st->ksp,SLEPC_DEFAULT_TOL,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  }
  *ksp = st->ksp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace_Default"
PetscErrorCode STCheckNullSpace_Default(ST st,BV V)
{
  PetscErrorCode ierr;
  PetscInt       nc,i,c;
  PetscReal      norm;
  Vec            *T,w,vi;
  Mat            A;
  PC             pc;
  MatNullSpace   nullsp;

  PetscFunctionBegin;
  ierr = BVGetNumConstraints(V,&nc);CHKERRQ(ierr);
  ierr = PetscMalloc1(nc,&T);CHKERRQ(ierr);
  if (!st->ksp) { ierr = STGetKSP(st,&st->ksp);CHKERRQ(ierr); }
  ierr = KSPGetPC(st->ksp,&pc);CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&A,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,NULL,&w);CHKERRQ(ierr);
  c = 0;
  for (i=0;i<nc;i++) {
    ierr = BVGetColumn(V,-nc+i,&vi);CHKERRQ(ierr);
    ierr = MatMult(A,vi,w);CHKERRQ(ierr);
    ierr = VecNorm(w,NORM_2,&norm);CHKERRQ(ierr);
    if (norm < 1e-8) {
      ierr = PetscInfo2(st,"Vector %D norm=%g\n",i,(double)norm);CHKERRQ(ierr);
      ierr = BVCreateVec(V,T+c);CHKERRQ(ierr);
      ierr = VecCopy(vi,T[c]);CHKERRQ(ierr);
      c++;
    }
    ierr = BVRestoreColumn(V,-nc+i,&vi);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  if (c>0) {
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)st),PETSC_FALSE,c,T,&nullsp);CHKERRQ(ierr);
    ierr = MatSetNullSpace(A,nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    ierr = VecDestroyVecs(c,&T);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(T);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "STCheckNullSpace"
/*@
   STCheckNullSpace - Given a basis vectors object, this function tests each
   of its constraint vectors to be a nullspace vector of the coefficient
   matrix of the associated KSP object. All these nullspace vectors are passed
   to the KSP object.

   Collective on ST

   Input Parameters:
+  st - the spectral transformation context
-  V  - basis vectors to be checked

   Note:
   This function allows to handle singular pencils and to solve some problems
   in which the nullspace is important (see the users guide for details).

   Level: developer

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode STCheckNullSpace(ST st,BV V)
{
  PetscErrorCode ierr;
  PetscInt       nc;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_CLASSID,1);
  PetscValidHeaderSpecific(V,BV_CLASSID,2);
  PetscValidType(st,1);
  PetscCheckSameComm(st,1,V,2);

  ierr = BVGetNumConstraints(V,&nc);CHKERRQ(ierr);
  if (nc && st->ops->checknullspace) {
    ierr = (*st->ops->checknullspace)(st,V);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

