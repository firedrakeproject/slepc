/*
      EPS routines related to problem setup.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/epsimpl.h"   /*I "slepceps.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp"
/*@
   EPSSetUp - Sets up all the internal data structures necessary for the
   execution of the eigensolver. Then calls STSetUp() for any set-up
   operations associated to the ST object.

   Collective on EPS

   Input Parameter:
.  eps   - eigenproblem solver context

   Notes:
   This function need not be called explicitly in most cases, since EPSSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   This function sets a random initial vector if none has been provided.

   Level: advanced

.seealso: EPSCreate(), EPSSolve(), EPSDestroy(), STSetUp(), EPSSetInitialVector()
@*/
PetscErrorCode EPSSetUp(EPS eps)
{
  PetscErrorCode ierr;
  Vec            v0,w0,vds;  
  Mat            A,B; 
  PetscInt       i,k;
  PetscTruth     isCayley,lindep;
  PetscScalar    *pDS;
  PetscReal      norm;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    sigma;
#endif
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);

  if (eps->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!((PetscObject)eps)->type_name) {
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  }
  if (!eps->balance) eps->balance = EPSBALANCE_NONE;
  
  /* Set problem dimensions */
  ierr = STGetOperators(eps->OP,&A,&B);CHKERRQ(ierr);
  ierr = MatGetSize(A,&eps->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&eps->nloc,PETSC_NULL);CHKERRQ(ierr);

  /* Set default problem type */
  if (!eps->problem_type) {
    if (B==PETSC_NULL) {
      ierr = EPSSetProblemType(eps,EPS_NHEP);CHKERRQ(ierr);
    }
    else {
      ierr = EPSSetProblemType(eps,EPS_GNHEP);CHKERRQ(ierr);
    }
  } else if ((B && !eps->isgeneralized) || (!B && eps->isgeneralized)) {
    SETERRQ(0,"Warning: Inconsistent EPS state"); 
  }
#if defined(PETSC_USE_COMPLEX)
  ierr = STGetShift(eps->OP,&sigma);CHKERRQ(ierr);
  if (eps->ishermitian && PetscImaginaryPart(sigma) != 0.0) {
    SETERRQ(1,"Hermitian problems are not compatible with complex shifts")
  }
#endif
  
  if (eps->ispositive) {
    ierr = STGetBilinearForm(eps->OP,&B);CHKERRQ(ierr);
    ierr = IPSetBilinearForm(eps->ip,B,IPINNER_HERMITIAN);CHKERRQ(ierr);
    ierr = MatDestroy(B);CHKERRQ(ierr);
  } else {
    ierr = IPSetBilinearForm(eps->ip,PETSC_NULL,IPINNER_HERMITIAN);CHKERRQ(ierr);
  }
  
  /* Create random initial vectors if not set */
  /* right */
  ierr = EPSGetInitialVector(eps,&v0);CHKERRQ(ierr);
  if (!v0) {
    ierr = MatGetVecs(A,&v0,PETSC_NULL);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(v0);CHKERRQ(ierr);
    eps->vec_initial = v0;
  }
  /* left */
  ierr = EPSGetLeftInitialVector(eps,&w0);CHKERRQ(ierr);
  if (!w0) {
    ierr = MatGetVecs(A,PETSC_NULL,&w0);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(w0);CHKERRQ(ierr);
    eps->vec_initial_left = w0;
  }

  if (eps->nev > eps->n) eps->nev = eps->n;
  if (eps->ncv > eps->n) eps->ncv = eps->n;

  ierr = (*eps->ops->setup)(eps);CHKERRQ(ierr);
  ierr = STSetUp(eps->OP); CHKERRQ(ierr); 
  
  ierr = PetscTypeCompare((PetscObject)eps->OP,STCAYLEY,&isCayley);CHKERRQ(ierr);
  if (isCayley && eps->problem_type == EPS_PGNHEP) {
    SETERRQ(PETSC_ERR_SUP,"Cayley spectral transformation is not compatible with PGNHEP");
  }

  if (eps->nds>0) {
    if (!eps->ds_ortho) {
      /* allocate memory and copy deflation basis vectors into DS */
      ierr = PetscMalloc(eps->nds*eps->nloc*sizeof(PetscScalar),&pDS);CHKERRQ(ierr);
      for (i=0;i<eps->nds;i++) {
        ierr = VecCreateMPIWithArray(((PetscObject)eps)->comm,eps->nloc,PETSC_DECIDE,pDS+i*eps->nloc,&vds);CHKERRQ(ierr);
        ierr = VecCopy(eps->DS[i],vds);CHKERRQ(ierr);
        ierr = VecDestroy(eps->DS[i]);CHKERRQ(ierr);
        eps->DS[i] = vds;
      }
      /* orthonormalize vectors in DS */
      ierr = IPQRDecomposition(eps->ip,eps->DS,0,eps->nds,PETSC_NULL,0);CHKERRQ(ierr);
      eps->ds_ortho = PETSC_TRUE;
    }
    ierr = IPOrthogonalize(eps->ip,0,PETSC_NULL,eps->nds,PETSC_NULL,eps->DS,eps->vec_initial,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr); 
  }
  ierr = STCheckNullSpace(eps->OP,eps->nds,eps->DS);CHKERRQ(ierr);

  /* process initial vectors */
  if (eps->nini<0) {
    eps->nini = -eps->nini;
    if (eps->nini>eps->ncv) SETERRQ(1,"The number of initial vectors is larger than ncv")
    k = 0;
    for (i=0;i<eps->nini;i++) {
      ierr = VecCopy(eps->IS[i],eps->V[k]);CHKERRQ(ierr);
      ierr = VecDestroy(eps->IS[i]);CHKERRQ(ierr);
      ierr = IPOrthogonalize(eps->ip,eps->nds,eps->DS,k,PETSC_NULL,eps->V,eps->V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
      if (norm==0.0 || lindep) PetscInfo(eps,"Linearly dependent initial vector found, removing...\n");
      else {
        ierr = VecScale(eps->V[k],1.0/norm);CHKERRQ(ierr);
        k++;
      }
    }
    eps->nini = k;
    ierr = PetscFree(eps->IS);CHKERRQ(ierr);
  }

  /* Build balancing matrix if required */
  if (!eps->ishermitian && (eps->balance==EPSBALANCE_ONESIDE || eps->balance==EPSBALANCE_TWOSIDE)) {
    if (!eps->D) {
      ierr = VecDuplicate(eps->vec_initial,&eps->D);CHKERRQ(ierr);
    }
    else {
      ierr = VecSet(eps->D,1.0);CHKERRQ(ierr);
    }
    ierr = EPSBuildBalance_Krylov(eps);CHKERRQ(ierr);
    ierr = STSetBalanceMatrix(eps->OP,eps->D);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(EPS_SetUp,eps,0,0,0);CHKERRQ(ierr);
  eps->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialVector"
/*@
   EPSSetInitialVector - Sets the initial vector from which the 
   eigensolver starts to iterate.

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: EPSGetInitialVector(), EPSSetLeftInitialVector()

@*/
PetscErrorCode EPSSetInitialVector(EPS eps,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial); CHKERRQ(ierr);
  }
  eps->vec_initial = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetInitialVector"
/*@
   EPSGetInitialVector - Gets the initial vector associated with the 
   eigensolver; if the vector was not set it will return a null pointer or
   a vector randomly generated by EPSSetUp().

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: EPSSetInitialVector(), EPSGetLeftInitialVector()

@*/
PetscErrorCode EPSGetInitialVector(EPS eps,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,2);
  *vec = eps->vec_initial;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetLeftInitialVector"
/*@
   EPSSetLeftInitialVector - Sets the initial vector from which the eigensolver 
   starts to iterate, corresponding to the left recurrence (two-sided solvers).

   Collective on EPS and Vec

   Input Parameters:
+  eps - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: EPSGetLeftInitialVector(), EPSSetInitialVector()

@*/
PetscErrorCode EPSSetLeftInitialVector(EPS eps,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(eps,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (eps->vec_initial_left) {
    ierr = VecDestroy(eps->vec_initial_left); CHKERRQ(ierr);
  }
  eps->vec_initial_left = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSGetLeftInitialVector"
/*@
   EPSGetLeftInitialVector - Gets the left initial vector associated with the 
   eigensolver; if the vector was not set it will return a null pointer or
   a vector randomly generated by EPSSetUp().

   Not collective, but vector is shared by all processors that share the EPS

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: EPSSetLeftInitialVector(), EPSGetLeftInitialVector()

@*/
PetscErrorCode EPSGetLeftInitialVector(EPS eps,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidPointer(vec,2);
  *vec = eps->vec_initial_left;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetOperators"
/*@
   EPSSetOperators - Sets the matrices associated with the eigenvalue problem.

   Collective on EPS and Mat

   Input Parameters:
+  eps - the eigenproblem solver context
.  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Notes: 
   To specify a standard eigenproblem, use PETSC_NULL for parameter B.

   Level: beginner

.seealso: EPSSolve(), EPSGetST(), STGetOperators()
@*/
PetscErrorCode EPSSetOperators(EPS eps,Mat A,Mat B)
{
  PetscErrorCode ierr;
  PetscInt       m,n,m0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  PetscValidHeaderSpecific(A,MAT_COOKIE,2);
  if (B) PetscValidHeaderSpecific(B,MAT_COOKIE,3);
  PetscCheckSameComm(eps,1,A,2);
  if (B) PetscCheckSameComm(eps,1,B,3);

  /* Check for square matrices */
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"A is a non-square matrix"); }
  if (B) { 
    ierr = MatGetSize(B,&m0,&n);CHKERRQ(ierr);
    if (m0!=n) { SETERRQ(1,"B is a non-square matrix"); }
    if (m!=m0) { SETERRQ(1,"Dimensions of A and B do not match"); }
  }

  ierr = STSetOperators(eps->OP,A,B);CHKERRQ(ierr);
  eps->setupcalled = 0;  /* so that next solve call will call setup */

  /* Destroy randomly generated initial vectors */
  if (eps->vec_initial) {
    ierr = VecDestroy(eps->vec_initial);CHKERRQ(ierr);
    eps->vec_initial = PETSC_NULL;
  }
  if (eps->vec_initial_left) {
    ierr = VecDestroy(eps->vec_initial_left);CHKERRQ(ierr);
    eps->vec_initial_left = PETSC_NULL;
  }
  if (eps->D) {
    ierr = VecDestroy(eps->D);CHKERRQ(ierr);
    eps->D = PETSC_NULL;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EPSGetOperators"
/*@
   EPSGetOperators - Gets the matrices associated with the eigensystem.

   Collective on EPS and Mat

   Input Parameter:
.  eps - the EPS context

   Output Parameters:
+  A  - the matrix associated with the eigensystem
-  B  - the second matrix in the case of generalized eigenproblems

   Level: intermediate

.seealso: EPSSolve(), EPSGetST(), STGetOperators(), STSetOperators()
@*/
PetscErrorCode EPSGetOperators(EPS eps, Mat *A, Mat *B)
{
  PetscErrorCode ierr;
  ST             st;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (A) PetscValidPointer(A,2);
  if (B) PetscValidPointer(B,3);
  ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
  ierr = STGetOperators(st,A,B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetDeflationSpace"
/*@
   EPSSetDeflationSpace - Specify a basis of vectors that constitute
   the deflation space.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  ds    - set of basis vectors of the deflation space

   Notes:
   When a deflation space is given, the eigensolver seeks the eigensolution
   in the restriction of the problem to the orthogonal complement of this
   space. This can be used for instance in the case that an invariant 
   subspace is known beforehand (such as the nullspace of the matrix).

   Basis vectors set by a previous call to EPSSetDeflationSpace() are
   replaced.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   These vectors persist from one EPSSolve() call to the other, use
   EPSRemoveDeflationSpace() to eliminate them.

   Level: intermediate

.seealso: EPSRemoveDeflationSpace()
@*/
PetscErrorCode EPSSetDeflationSpace(EPS eps,PetscInt n,Vec *ds)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (n<=0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Argument n out of range"); 

  /* free previous vectors */
  ierr = EPSRemoveDeflationSpace(eps);CHKERRQ(ierr);

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&eps->DS);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)ds[i]);CHKERRQ(ierr);
    eps->DS[i] = ds[i];
  }

  eps->nds = n;
  eps->setupcalled = 0;
  eps->ds_ortho = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSRemoveDeflationSpace"
/*@
   EPSRemoveDeflationSpace - Removes the deflation space.

   Collective on EPS

   Input Parameter:
.  eps   - the eigenproblem solver context

   Level: intermediate

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode EPSRemoveDeflationSpace(EPS eps)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *pV;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (eps->nds > 0) {
    ierr = VecGetArray(eps->DS[0],&pV);CHKERRQ(ierr);
    ierr = VecRestoreArray(eps->DS[0],PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<eps->nds;i++) {
      ierr = VecDestroy(eps->DS[i]);CHKERRQ(ierr);
    }
    if (eps->setupcalled) {  /* before EPSSetUp, DS are just references */
      ierr = PetscFree(pV);CHKERRQ(ierr);
    }
    ierr = PetscFree(eps->DS);CHKERRQ(ierr);
  }
  eps->nds = 0;
  eps->setupcalled = 0;
  eps->ds_ortho = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSetInitialSpace"
/*@
   EPSSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on EPS and Vec

   Input Parameter:
+  eps   - the eigenproblem solver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   In contrast to EPSSetDeflationSpace(), these vectors do not persist from one
   EPSSolve() call to the other, so the initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Level: intermediate

.seealso: EPSSetDeflationSpace()
@*/
PetscErrorCode EPSSetInitialSpace(EPS eps,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_COOKIE,1);
  if (n<=0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Argument n out of range"); 

  /* free previous non-processed vectors */
  if (eps->nini<0) {
    for (i=0;i<-eps->nini;i++) {
      ierr = VecDestroy(eps->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(eps->IS);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&eps->IS);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    eps->IS[i] = is[i];
  }

  eps->nini = -n;
  eps->setupcalled = 0;
  PetscFunctionReturn(0);
}


