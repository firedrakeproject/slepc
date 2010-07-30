/*
      QEP routines related to problem setup.

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

#include "private/qepimpl.h"   /*I "slepcqep.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "QEPSetUp"
/*@
   QEPSetUp - Sets up all the internal data structures necessary for the
   execution of the QEP solver.

   Collective on QEP

   Input Parameter:
.  qep   - solver context

   Notes:
   This function need not be called explicitly in most cases, since QEPSolve()
   calls it. It can be useful when one wants to measure the set-up time 
   separately from the solve time.

   Level: advanced

.seealso: QEPCreate(), QEPSolve(), QEPDestroy()
@*/
PetscErrorCode QEPSetUp(QEP qep)
{
  PetscErrorCode ierr;
  PetscInt       i,k;
  PetscScalar    *pV;
  PetscTruth     khas,mhas,lindep;
  PetscReal      knorm,mnorm,norm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);

  if (qep->setupcalled) PetscFunctionReturn(0);

  ierr = PetscLogEventBegin(QEP_SetUp,qep,0,0,0);CHKERRQ(ierr);

  /* Set default solver type */
  if (!((PetscObject)qep)->type_name) {
    ierr = QEPSetType(qep,QEPLINEAR);CHKERRQ(ierr);
  }

  /* Check matrices */
  if (!qep->M || !qep->C || !qep->K)
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "QEPSetOperators must be called first"); 
  
  /* Set problem dimensions */
  ierr = MatGetSize(qep->M,&qep->n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetLocalSize(qep->M,&qep->nloc,PETSC_NULL);CHKERRQ(ierr);

  /* Set default problem type */
  if (!qep->problem_type) {
    ierr = QEPSetProblemType(qep,QEP_GENERAL);CHKERRQ(ierr);
  }

  /* Compute scaling factor if not set by user */
  if (qep->sfactor==0.0) {
    ierr = MatHasOperation(qep->K,MATOP_NORM,&khas);CHKERRQ(ierr);
    ierr = MatHasOperation(qep->M,MATOP_NORM,&mhas);CHKERRQ(ierr);
    if (khas && mhas) {
      ierr = MatNorm(qep->K,NORM_INFINITY,&knorm);CHKERRQ(ierr);
      ierr = MatNorm(qep->M,NORM_INFINITY,&mnorm);CHKERRQ(ierr);
      qep->sfactor = sqrt(knorm/mnorm);
    }
    else qep->sfactor = 1.0;
  }

  /* initialize the random number generator */
  ierr = PetscRandomCreate(((PetscObject)qep)->comm,&qep->rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(qep->rand);CHKERRQ(ierr);

  /* Call specific solver setup */
  ierr = (*qep->ops->setup)(qep);CHKERRQ(ierr);

  if (qep->ncv > 2*qep->n)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"ncv must be twice the problem size at most");
  if (qep->nev > qep->ncv)
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"nev bigger than ncv");

  /* Free memory for previous solution  */
  if (qep->eigr) { 
    ierr = PetscFree(qep->eigr);CHKERRQ(ierr);
    ierr = PetscFree(qep->eigi);CHKERRQ(ierr);
    ierr = PetscFree(qep->perm);CHKERRQ(ierr);
    ierr = PetscFree(qep->errest);CHKERRQ(ierr);
    ierr = VecGetArray(qep->V[0],&pV);CHKERRQ(ierr);
    for (i=0;i<qep->ncv;i++) {
      ierr = VecDestroy(qep->V[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(pV);CHKERRQ(ierr);
    ierr = PetscFree(qep->V);CHKERRQ(ierr);
  }

  /* Allocate memory for next solution */
  ierr = PetscMalloc(qep->ncv*sizeof(PetscScalar),&qep->eigr);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*sizeof(PetscScalar),&qep->eigi);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*sizeof(PetscInt),&qep->perm);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*sizeof(PetscReal),&qep->errest);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*sizeof(Vec),&qep->V);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*qep->nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
  for (i=0;i<qep->ncv;i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,qep->nloc,PETSC_DECIDE,pV+i*qep->nloc,&qep->V[i]);CHKERRQ(ierr);
  }

  /* process initial vectors */
  if (qep->nini<0) {
    qep->nini = -qep->nini;
    if (qep->nini>qep->ncv) SETERRQ(1,"The number of initial vectors is larger than ncv")
    k = 0;
    for (i=0;i<qep->nini;i++) {
      ierr = VecCopy(qep->IS[i],qep->V[k]);CHKERRQ(ierr);
      ierr = VecDestroy(qep->IS[i]);CHKERRQ(ierr);
      ierr = IPOrthogonalize(qep->ip,0,PETSC_NULL,k,PETSC_NULL,qep->V,qep->V[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
      if (norm==0.0 || lindep) PetscInfo(qep,"Linearly dependent initial vector found, removing...\n");
      else {
        ierr = VecScale(qep->V[k],1.0/norm);CHKERRQ(ierr);
        k++;
      }
    }
    qep->nini = k;
    ierr = PetscFree(qep->IS);CHKERRQ(ierr);
  }
  if (qep->ninil<0) {
    if (!qep->leftvecs) PetscInfo(qep,"Ignoring initial left vectors\n");
    else {
      qep->ninil = -qep->ninil;
      if (qep->ninil>qep->ncv) SETERRQ(1,"The number of initial left vectors is larger than ncv")
      k = 0;
      for (i=0;i<qep->ninil;i++) {
        ierr = VecCopy(qep->ISL[i],qep->W[k]);CHKERRQ(ierr);
        ierr = VecDestroy(qep->ISL[i]);CHKERRQ(ierr);
        ierr = IPOrthogonalize(qep->ip,0,PETSC_NULL,k,PETSC_NULL,qep->W,qep->W[k],PETSC_NULL,&norm,&lindep);CHKERRQ(ierr); 
        if (norm==0.0 || lindep) PetscInfo(qep,"Linearly dependent initial left vector found, removing...\n");
        else {
          ierr = VecScale(qep->W[k],1.0/norm);CHKERRQ(ierr);
          k++;
        }
      }
      qep->ninil = k;
      ierr = PetscFree(qep->ISL);CHKERRQ(ierr);
    }
  }

  ierr = PetscLogEventEnd(QEP_SetUp,qep,0,0,0);CHKERRQ(ierr);
  qep->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetOperators"
/*@
   QEPSetOperators - Sets the matrices associated with the quadratic eigenvalue problem.

   Collective on QEP and Mat

   Input Parameters:
+  qep - the eigenproblem solver context
.  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Notes: 
   The quadratic eigenproblem is defined as (l^2*M + l*C + K)*x = 0, where l is
   the eigenvalue and x is the eigenvector.

   Level: beginner

.seealso: QEPSolve(), QEPGetOperators()
@*/
PetscErrorCode QEPSetOperators(QEP qep,Mat M,Mat C,Mat K)
{
  PetscErrorCode ierr;
  PetscInt       m,n,m0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidHeaderSpecific(M,MAT_COOKIE,2);
  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidHeaderSpecific(K,MAT_COOKIE,4);
  PetscCheckSameComm(qep,1,M,2);
  PetscCheckSameComm(qep,1,C,3);
  PetscCheckSameComm(qep,1,K,4);

  /* Check for square matrices */
  ierr = MatGetSize(M,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"M is a non-square matrix"); }
  m0=m;
  ierr = MatGetSize(C,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"C is a non-square matrix"); }
  if (m!=m0) { SETERRQ(1,"Dimensions of M and C do not match"); }
  ierr = MatGetSize(K,&m,&n);CHKERRQ(ierr);
  if (m!=n) { SETERRQ(1,"K is a non-square matrix"); }
  if (m!=m0) { SETERRQ(1,"Dimensions of M and K do not match"); }

  /* Store a copy of the matrices */
  ierr = PetscObjectReference((PetscObject)M);CHKERRQ(ierr);
  if (qep->M) {
    ierr = MatDestroy(qep->M);CHKERRQ(ierr);
  }
  qep->M = M;
  ierr = PetscObjectReference((PetscObject)C);CHKERRQ(ierr);
  if (qep->C) {
    ierr = MatDestroy(qep->C);CHKERRQ(ierr);
  }
  qep->C = C;
  ierr = PetscObjectReference((PetscObject)K);CHKERRQ(ierr);
  if (qep->K) {
    ierr = MatDestroy(qep->K);CHKERRQ(ierr);
  }
  qep->K = K;

  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPGetOperators"
/*@
   QEPGetOperators - Gets the matrices associated with the quadratic eigensystem.

   Collective on QEP and Mat

   Input Parameter:
.  qep - the QEP context

   Output Parameters:
+  M   - the first coefficient matrix
.  C   - the second coefficient matrix
-  K   - the third coefficient matrix

   Level: intermediate

.seealso: QEPSolve(), QEPSetOperators()
@*/
PetscErrorCode QEPGetOperators(QEP qep, Mat *M, Mat *C,Mat *K)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (M) { PetscValidPointer(M,2); *M = qep->M; }
  if (C) { PetscValidPointer(C,3); *C = qep->C; }
  if (K) { PetscValidPointer(K,4); *K = qep->K; }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetInitialSpace"
/*@
   QEPSetInitialSpace - Specify a basis of vectors that constitute the initial
   space, that is, the subspace from which the solver starts to iterate.

   Collective on QEP and Vec

   Input Parameter:
+  qep   - the quadratic eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial space

   Notes:
   Some solvers start to iterate on a single vector (initial vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one QEPSolve() call to the other, so the
   initial space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: QEPSetInitialSpaceLeft()
@*/
PetscErrorCode QEPSetInitialSpace(QEP qep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (n<0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (qep->nini<0) {
    for (i=0;i<-qep->nini;i++) {
      ierr = VecDestroy(qep->IS[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(qep->IS);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&qep->IS);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    qep->IS[i] = is[i];
  }

  qep->nini = -n;
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetInitialSpaceLeft"
/*@
   QEPSetInitialSpaceLeft - Specify a basis of vectors that constitute the initial
   left space, that is, the subspace from which the solver starts to iterate for
   building the left subspace (in methods that work with two subspaces).

   Collective on QEP and Vec

   Input Parameter:
+  qep   - the quadratic eigensolver context
.  n     - number of vectors
-  is    - set of basis vectors of the initial left space

   Notes:
   Some solvers start to iterate on a single vector (initial left vector). In that case,
   the other vectors are ignored.

   These vectors do not persist from one QEPSolve() call to the other, so the
   initial left space should be set every time.

   The vectors do not need to be mutually orthonormal, since they are explicitly
   orthonormalized internally.

   Common usage of this function is when the user can provide a rough approximation
   of the wanted left eigenspace. Then, convergence may be faster.

   Level: intermediate

.seealso: QEPSetInitialSpace()
@*/
PetscErrorCode QEPSetInitialSpaceLeft(QEP qep,PetscInt n,Vec *is)
{
  PetscErrorCode ierr;
  PetscInt       i;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  if (n<0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Argument n cannot be negative"); 

  /* free previous non-processed vectors */
  if (qep->ninil<0) {
    for (i=0;i<-qep->ninil;i++) {
      ierr = VecDestroy(qep->ISL[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(qep->ISL);CHKERRQ(ierr);
  }

  /* get references of passed vectors */
  ierr = PetscMalloc(n*sizeof(Vec),&qep->ISL);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
    qep->ISL[i] = is[i];
  }

  qep->ninil = -n;
  qep->setupcalled = 0;
  PetscFunctionReturn(0);
}

