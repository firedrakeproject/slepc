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
  PetscInt       i,N,nloc;
  PetscScalar    *pV;
  
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
  
  /* Set default problem type */
  if (!qep->problem_type) {
    ierr = QEPSetProblemType(qep,QEP_GENERAL);CHKERRQ(ierr);
  }

  /* Create random initial vector if not set */
  if (!qep->vec_initial) {
    ierr = MatGetVecs(qep->M,&qep->vec_initial,PETSC_NULL);CHKERRQ(ierr);
    ierr = SlepcVecSetRandom(qep->vec_initial);CHKERRQ(ierr);
  }

  /* Call specific solver setup */
  ierr = (*qep->ops->setup)(qep);CHKERRQ(ierr);

  ierr = VecGetSize(qep->vec_initial,&N);CHKERRQ(ierr);
  if (qep->ncv > 2*N)
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
  ierr = VecGetLocalSize(qep->vec_initial,&nloc);CHKERRQ(ierr);
  ierr = PetscMalloc(qep->ncv*nloc*sizeof(PetscScalar),&pV);CHKERRQ(ierr);
  for (i=0;i<qep->ncv;i++) {
    ierr = VecCreateMPIWithArray(((PetscObject)qep)->comm,nloc,PETSC_DECIDE,pV+i*nloc,&qep->V[i]);CHKERRQ(ierr);
  }

  ierr = PetscLogEventEnd(QEP_SetUp,qep,0,0,0);CHKERRQ(ierr);
  qep->setupcalled = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPSetInitialVector"
/*@
   QEPSetInitialVector - Sets the initial vector from which the quadratic
   eigensolver starts to iterate.

   Collective on QEP and Vec

   Input Parameters:
+  qep - the eigensolver context
-  vec - the vector

   Level: intermediate

.seealso: QEPGetInitialVector()

@*/
PetscErrorCode QEPSetInitialVector(QEP qep,Vec vec)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidHeaderSpecific(vec,VEC_COOKIE,2);
  PetscCheckSameComm(qep,1,vec,2);
  ierr = PetscObjectReference((PetscObject)vec);CHKERRQ(ierr);
  if (qep->vec_initial) {
    ierr = VecDestroy(qep->vec_initial);CHKERRQ(ierr);
  }
  qep->vec_initial = vec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "QEPGetInitialVector"
/*@
   QEPGetInitialVector - Gets the initial vector associated with the quadratic
   eigensolver; if the vector was not set it will return a null pointer or
   a vector randomly generated by QEPSetUp().

   Not collective, but vector is shared by all processors that share the QEP

   Input Parameter:
.  qep - the eigensolver context

   Output Parameter:
.  vec - the vector

   Level: intermediate

.seealso: QEPSetInitialVector()

@*/
PetscErrorCode QEPGetInitialVector(QEP qep,Vec *vec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(qep,QEP_COOKIE,1);
  PetscValidPointer(vec,2);
  *vec = qep->vec_initial;
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

  /* Destroy randomly generated initial vector */
  if (qep->vec_initial) {
    ierr = VecDestroy(qep->vec_initial);CHKERRQ(ierr);
    qep->vec_initial = PETSC_NULL;
  }

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

