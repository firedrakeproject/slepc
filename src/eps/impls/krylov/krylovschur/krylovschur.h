/*                       

   Private header for Krylov-Schur.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#if !defined(__KRYLOVSCHUR_H)
#define __KRYLOVSCHUR_H

PETSC_INTERN PetscErrorCode EPSSolve_KrylovSchur_Default(EPS);
PETSC_INTERN PetscErrorCode EPSSolve_KrylovSchur_Symm(EPS);
PETSC_INTERN PetscErrorCode EPSSolve_KrylovSchur_Slice(EPS);
PETSC_INTERN PetscErrorCode EPSSolve_KrylovSchur_Indefinite(EPS);
PETSC_INTERN PetscErrorCode EPSGetArbitraryValues(EPS,PetscScalar*,PetscScalar*);

/* Structure characterizing a shift in spectrum slicing */
typedef struct _n_shift *shift;
struct _n_shift {
  PetscReal     value;
  PetscInt      inertia;
  PetscBool     comp[2];    /* Shows completion of subintervals (left and right) */
  shift         neighb[2];  /* Adjacent shifts */
  PetscInt      index;      /* Index in eig where found values are stored */
  PetscInt      neigs;      /* Number of values found */
  PetscReal     ext[2];     /* Limits for accepted values */ 
  PetscInt      nsch[2];    /* Number of missing values for each subinterval */
  PetscInt      nconv[2];   /* Converged on each side (accepted or not) */
};

/* Structure for storing the state of spectrum slicing */
struct _n_SR {
  PetscReal     int0,int1;  /* Extremes of the interval */
  PetscInt      dir;        /* Determines the order of values in eig (+1 incr, -1 decr) */
  PetscBool     hasEnd;     /* Tells whether the interval has an end */
  PetscInt      inertia0,inertia1;
  Vec           *V;
  PetscScalar   *eig,*eigi,*monit,*back;
  PetscReal     *errest;
  PetscInt      *perm;      /* Permutation for keeping the eigenvalues in order */
  PetscInt      numEigs;    /* Number of eigenvalues in the interval */
  PetscInt      indexEig;
  shift         sPres;      /* Present shift */
  shift         *pending;   /* Pending shifts array */
  PetscInt      nPend;      /* Number of pending shifts */
  PetscInt      maxPend;    /* Size of "pending" array */
  Vec           *VDef;      /* Vector for deflation */
  PetscInt      *idxDef;    /* For deflation */
  PetscInt      nMAXCompl;
  PetscInt      iterCompl;
  PetscInt      itsKs;      /* Krylovschur restarts */
  PetscInt      nleap;
  shift         s0;         /* Initial shift */
  PetscScalar   *S;         /* Matrix for projected problem */
  PetscInt      nS;
  PetscReal     beta;
  shift         sPrev;
};
typedef struct _n_SR  *SR;

typedef struct {
  PetscReal     keep;
  SR            sr;
} EPS_KRYLOVSCHUR;

#endif
