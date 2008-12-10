/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#ifndef _STIMPL
#define _STIMPL

#include "slepceps.h"

extern PetscLogEvent ST_SetUp, ST_Apply, ST_ApplyB, ST_ApplyTranspose;
extern PetscFList STList;

typedef struct _STOps *STOps;

struct _STOps {
  PetscErrorCode   (*setup)(ST);
  PetscErrorCode   (*apply)(ST,Vec,Vec);
  PetscErrorCode   (*getbilinearform)(ST,Mat*);
  PetscErrorCode   (*applytrans)(ST,Vec,Vec);
  PetscErrorCode   (*setshift)(ST,PetscScalar);
  PetscErrorCode   (*setfromoptions)(ST);
  PetscErrorCode   (*postsolve)(ST);  
  PetscErrorCode   (*backtr)(ST,PetscScalar*,PetscScalar*);  
  PetscErrorCode   (*destroy)(ST);
  PetscErrorCode   (*view)(ST,PetscViewer);
};

struct _p_ST {
  PETSCHEADER(struct _STOps);
  /*------------------------- User parameters --------------------------*/
  Mat            A,B;              /* Matrices which define the eigensystem */
  PetscScalar    sigma;            /* Value of the shift */
  STMatMode      shift_matrix;
  MatStructure   str;          /* whether matrices have the same pattern or not */
  Mat            mat;

  /*------------------------- Misc data --------------------------*/
  KSP            ksp;
  Vec            w;
  void           *data;
  PetscInt       setupcalled;
  PetscInt       lineariterations;
  PetscInt       applys;
  PetscErrorCode (*checknullspace)(ST,PetscInt,const Vec[]);
  
};

EXTERN PetscErrorCode STRegisterAll(char*);

EXTERN PetscErrorCode STGetBilinearForm_Default(ST,Mat*);
EXTERN PetscErrorCode STView_Default(ST,PetscViewer);
EXTERN PetscErrorCode STAssociatedKSPSolve(ST,Vec,Vec);
EXTERN PetscErrorCode STAssociatedKSPSolveTranspose(ST,Vec,Vec);
EXTERN PetscErrorCode STCheckNullSpace_Default(ST,PetscInt,const Vec[]);
EXTERN PetscErrorCode STMatShellCreate(ST st,Mat *mat);

#endif

