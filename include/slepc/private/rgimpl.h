/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(_RGIMPL)
#define _RGIMPL

#include <slepcrg.h>
#include <slepc/private/slepcimpl.h>

PETSC_EXTERN PetscBool RGRegisterAllCalled;
PETSC_EXTERN PetscErrorCode RGRegisterAll(void);

typedef struct _RGOps *RGOps;

struct _RGOps {
  PetscErrorCode (*istrivial)(RG,PetscBool*);
  PetscErrorCode (*computecontour)(RG,PetscInt,PetscScalar*,PetscScalar*);
  PetscErrorCode (*computebbox)(RG,PetscReal*,PetscReal*,PetscReal*,PetscReal*);
  PetscErrorCode (*checkinside)(RG,PetscReal,PetscReal,PetscInt*);
  PetscErrorCode (*setfromoptions)(PetscOptionItems*,RG);
  PetscErrorCode (*view)(RG,PetscViewer);
  PetscErrorCode (*destroy)(RG);
};

struct _p_RG {
  PETSCHEADER(struct _RGOps);
  PetscBool   complement;    /* region is the complement of the specified one */
  PetscReal   sfactor;       /* scaling factor */
  PetscReal   osfactor;      /* old scaling factor, before RGPushScale */
  void        *data;
};

/* show an inf instead of PETSC_MAX_REAL */
#define RGShowReal(r) (double)((PetscAbsReal(r)>=PETSC_MAX_REAL)?10*(r):(r))

#endif
