/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#ifndef _IPIMPL
#define _IPIMPL

#include "slepcip.h"

extern PetscCookie IP_COOKIE;
extern PetscLogEvent IP_InnerProduct,IP_Orthogonalize,IP_ApplyMatrix;

struct _p_IP {
  PETSCHEADER(int);
  IPOrthogonalizationType orthog_type; /* which orthogonalization to use */
  IPOrthogonalizationRefinementType orthog_ref;   /* refinement method */
  PetscReal orthog_eta;
  IPBilinearForm bilinear_form;
  Mat matrix;
  PetscInt innerproducts;

  /*------------------------- Cache Bx product -------------------*/
  PetscInt       xid;
  PetscInt       xstate;
  Vec            Bx;
};

EXTERN PetscErrorCode IPApplyMatrix_Private(IP,Vec);

#endif
