/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#ifndef _IPIMPL
#define _IPIMPL

#include <slepcip.h>

extern PetscLogEvent IP_InnerProduct,IP_Orthogonalize,IP_ApplyMatrix;

struct _p_IP {
  PETSCHEADER(int);
  IPOrthogType       orthog_type;    /* which orthogonalization to use */
  IPOrthogRefineType orthog_ref;     /* refinement method */
  PetscReal          orthog_eta;     /* refinement threshold */
  IPBilinearForm     bilinear_form;
  Mat                matrix;
  PetscInt           innerproducts;

  /*------------------------- Cache Bx product -------------------*/
  PetscInt           xid;
  PetscInt           xstate;
  Vec                Bx;
};

extern PetscErrorCode IPInitializePackage(const char *);
extern PetscErrorCode IPApplyMatrix_Private(IP,Vec);
extern PetscErrorCode IPOrthogonalizeCGS1(IP,PetscInt,Vec*,PetscInt,PetscBool*,Vec*,Vec,PetscScalar*,PetscReal*,PetscReal*);

#endif
