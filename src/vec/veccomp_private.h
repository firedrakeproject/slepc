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

#ifndef _VECCOMP_P_
#define _VECCOMP_P_

EXTERN_C_BEGIN
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Comp(Vec V);
EXTERN_C_END
PetscErrorCode VecDuplicate_Comp(Vec win, Vec *V);
PetscErrorCode VecDestroy_Comp(Vec v);
PetscErrorCode VecSet_Comp(Vec v, PetscScalar alpha);
PetscErrorCode VecView_Comp(Vec v, PetscViewer viewer);
PetscErrorCode VecScale_Comp(Vec v, PetscScalar alpha);
PetscErrorCode VecCopy_Comp(Vec v, Vec w);
PetscErrorCode VecSwap_Comp(Vec v, Vec w);
PetscErrorCode VecAXPY_Comp(Vec v, PetscScalar alpha, Vec w);
PetscErrorCode VecAXPBY_Comp(Vec v, PetscScalar alpha, PetscScalar beta, Vec w);
PetscErrorCode VecMAXPY_Comp(Vec v, PetscInt n, const PetscScalar *alpha,
                            Vec *w);
PetscErrorCode VecWAXPY_Comp(Vec v, PetscScalar alpha, Vec w, Vec z);
PetscErrorCode VecAXPBYPCZ_Comp(Vec v, PetscScalar alpha, PetscScalar beta,
                                PetscScalar gamma, Vec w, Vec z);
PetscErrorCode VecPointwiseMult_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecPointwiseDivide_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecGetSize_Comp(Vec v, PetscInt *size);
PetscErrorCode VecGetLocalSize_Comp(Vec v, PetscInt *size);
PetscErrorCode VecMax_Comp(Vec v, PetscInt *idx, PetscReal *z);
PetscErrorCode VecMin_Comp(Vec v, PetscInt *idx, PetscReal *z);
PetscErrorCode VecSetRandom_Comp(Vec v, PetscRandom r);
PetscErrorCode VecConjugate_Comp(Vec v);
PetscErrorCode VecReciprocal_Comp(Vec v);
PetscErrorCode VecMaxPointwiseDivide_Comp(Vec v, Vec w, PetscReal *m);
PetscErrorCode VecPointwiseMax_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecPointwiseMaxAbs_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecPointwiseMin_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecDotNorm2_Comp_Seq(Vec v, Vec w, PetscScalar *dp,
                                    PetscScalar *nm);
PetscErrorCode VecDotNorm2_Comp_MPI(Vec v, Vec w, PetscScalar *dp,
                                    PetscScalar *nm);
PetscErrorCode VecSqrt_Comp(Vec v);
PetscErrorCode VecAbs_Comp(Vec v);
PetscErrorCode VecExp_Comp(Vec v);
PetscErrorCode VecLog_Comp(Vec v);
PetscErrorCode VecShift_Comp(Vec v, PetscScalar alpha);

#endif
