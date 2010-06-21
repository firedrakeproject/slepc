
#ifndef _VECCOMP_P_
#define _VECCOMP_P_

PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Comp(Vec V);
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
PetscErrorCode VecMax_Comp(Vec v, PetscInt *idx, PetscReal *z);
PetscErrorCode VecMin_Comp(Vec v, PetscInt *idx, PetscReal *z);
PetscErrorCode VecSetRandom_Comp(Vec v, PetscRandom r);
PetscErrorCode VecConjugate_Comp(Vec v);
PetscErrorCode VecReciprocal_Comp(Vec v);
PetscErrorCode VecMaxPointwiseDivide_Comp(Vec v, Vec w, PetscReal *m);
PetscErrorCode VecPointwiseMax_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecPointwiseMaxAbs_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecPointwiseMin_Comp(Vec v, Vec w, Vec z);
PetscErrorCode VecSqrt_Comp(Vec v);
PetscErrorCode VecAbs_Comp(Vec v);
PetscErrorCode VecExp_Comp(Vec v);
PetscErrorCode VecLog_Comp(Vec v);
PetscErrorCode VecShift_Comp(Vec v, PetscScalar alpha);

#endif
