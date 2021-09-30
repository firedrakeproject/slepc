/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepccontour.h>
#include <slepcblaslapack.h>

/*
   SlepcContourDataCreate - Create a contour data structure.

   Input Parameters:
   n - the number of integration points
   npart - number of partitions for the subcommunicator
   parent - parent object
*/
PetscErrorCode SlepcContourDataCreate(PetscInt n,PetscInt npart,PetscObject parent,SlepcContourData *contour)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(contour);CHKERRQ(ierr);
  (*contour)->parent = parent;
  ierr = PetscSubcommCreate(PetscObjectComm(parent),&(*contour)->subcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber((*contour)->subcomm,npart);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = PetscSubcommSetType((*contour)->subcomm,PETSC_SUBCOMM_INTERLACED);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(parent,sizeof(PetscSubcomm));CHKERRQ(ierr);
  (*contour)->npoints = n / npart;
  if (n%npart > (*contour)->subcomm->color) (*contour)->npoints++;
  PetscFunctionReturn(0);
}

/*
   SlepcContourDataReset - Resets the KSP objects in a contour data structure,
   and destroys any objects whose size depends on the problem size.
*/
PetscErrorCode SlepcContourDataReset(SlepcContourData contour)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (contour->ksp) {
    for (i=0;i<contour->npoints;i++) {
      ierr = KSPReset(contour->ksp[i]);CHKERRQ(ierr);
    }
  }
  if (contour->pA) {
    for (i=0;i<contour->nmat;i++) {
      ierr = MatDestroy(&contour->pA[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(contour->pA);CHKERRQ(ierr);
    contour->pA = NULL;
    contour->nmat = 0;
  }
  ierr = VecScatterDestroy(&contour->scatterin);CHKERRQ(ierr);
  ierr = VecDestroy(&contour->xsub);CHKERRQ(ierr);
  ierr = VecDestroy(&contour->xdup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SlepcContourDataDestroy - Destroys the contour data structure.
*/
PetscErrorCode SlepcContourDataDestroy(SlepcContourData *contour)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (!(*contour)) PetscFunctionReturn(0);
  if ((*contour)->ksp) {
    for (i=0;i<(*contour)->npoints;i++) {
      ierr = KSPDestroy(&(*contour)->ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*contour)->ksp);CHKERRQ(ierr);
  }
  ierr = PetscSubcommDestroy(&(*contour)->subcomm);CHKERRQ(ierr);
  ierr = PetscFree((*contour));CHKERRQ(ierr);
  *contour = NULL;
  PetscFunctionReturn(0);
}

/*
   SlepcContourRedundantMat - Creates redundant copies of the passed matrices in the subcomm.

   Input Parameters:
   nmat - the number of matrices
   A    - array of matrices
*/
PetscErrorCode SlepcContourRedundantMat(SlepcContourData contour,PetscInt nmat,Mat *A)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (contour->pA) {
    for (i=0;i<contour->nmat;i++) {
      ierr = MatDestroy(&contour->pA[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(contour->pA);CHKERRQ(ierr);
    contour->pA = NULL;
    contour->nmat = 0;
  }
  if (contour->subcomm && contour->subcomm->n != 1) {
    ierr = PetscCalloc1(nmat,&contour->pA);CHKERRQ(ierr);
    for (i=0;i<nmat;i++) {
      ierr = MatCreateRedundantMatrix(A[i],contour->subcomm->n,PetscSubcommChild(contour->subcomm),MAT_INITIAL_MATRIX,&contour->pA[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(contour->parent,(PetscObject)contour->pA[i]);CHKERRQ(ierr);
    }
    contour->nmat = nmat;
  }
  PetscFunctionReturn(0);
}

/*
   SlepcContourScatterCreate - Creates a scatter context to communicate between a
   regular vector and a vector xdup that can hold one duplicate per each subcommunicator
   on the contiguous parent communicator. Also creates auxiliary vectors xdup and xsub
   (the latter with the same layout as the redundant matrices in the subcommunicator).

   Input Parameters:
   v - the regular vector from which dimensions are taken
*/
PetscErrorCode SlepcContourScatterCreate(SlepcContourData contour,Vec v)
{
  PetscErrorCode ierr;
  IS             is1,is2;
  PetscInt       i,j,k,m,mstart,mend,mlocal;
  PetscInt       *idx1,*idx2,mloc_sub;

  PetscFunctionBegin;
  ierr = VecDestroy(&contour->xsub);CHKERRQ(ierr);
  ierr = MatCreateVecsEmpty(contour->pA[0],&contour->xsub,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&contour->xdup);CHKERRQ(ierr);
  ierr = MatGetLocalSize(contour->pA[0],&mloc_sub,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PetscSubcommContiguousParent(contour->subcomm),&contour->xdup);CHKERRQ(ierr);
  ierr = VecSetSizes(contour->xdup,mloc_sub,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(contour->xdup,((PetscObject)v)->type_name);CHKERRQ(ierr);

  ierr = VecScatterDestroy(&contour->scatterin);CHKERRQ(ierr);
  ierr = VecGetSize(v,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(v,&mstart,&mend);CHKERRQ(ierr);
  mlocal = mend - mstart;
  ierr = PetscMalloc2(contour->subcomm->n*mlocal,&idx1,contour->subcomm->n*mlocal,&idx2);CHKERRQ(ierr);
  j = 0;
  for (k=0;k<contour->subcomm->n;k++) {
    for (i=mstart;i<mend;i++) {
      idx1[j]   = i;
      idx2[j++] = i + m*k;
    }
  }
  ierr = ISCreateGeneral(PetscSubcommParent(contour->subcomm),contour->subcomm->n*mlocal,idx1,PETSC_COPY_VALUES,&is1);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscSubcommParent(contour->subcomm),contour->subcomm->n*mlocal,idx2,PETSC_COPY_VALUES,&is2);CHKERRQ(ierr);
  ierr = VecScatterCreate(v,is1,contour->xdup,is2,&contour->scatterin);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);
  ierr = PetscFree2(idx1,idx2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SlepcCISS_isGhost - Determine if any of the computed eigenpairs are spurious.

   Input Parameters:
   X - the matrix of eigenvectors (MATSEQDENSE)
   n - the number of columns to consider
   sigma - the singular values
   thresh - threshold to decide whether a value is spurious

   Output Parameter:
   fl - array of n booleans
*/
PetscErrorCode SlepcCISS_isGhost(Mat X,PetscInt n,PetscReal *sigma,PetscReal thresh,PetscBool *fl)
{
  PetscErrorCode    ierr;
  const PetscScalar *pX;
  PetscInt          i,j,m,ld;
  PetscReal         *tau,s1,s2,tau_max=0.0;

  PetscFunctionBegin;
  ierr = MatGetSize(X,&m,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(X,&ld);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&tau);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(X,&pX);CHKERRQ(ierr);
  for (j=0;j<n;j++) {
    s1 = 0.0;
    s2 = 0.0;
    for (i=0;i<m;i++) {
      s1 += PetscAbsScalar(PetscPowScalarInt(pX[i+j*ld],2));
      s2 += PetscPowRealInt(PetscAbsScalar(pX[i+j*ld]),2)/sigma[i];
    }
    tau[j] = s1/s2;
    tau_max = PetscMax(tau_max,tau[j]);
  }
  ierr = MatDenseRestoreArrayRead(X,&pX);CHKERRQ(ierr);
  for (j=0;j<n;j++) fl[j] = (tau[j]>=thresh*tau_max)? PETSC_TRUE: PETSC_FALSE;
  ierr = PetscFree(tau);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SlepcCISS_BH_SVD - Compute SVD of block Hankel matrix and its rank.

   Input Parameters:
   H  - block Hankel matrix obtained via CISS_BlockHankel()
   ml - dimension of rows and columns, equal to M*L
   delta - the tolerance used to determine the rank

   Output Parameters:
   sigma - computed singular values
   rank  - the rank of H
*/
PetscErrorCode SlepcCISS_BH_SVD(PetscScalar *H,PetscInt ml,PetscReal delta,PetscReal *sigma,PetscInt *rank)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBLASInt   m,n,lda,ldu,ldvt,lwork,info;
  PetscScalar    *work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork;
#endif

  PetscFunctionBegin;
  ierr = PetscMalloc1(5*ml,&work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(5*ml,&rwork);CHKERRQ(ierr);
#endif
  ierr = PetscBLASIntCast(ml,&m);CHKERRQ(ierr);
  n = m; lda = m; ldu = m; ldvt = m; lwork = 5*m;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&m,&n,H,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,rwork,&info));
#else
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("N","N",&m,&n,H,&lda,sigma,NULL,&ldu,NULL,&ldvt,work,&lwork,&info));
#endif
  SlepcCheckLapackInfo("gesvd",info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  (*rank) = 0;
  for (i=0;i<ml;i++) {
    if (sigma[i]/PetscMax(sigma[0],1.0)>delta) (*rank)++;
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

