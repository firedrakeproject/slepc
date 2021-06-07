/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepccontour.h>

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
    }
    contour->nmat = nmat;
  }
  PetscFunctionReturn(0);
}

/*
   SlepcContourScatterCreate - Creates a scatter context to communicate between a
   regular vector and a vector xdup that has similar layout but in the contiguous
   communicator. Also creates auxiliary vectors xdup and xsub (with the same layout
   as the redundant matrices).

   Input Parameters:
   v0 - the regular vector from which dimensions are taken
*/
PetscErrorCode SlepcContourScatterCreate(SlepcContourData contour,Vec v)
{
  PetscErrorCode ierr;
  IS             is1,is2;
  PetscInt       i,j,k,m,mstart,mend,mlocal;
  PetscInt       *idx1,*idx2,mloc_sub;

  PetscFunctionBegin;
  ierr = VecDestroy(&contour->xsub);CHKERRQ(ierr);
  ierr = MatCreateVecs(contour->pA[0],&contour->xsub,NULL);CHKERRQ(ierr);

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

