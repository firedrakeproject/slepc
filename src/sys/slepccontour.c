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
   SlepcContourDataReset - Resets the KSP objects in a contour data structure.
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

