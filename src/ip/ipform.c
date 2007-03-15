/*
     Routines for setting the bilinear form
*/
#include "src/ip/ipimpl.h"      /*I "slepcip.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "IPSetBilinearForm"
PetscErrorCode IPSetBilinearForm(IP ip,Mat mat,IPBilinearForm form)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  if (mat) {
    PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
    PetscObjectReference((PetscObject)mat);
  }
  if (ip->matrix) {
    ierr = MatDestroy(ip->matrix);CHKERRQ(ierr);
    ierr = VecDestroy(ip->work);CHKERRQ(ierr);
  }
  ip->matrix = mat;
  ip->bilinear_form = form;
  if (mat) { ierr = MatGetVecs(mat,&ip->work,PETSC_NULL);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPGetBilinearForm"
PetscErrorCode IPGetBilinearForm(IP ip,Mat* mat,IPBilinearForm* form)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  if (mat) *mat = ip->matrix;
  if (form) *form = ip->bilinear_form;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPApplyMatrix"
PetscErrorCode IPApplyMatrix(IP ip,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  if (ip->matrix) {
    ierr = MatMult(ip->matrix,x,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);  
}
