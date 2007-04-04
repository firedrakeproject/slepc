/*
     Routines for setting the bilinear form
*/
#include "src/ip/ipimpl.h"      /*I "slepcip.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "IPSetBilinearForm"
/*@
   IPSetBilinearForm - Specifies the bilinear form to be used for
   inner products.

   Collective on IP

   Input Parameters:
+  ip    - the inner product context
.  mat   - the matrix of the bilinear form (may be PETSC_NULL)
-  form  - the type of bilinear form

   Note:
   This function is called by EPSSetProblemType() and usually need not be
   called by the user.

   Level: developer

.seealso: IPGetBilinearForm(), IPInnerProduct(), IPNorm(), EPSSetProblemType() 
@*/
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
/*@C
   IPGetBilinearForm - Retrieves the bilinear form to be used for
   inner products.

   Input Parameter:
.  ip    - the inner product context

   Output Parameter:
+  mat   - the matrix of the bilinear form (may be PETSC_NULL)
-  form  - the type of bilinear form

   Level: developer

.seealso: IPSetBilinearForm(), IPInnerProduct(), IPNorm(), EPSSetProblemType()
@*/
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
/*@
   IPApplyMatrix - Multiplies a vector with the matrix associated to the
                   bilinear form.

   Collective on IP

   Input Parameters:
+  ip    - the inner product context
-  x     - the vector

   Output Parameter:
.  y     - the result  

   Note:
   If the bilinear form has no associated matrix this function copies the vector.

   Level: developer

.seealso: IPSetBilinearForm(), IPInnerProduct(), IPNorm(), EPSSetProblemType() 
@*/
PetscErrorCode IPApplyMatrix(IP ip,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  ierr = PetscLogEventBegin(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
  if (ip->matrix) {
    ierr = MatMult(ip->matrix,x,y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(IP_ApplyMatrix,ip,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
