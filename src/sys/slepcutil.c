
#include "slepc.h" /*I "slepc.h" I*/
#include <stdlib.h>

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecSetRandom"
/*@
   SlepcVecSetRandom - Sets all components of a vector to random numbers which
   follow a uniform distribution in [0,1).

   Collective on Vec

   Input/Output Parameter:
.  x  - the vector

   Note:
   This operation is equivalent to VecSetRandom - the difference is that the
   vector generated by SlepcVecSetRandom is the same irrespective of the size
   of the communicator.

   Level: developer

.seealso: VecSetRandom()
@*/
PetscErrorCode SlepcVecSetRandom(Vec x)
{
  PetscErrorCode ierr;
  int            i,n,low,high;
  PetscScalar    *px,t;
  static unsigned short seed[3] = { 1, 3, 2 };
  
  PetscFunctionBegin;
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
    t = erand48(seed);
    if (i>=low && i<high) px[i-low] = t;
  }
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcIsHermitian"
/*@
   SlepcIsHermitian - Checks if a matrix is Hermitian or not.

   Collective on Mat

   Input parameter:
.  A  - the matrix

   Output parameter:
.  is  - flag indicating if the matrix is Hermitian

   Notes: 
   The result of Ax and A^Hx (with a random x) is compared, but they 
   could be equal also for some non-Hermitian matrices.

   This routine will not work with BOPT=O_complex and matrix formats
   MATSEQSBAIJ or MATMPISBAIJ.
   
   Level: developer

@*/
PetscErrorCode SlepcIsHermitian(Mat A,PetscTruth *is)
{
  PetscErrorCode ierr;
  int            M,N,m,n;
  Vec            x,w1,w2;
  PetscScalar    alpha;
  MPI_Comm       comm;
  PetscReal      norm;
  PetscTruth     has;

  PetscFunctionBegin;

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
  ierr = PetscTypeCompare((PetscObject)A,MATMPISBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
#endif

  *is = PETSC_FALSE;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (M!=N) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = SlepcVecSetRandom(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w2);CHKERRQ(ierr);
  ierr = MatMult(A,x,w1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,w2);CHKERRQ(ierr);
  ierr = VecConjugate(w2);CHKERRQ(ierr);
  alpha = -1.0;
  ierr = VecAXPY(&alpha,w1,w2);CHKERRQ(ierr);
  ierr = VecNorm(w2,NORM_2,&norm);CHKERRQ(ierr);
  if (norm<1.0e-6) *is = PETSC_TRUE;
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(w1);CHKERRQ(ierr);
  ierr = VecDestroy(w2);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)

#undef __FUNCT__  
#define __FUNCT__ "SlepcAbsEigenvalue"
/*@C
   SlepcAbsEigenvalue - Computes the absolute value of a complex number given
   its real and imaginary parts.

   Not collective

   Input parameters:
+  x  - the real part of the complex number
-  y  - the imaginary part of the complex number

   Return value:
.  the absolute value of the number

   Notes: 
   This function computes sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Level: developer

@*/
PetscReal SlepcAbsEigenvalue(PetscScalar x,PetscScalar y)
{
  PetscReal xabs,yabs,w,z,t;
  PetscFunctionBegin;
  xabs = PetscAbsReal(x);
  yabs = PetscAbsReal(y);
  w = PetscMax(xabs,yabs);
  z = PetscMin(xabs,yabs);
  if (z == 0.0) PetscFunctionReturn(w);
  t = z/w;
  PetscFunctionReturn(w*sqrt(1.0+t*t));  
}

#endif
