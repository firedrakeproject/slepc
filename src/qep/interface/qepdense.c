/*                       
     This file contains routines for handling small-size dense quadratic problems.
     All routines are based on calls to LAPACK routines. Matrices passed in
     as arguments are assumed to be square matrices stored in column-major 
     format with a leading dimension equal to the number of rows.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

#include "private/qepimpl.h" /*I "slepcqep.h" I*/
#include "slepcblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "QEPSortDenseSchur"
/*@
   QEPSortDenseSchur - Reorders the Schur decomposition computed by
   EPSDenseSchur().

   Not Collective

   Input Parameters:
+  qep - the qep solver context
.  n     - dimension of the matrix 
.  k     - first active column
-  ldt   - leading dimension of T

   Input/Output Parameters:
+  T  - the upper (quasi-)triangular matrix
.  Q  - the orthogonal matrix of Schur vectors
.  wr - pointer to the array to store the computed eigenvalues
-  wi - imaginary part of the eigenvalues (only when using real numbers)

   Notes:
   This function reorders the eigenvalues in wr,wi located in positions k
   to n according to the sort order specified in EPSetWhicheigenpairs. 
   The Schur decomposition Z*T*Z^T, is also reordered by means of rotations 
   so that eigenvalues in the diagonal blocks of T follow the same order.

   Both T and Q are overwritten.
   
   This routine uses LAPACK routines xTREXC.

   Level: developer

.seealso: EPSDenseHessenberg(), EPSDenseSchur(), EPSDenseTridiagonal()
@*/
PetscErrorCode QEPSortDenseSchur(QEP qep,PetscInt n_,PetscInt k,PetscScalar *T,PetscInt ldt_,PetscScalar *Q,PetscScalar *wr,PetscScalar *wi)
{
#if defined(SLEPC_MISSING_LAPACK_TREXC)
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"TREXC - Lapack routine is unavailable.");
#else
  PetscErrorCode ierr;
  PetscScalar    re,im;
  PetscInt       i,j,pos,result;
  PetscBLASInt   ifst,ilst,info,n,ldt;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar    *work;
#endif
  
  PetscFunctionBegin;
  ierr = PetscLogEventBegin(QEP_Dense,0,0,0,0);CHKERRQ(ierr);
  n = PetscBLASIntCast(n_);
  ldt = PetscBLASIntCast(ldt_);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#endif
  
  /* selection sort */
  for (i=k;i<n-1;i++) {
    re = wr[i];
    im = wi[i];
    pos = 0;
    j=i+1; /* j points to the next eigenvalue */
#if !defined(PETSC_USE_COMPLEX)
    if (im != 0) j=i+2;
#endif
    /* find minimum eigenvalue */
    for (;j<n;j++) { 
      ierr = QEPCompareEigenvalues(qep,re,im,wr[j],wi[j],&result);CHKERRQ(ierr);
      if (result < 0) {
        re = wr[j];
        im = wi[j];
        pos = j;
      }
#if !defined(PETSC_USE_COMPLEX)
      if (wi[j] != 0) j++;
#endif
    }
    if (pos) {
      /* interchange blocks */
      ifst = PetscBLASIntCast(pos + 1);
      ilst = PetscBLASIntCast(i + 1);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,work,&info);
#else
      LAPACKtrexc_("V",&n,T,&ldt,Q,&n,&ifst,&ilst,&info);
#endif
      if (info) SETERRQ1(PETSC_ERR_LIB,"Error in Lapack xTREXC %d",info);
      /* recover original eigenvalues from T matrix */
      for (j=i;j<n;j++) {
        wr[j] = T[j*ldt+j];
#if !defined(PETSC_USE_COMPLEX)
        if (j<n-1 && T[j*ldt+j+1] != 0.0) {
          /* complex conjugate eigenvalue */
          wi[j] = sqrt(PetscAbsReal(T[j*ldt+j+1])) *
                  sqrt(PetscAbsReal(T[(j+1)*ldt+j]));
          wr[j+1] = wr[j];
          wi[j+1] = -wi[j];
          j++;
        } else
#endif
        wi[j] = 0.0;
      }
    }
#if !defined(PETSC_USE_COMPLEX)
    if (wi[i] != 0) i++;
#endif
  }

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(work);CHKERRQ(ierr);
#endif
  ierr = PetscLogEventEnd(QEP_Dense,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);

#endif 
}
