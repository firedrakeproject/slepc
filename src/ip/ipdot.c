/*
     Dot product routines

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

#include "private/ipimpl.h"      /*I "slepcip.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "IPNorm"
/*@
   IPNorm - Computes the norm of a vector as the square root of the inner 
   product (x,x) as defined by IPInnerProduct().

   Collective on IP and Vec

   Input Parameters:
+  ip - the inner product context
-  x  - input vector

   Output Parameter:
.  norm - the computed norm

   Notes:
   This function will usually compute the 2-norm of a vector, ||x||_2. But
   this behaviour may be different if using a non-standard inner product changed 
   via IPSetBilinearForm(). For example, if using the B-inner product for 
   positive definite B, (x,y)_B=y^H Bx, then the computed norm is ||x||_B = 
   sqrt( x^H Bx ).

   Level: developer

.seealso: IPInnerProduct()
@*/
PetscErrorCode IPNorm(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  if (!ip->matrix && ip->bilinear_form == IPINNER_HERMITIAN) {
    ierr = VecNorm(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProduct(ip,x,x,&p);CHKERRQ(ierr);
    if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
      PetscInfo(ip,"Zero norm, either the vector is zero or a semi-inner product is being used\n");
#if defined(PETSC_USE_COMPLEX)
    if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))>PETSC_MACHINE_EPSILON) 
       SETERRQ(1,"IPNorm: The inner product is not well defined");
    *norm = PetscSqrtScalar(PetscRealPart(p));
#else
    if (p<0.0) SETERRQ(1,"IPNorm: The inner product is not well defined");
    *norm = PetscSqrtScalar(p);
#endif
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormBegin"
/*@
   IPNormBegin - Starts a split phase norm computation.

   Input Parameters:
+  ip   - the inner product context
.  x    - input vector
-  norm - where the result will go

   Level: developer

   Notes:
   Each call to IPNormBegin() should be paired with a call to IPNormEnd().

.seealso: IPNormEnd(), IPNorm(), IPInnerProduct(), IPMInnerProduct(), 
          IPInnerProductBegin(), IPInnerProductEnd()

@*/
PetscErrorCode IPNormBegin(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  if (!ip->matrix && ip->bilinear_form == IPINNER_HERMITIAN) {
    ierr = VecNormBegin(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductBegin(ip,x,x,&p);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormEnd"
/*@
   IPNormEnd - Ends a split phase norm computation.

   Input Parameters:
+  ip   - the inner product context
-  x    - input vector

   Output Parameter:
.  norm - the computed norm

   Level: developer

   Notes:
   Each call to IPNormBegin() should be paired with a call to IPNormEnd().

.seealso: IPNormBegin(), IPNorm(), IPInnerProduct(), IPMInnerProduct(), 
          IPInnerProductBegin(), IPInnerProductEnd()

@*/
PetscErrorCode IPNormEnd(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  if (!ip->matrix && ip->bilinear_form == IPINNER_HERMITIAN) {
    ierr = VecNormEnd(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductEnd(ip,x,x,&p);CHKERRQ(ierr);
    if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
      PetscInfo(ip,"Zero norm, either the vector is zero or a semi-inner product is being used\n");

#if defined(PETSC_USE_COMPLEX)
    if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))>PETSC_MACHINE_EPSILON) 
       SETERRQ(1,"IPNorm: The inner product is not well defined");
    *norm = PetscSqrtScalar(PetscRealPart(p));
#else
    if (p<0.0) SETERRQ(1,"IPNorm: The inner product is not well defined");
    *norm = PetscSqrtScalar(p);
#endif
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProduct"
/*@
   IPInnerProduct - Computes the inner product of two vectors.

   Collective on IP and Vec

   Input Parameters:
+  ip - the inner product context
.  x  - input vector
-  y  - input vector

   Output Parameter:
.  p - result of the inner product

   Notes:
   This function will usually compute the standard dot product of vectors
   x and y, (x,y)=y^H x. However this behaviour may be different if changed 
   via IPSetBilinearForm(). This allows use of other inner products such as
   the indefinite product y^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y)_B=y^H Bx.

   Level: developer

.seealso: IPSetBilinearForm(), IPApplyB(), VecDot(), IPMInnerProduct()
@*/
PetscErrorCode IPInnerProduct(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
      
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts++;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDot(ip->Bx,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDot(ip->Bx,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDot(x,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDot(x,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductBegin"
/*@
   IPInnerProductBegin - Starts a split phase inner product computation.

   Input Parameters:
+  ip - the inner product context
.  x  - the first vector
.  y  - the second vector
-  p  - where the result will go

   Level: developer

   Notes:
   Each call to IPInnerProductBegin() should be paired with a call to IPInnerProductEnd().

.seealso: IPInnerProductEnd(), IPInnerProduct(), IPNorm(), IPNormBegin(), 
          IPNormEnd(), IPMInnerProduct() 

@*/
PetscErrorCode IPInnerProductBegin(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
  
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts++;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDotBegin(ip->Bx,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDotBegin(ip->Bx,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDotBegin(x,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDotBegin(x,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductEnd"
/*@
   IPInnerProductEnd - Ends a split phase inner product computation.

   Input Parameters:
+  ip - the inner product context
.  x  - the first vector
-  y  - the second vector

   Output Parameter:
.  p  - result of the inner product

   Level: developer

   Notes:
   Each call to IPInnerProductBegin() should be paired with a call to IPInnerProductEnd().

.seealso: IPInnerProductBegin(), IPInnerProduct(), IPNorm(), IPNormBegin(), 
          IPNormEnd(), IPMInnerProduct() 

@*/
PetscErrorCode IPInnerProductEnd(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
  
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  if (ip->matrix) {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDotEnd(ip->Bx,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDotEnd(ip->Bx,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecDotEnd(x,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecTDotEnd(x,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProduct"
/*@
   IPMInnerProduct - Computes the inner products a vector x with a set of
   vectors (columns of Y).

   Collective on IP and Vec

   Input Parameters:
+  ip - the inner product context
.  x  - the first input vector
.  n  - number of vectors in y
-  y  - array of vectors

   Output Parameter:
.  p - result of the inner products

   Notes:
   This function will usually compute the standard dot product of x and y_i, 
   (x,y_i)=y_i^H x, for each column of Y. However this behaviour may be different
   if changed via IPSetBilinearForm(). This allows use of other inner products 
   such as the indefinite product y_i^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y_i)_B=y_i^H Bx.

   Level: developer

.seealso: IPSetBilinearForm(), IPApplyB(), VecMDot(), IPInnerProduct()
@*/
PetscErrorCode IPMInnerProduct(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts += n;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDot(ip->Bx,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDot(ip->Bx,n,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDot(x,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDot(x,n,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductBegin"
/*@
   IPMInnerProductBegin - Starts a split phase multiple inner product computation.

   Input Parameters:
+  ip - the inner product context
.  x  - the first input vector
.  n  - number of vectors in y
.  y  - array of vectors
-  p  - where the result will go

   Level: developer

   Notes:
   Each call to IPMInnerProductBegin() should be paired with a call to IPMInnerProductEnd().

.seealso: IPMInnerProductEnd(), IPMInnerProduct(), IPNorm(), IPNormBegin(), 
          IPNormEnd(), IPInnerProduct() 

@*/
PetscErrorCode IPMInnerProductBegin(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts += n;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDotBegin(ip->Bx,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDotBegin(ip->Bx,n,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDotBegin(x,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDotBegin(x,n,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductEnd"
/*@
   IPMInnerProductEnd - Ends a split phase multiple inner product computation.

   Input Parameters:
+  ip - the inner product context
.  x  - the first input vector
.  n  - number of vectors in y
-  y  - array of vectors

   Output Parameter:
.  p - result of the inner products

   Level: developer

   Notes:
   Each call to IPMInnerProductBegin() should be paired with a call to IPMInnerProductEnd().

.seealso: IPMInnerProductBegin(), IPMInnerProduct(), IPNorm(), IPNormBegin(), 
          IPNormEnd(), IPInnerProduct() 

@*/
PetscErrorCode IPMInnerProductEnd(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  if (ip->matrix) {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDotEnd(ip->Bx,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDotEnd(ip->Bx,n,y,p);CHKERRQ(ierr);
    }
  } else {
    if (ip->bilinear_form == IPINNER_HERMITIAN) {
      ierr = VecMDotEnd(x,n,y,p);CHKERRQ(ierr);
    } else {
      ierr = VecMTDotEnd(x,n,y,p);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
