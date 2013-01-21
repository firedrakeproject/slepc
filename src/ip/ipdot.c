/*
     Dot product routines

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2012, Universitat Politecnica de Valencia, Spain

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

#include <slepc-private/ipimpl.h>      /*I "slepcip.h" I*/

/* The following definitions are intended to avoid using the "T" versions
   of dot products in the case of real scalars */
#if defined(PETSC_USE_COMPLEX)
#define VecXDotBegin  VecTDotBegin
#define VecXDotEnd    VecTDotEnd  
#define VecMXDotBegin VecMTDotBegin
#define VecMXDotEnd   VecMTDotEnd  
#else
#define VecXDotBegin  VecDotBegin
#define VecXDotEnd    VecDotEnd  
#define VecMXDotBegin VecMDotBegin
#define VecMXDotEnd   VecMDotEnd  
#endif

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
   via IPSetMatrix(). For example, if using the B-inner product for 
   positive definite B, (x,y)_B=y^H Bx, then the computed norm is ||x||_B = 
   sqrt(x^H Bx).

   In an indefinite inner product, matrix B is indefinite and the norm is
   defined as s*sqrt(abs(x^H Bx)), where s = sign(x^H Bx).

   Level: developer

.seealso: IPInnerProduct()
@*/
PetscErrorCode IPNorm(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(norm,3);
  ierr = (*ip->ops->normbegin)(ip,x,norm);CHKERRQ(ierr);
  ierr = (*ip->ops->normend)(ip,x,norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormBegin_Bilinear"
PetscErrorCode IPNormBegin_Bilinear(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  ierr = IPInnerProductBegin(ip,x,x,&p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormBegin_Sesquilinear"
PetscErrorCode IPNormBegin_Sesquilinear(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  if (!ip->matrix) {
    ierr = VecNormBegin(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductBegin(ip,x,x,&p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormBegin_Indefinite"
PetscErrorCode IPNormBegin_Indefinite(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  if (!ip->matrix) {
    ierr = VecNormBegin(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductBegin(ip,x,x,&p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormBegin"
/*@
   IPNormBegin - Starts a split phase norm computation.

   Collective on IP and Vec

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(norm,3);
  ierr = (*ip->ops->normbegin)(ip,x,norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormEnd_Bilinear"
PetscErrorCode IPNormEnd_Bilinear(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  ierr = IPInnerProductEnd(ip,x,x,&p);CHKERRQ(ierr);
  if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
    ierr = PetscInfo(ip,"Zero norm, either the vector is zero or a semi-inner product is being used\n");CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))>PETSC_MACHINE_EPSILON) 
    SETERRQ(((PetscObject)ip)->comm,1,"IPNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(PetscRealPart(p));
#else
  if (p<0.0) SETERRQ(((PetscObject)ip)->comm,1,"IPNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(p);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormEnd_Sesquilinear"
PetscErrorCode IPNormEnd_Sesquilinear(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  if (!ip->matrix) {
    ierr = VecNormEnd(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductEnd(ip,x,x,&p);CHKERRQ(ierr);
    if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
      ierr = PetscInfo(ip,"Zero norm, either the vector is zero or a semi-inner product is being used\n");CHKERRQ(ierr);
    if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))/PetscAbsScalar(p)>PETSC_MACHINE_EPSILON) 
      SETERRQ(((PetscObject)ip)->comm,1,"IPNorm: The inner product is not well defined");
    *norm = PetscSqrtScalar(PetscRealPart(p));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormEnd_Indefinite"
PetscErrorCode IPNormEnd_Indefinite(IP ip,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  if (!ip->matrix) {
    ierr = VecNormEnd(x,NORM_2,norm);CHKERRQ(ierr);
  } else {
    ierr = IPInnerProductEnd(ip,x,x,&p);CHKERRQ(ierr);
    if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
      ierr = PetscInfo(ip,"Zero norm, either the vector is zero or a semi-inner product is being used\n");CHKERRQ(ierr);
    if (PetscAbsReal(PetscImaginaryPart(p))/PetscAbsScalar(p)>PETSC_MACHINE_EPSILON) 
      SETERRQ(((PetscObject)ip)->comm,1,"IPNorm: The inner product is not well defined");
    if (PetscRealPart(p)<0.0) *norm = -PetscSqrtScalar(-PetscRealPart(p));
    else *norm = PetscSqrtScalar(PetscRealPart(p));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPNormEnd"
/*@
   IPNormEnd - Ends a split phase norm computation.

   Collective on IP and Vec

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidPointer(norm,3);
  ierr = (*ip->ops->normend)(ip,x,norm);CHKERRQ(ierr);
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
   via IPSetMatrix(). This allows use of other inner products such as
   the indefinite product y^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y)_B=y^H Bx.

   Level: developer

.seealso: IPSetMatrix(), VecDot(), IPMInnerProduct()
@*/
PetscErrorCode IPInnerProduct(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidScalarPointer(p,4);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts++;
  ierr = (*ip->ops->innerproductbegin)(ip,x,y,p);CHKERRQ(ierr);
  ierr = (*ip->ops->innerproductend)(ip,x,y,p);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductBegin_Bilinear"
PetscErrorCode IPInnerProductBegin_Bilinear(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    ierr = VecXDotBegin(ip->Bx,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecXDotBegin(x,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductBegin_Sesquilinear"
PetscErrorCode IPInnerProductBegin_Sesquilinear(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    ierr = VecDotBegin(ip->Bx,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecDotBegin(x,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductBegin"
/*@
   IPInnerProductBegin - Starts a split phase inner product computation.

   Collective on IP and Vec

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
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidScalarPointer(p,4);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts++;
  ierr = (*ip->ops->innerproductbegin)(ip,x,y,p);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductEnd_Bilinear"
PetscErrorCode IPInnerProductEnd_Bilinear(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = VecXDotEnd(ip->Bx,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecXDotEnd(x,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductEnd_Sesquilinear"
PetscErrorCode IPInnerProductEnd_Sesquilinear(IP ip,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = VecDotEnd(ip->Bx,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecDotEnd(x,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPInnerProductEnd"
/*@
   IPInnerProductEnd - Ends a split phase inner product computation.

   Collective on IP and Vec

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
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  PetscValidScalarPointer(p,4);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ierr = (*ip->ops->innerproductend)(ip,x,y,p);CHKERRQ(ierr);
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
   if changed via IPSetMatrix(). This allows use of other inner products 
   such as the indefinite product y_i^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y_i)_B=y_i^H Bx.

   Level: developer

.seealso: IPSetMatrix(), VecMDot(), IPInnerProduct()
@*/
PetscErrorCode IPMInnerProduct(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_CLASSID,4);
  PetscValidScalarPointer(p,5);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts += n;
  ierr = (*ip->ops->minnerproductbegin)(ip,x,n,y,p);CHKERRQ(ierr);
  ierr = (*ip->ops->minnerproductend)(ip,x,n,y,p);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductBegin_Bilinear"
PetscErrorCode IPMInnerProductBegin_Bilinear(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    ierr = VecMXDotBegin(ip->Bx,n,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecMXDotBegin(x,n,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductBegin_Sesquilinear"
PetscErrorCode IPMInnerProductBegin_Sesquilinear(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = IPApplyMatrix_Private(ip,x);CHKERRQ(ierr);
    ierr = VecMDotBegin(ip->Bx,n,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecMDotBegin(x,n,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductBegin"
/*@
   IPMInnerProductBegin - Starts a split phase multiple inner product computation.

   Collective on IP and Vec

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
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (n == 0) PetscFunctionReturn(0);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_CLASSID,4);
  PetscValidScalarPointer(p,5);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ip->innerproducts += n;
  ierr = (*ip->ops->minnerproductbegin)(ip,x,n,y,p);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductEnd_Bilinear"
PetscErrorCode IPMInnerProductEnd_Bilinear(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = VecMXDotEnd(ip->Bx,n,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecMXDotEnd(x,n,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductEnd_Sesquilinear"
PetscErrorCode IPMInnerProductEnd_Sesquilinear(IP ip,Vec x,PetscInt n,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ip->matrix) {
    ierr = VecMDotEnd(ip->Bx,n,y,p);CHKERRQ(ierr);
  } else {
    ierr = VecMDotEnd(x,n,y,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "IPMInnerProductEnd"
/*@
   IPMInnerProductEnd - Ends a split phase multiple inner product computation.

   Collective on IP and Vec

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
  PetscValidHeaderSpecific(ip,IP_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,3);
  if (n == 0) PetscFunctionReturn(0);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_CLASSID,4);
  PetscValidScalarPointer(p,5);
  ierr = PetscLogEventBegin(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  ierr = (*ip->ops->minnerproductend)(ip,x,n,y,p);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(IP_InnerProduct,ip,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "IPCreate_Bilinear"
PetscErrorCode IPCreate_Bilinear(IP ip)
{
  PetscFunctionBegin;
  ip->ops->normbegin          = IPNormBegin_Bilinear;
  ip->ops->normend            = IPNormEnd_Bilinear;
  ip->ops->innerproductbegin  = IPInnerProductBegin_Bilinear;
  ip->ops->innerproductend    = IPInnerProductEnd_Bilinear;
  ip->ops->minnerproductbegin = IPMInnerProductBegin_Bilinear;
  ip->ops->minnerproductend   = IPMInnerProductEnd_Bilinear;
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
#undef __FUNCT__  
#define __FUNCT__ "IPCreate_Sesquilinear"
PetscErrorCode IPCreate_Sesquilinear(IP ip)
{
  PetscFunctionBegin;
  ip->ops->normbegin          = IPNormBegin_Sesquilinear;
  ip->ops->normend            = IPNormEnd_Sesquilinear;
  ip->ops->innerproductbegin  = IPInnerProductBegin_Sesquilinear;
  ip->ops->innerproductend    = IPInnerProductEnd_Sesquilinear;
  ip->ops->minnerproductbegin = IPMInnerProductBegin_Sesquilinear;
  ip->ops->minnerproductend   = IPMInnerProductEnd_Sesquilinear;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "IPCreate_Indefinite"
PetscErrorCode IPCreate_Indefinite(IP ip)
{
  PetscFunctionBegin;
  ip->ops->normbegin          = IPNormBegin_Indefinite;
  ip->ops->normend            = IPNormEnd_Indefinite;
#if defined(PETSC_USE_COMPLEX)
  ip->ops->innerproductbegin  = IPInnerProductBegin_Sesquilinear;
  ip->ops->innerproductend    = IPInnerProductEnd_Sesquilinear;
  ip->ops->minnerproductbegin = IPMInnerProductBegin_Sesquilinear;
  ip->ops->minnerproductend   = IPMInnerProductEnd_Sesquilinear;
#else
  ip->ops->innerproductbegin  = IPInnerProductBegin_Bilinear;
  ip->ops->innerproductend    = IPInnerProductEnd_Bilinear;
  ip->ops->minnerproductbegin = IPMInnerProductBegin_Bilinear;
  ip->ops->minnerproductend   = IPMInnerProductEnd_Bilinear;
#endif
  PetscFunctionReturn(0);
}
EXTERN_C_END

