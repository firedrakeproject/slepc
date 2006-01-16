
/*
    The ST (spectral transformation) interface routines, callable by users.
*/

#include "src/st/stimpl.h"            /*I "slepcst.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "STApply"
/*@
   STApply - Applies the spectral transformation operator to a vector, for
   instance (A - sB)^-1 B in the case of the shift-and-invert tranformation
   and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApplyB(), STApplyNoB()
@*/
PetscErrorCode STApply(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->apply)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_Apply,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyB"
/*@
   STApplyB - Applies the B matrix to a vector.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply(), STApplyNoB()
@*/
PetscErrorCode STApplyB(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  if (x->id == st->xid && x->state == st->xstate) {
    ierr = VecCopy(st->Bx, y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscLogEventBegin(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applyB)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyB,st,x,y,0);CHKERRQ(ierr);
  
  st->xid = x->id;
  st->xstate = x->state;
  ierr = VecCopy(y,st->Bx);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyNoB"
/*@
   STApplyNoB - Applies the spectral transformation operator to a vector 
   which has already been multiplied by matrix B. For instance, this routine
   would perform the operation y =(A - sB)^-1 x in the case of the 
   shift-and-invert tranformation and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector, where it is assumed that x=Bw for some vector w

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply(), STApplyB()
@*/
PetscErrorCode STApplyNoB(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_ApplyNoB,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applynoB)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyNoB,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STApplyTranspose"
/*@
   STApplyTranspose - Applies the transpose of the operator to a vector, for
   instance B^T(A - sB)^-T in the case of the shift-and-invert tranformation
   and generalized eigenproblem.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  y - output vector

   Level: developer

.seealso: STApply()
@*/
PetscErrorCode STApplyTranspose(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  if (x == y) SETERRQ(PETSC_ERR_ARG_IDN,"x and y must be different vectors");

  if (!st->setupcalled) { ierr = STSetUp(st); CHKERRQ(ierr); }

  ierr = PetscLogEventBegin(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  ierr = (*st->ops->applytrans)(st,x,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(ST_ApplyTranspose,st,x,y,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STComputeExplicitOperator"
/*@
    STComputeExplicitOperator - Computes the explicit operator associated
    to the eigenvalue problem with the specified spectral transformation.  

    Collective on ST

    Input Parameter:
.   st - the spectral transform context

    Output Parameter:
.   mat - the explicit operator

    Notes:
    This routine builds a matrix containing the explicit operator. For 
    example, in generalized problems with shift-and-invert spectral
    transformation the result would be matrix (A - s B)^-1 B.
    
    This computation is done by applying the operator to columns of the 
    identity matrix. Note that the result is a dense matrix.

    Level: advanced

.seealso: STApply()   
@*/
PetscErrorCode STComputeExplicitOperator(ST st,Mat *mat)
{
  PetscErrorCode ierr;
  Vec            in,out;
  PetscInt       i,M,m,*rows,start,end;
  PetscScalar    *array,one = 1.0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidPointer(mat,2);

  ierr = MatGetVecs(st->A,&in,&out);CHKERRQ(ierr);
  ierr = VecGetSize(out,&M);CHKERRQ(ierr);
  ierr = VecGetLocalSize(out,&m);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(out,&start,&end);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(int),&rows);CHKERRQ(ierr);
  for (i=0; i<m; i++) rows[i] = start + i;

  ierr = MatCreateMPIDense(st->comm,m,m,M,M,PETSC_NULL,mat);CHKERRQ(ierr);

  for (i=0; i<M; i++) {
    ierr = VecSet(in,0.0);CHKERRQ(ierr);
    ierr = VecSetValues(in,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(in);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(in);CHKERRQ(ierr);

    ierr = STApply(st,in,out); CHKERRQ(ierr);
    
    ierr = VecGetArray(out,&array);CHKERRQ(ierr);
    ierr = MatSetValues(*mat,m,rows,1,&i,array,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = VecRestoreArray(out,&array);CHKERRQ(ierr);
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecDestroy(in);CHKERRQ(ierr);
  ierr = VecDestroy(out);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STNorm"
/*@
   STNorm - Computes de norm of a vector as the square root of the inner 
   product (x,x) as defined by STInnerProduct().

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
-  x  - input vector

   Output Parameter:
.  norm - the computed norm

   Notes:
   This function will usually compute the 2-norm of a vector, ||x||_2. But
   this behaviour may be different if using a non-standard inner product changed 
   via STSetBilinearForm(). For example, if using the B-inner product for 
   positive definite B, (x,y)_B=y^H Bx, then the computed norm is ||x||_B = 
   sqrt( x^H Bx ).

   Level: developer

.seealso: STInnerProduct()
@*/
PetscErrorCode STNorm(ST st,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  ierr = STInnerProduct(st,x,x,&p);CHKERRQ(ierr);

  if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
    PetscInfo(st,"Zero norm, either the vector is zero or a semi-inner product is being used\n");

#if defined(PETSC_USE_COMPLEX)
  if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))>PETSC_MACHINE_EPSILON) 
     SETERRQ(1,"STNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(PetscRealPart(p));
#else
  if (p<0.0) SETERRQ(1,"STNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(p);
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STNormBegin"
/*@
   STNormBegin - Starts a split phase norm computation.

   Input Parameters:
+  st   - the spectral transformation context
.  x    - input vector
-  norm - where the result will go

   Level: developer

   Notes:
   Each call to STNormBegin() should be paired with a call to STNormEnd().

.seealso: STNormEnd(), STNorm(), STInnerProduct(), STMInnerProduct(), 
          STInnerProductBegin(), STInnerProductEnd()

@*/
PetscErrorCode STNormBegin(ST st,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  ierr = STInnerProductBegin(st,x,x,&p);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STNormEnd"
/*@
   STNormEnd - Ends a split phase norm computation.

   Input Parameters:
+  st   - the spectral transformation context
-  x    - input vector

   Output Parameter:
.  norm - the computed norm

   Level: developer

   Notes:
   Each call to STNormBegin() should be paired with a call to STNormEnd().

.seealso: STNormBegin(), STNorm(), STInnerProduct(), STMInnerProduct(), 
          STInnerProductBegin(), STInnerProductEnd()

@*/
PetscErrorCode STNormEnd(ST st,Vec x,PetscReal *norm)
{
  PetscErrorCode ierr;
  PetscScalar    p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidPointer(norm,3);
  
  ierr = STInnerProductEnd(st,x,x,&p);CHKERRQ(ierr);

  if (PetscAbsScalar(p)<PETSC_MACHINE_EPSILON)
    PetscInfo(st,"Zero norm, either the vector is zero or a semi-inner product is being used\n");

#if defined(PETSC_USE_COMPLEX)
  if (PetscRealPart(p)<0.0 || PetscAbsReal(PetscImaginaryPart(p))>PETSC_MACHINE_EPSILON) 
     SETERRQ(1,"STNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(PetscRealPart(p));
#else
  if (p<0.0) SETERRQ(1,"STNorm: The inner product is not well defined");
  *norm = PetscSqrtScalar(p);
#endif

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STInnerProduct"
/*@
   STInnerProduct - Computes the inner product of two vectors.

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
.  x  - input vector
-  y  - input vector

   Output Parameter:
.  p - result of the inner product

   Notes:
   This function will usually compute the standard dot product of vectors
   x and y, (x,y)=y^H x. However this behaviour may be different if changed 
   via STSetBilinearForm(). This allows use of other inner products such as
   the indefinite product y^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y)_B=y^H Bx.

   Level: developer

.seealso: STSetBilinearForm(), STApplyB(), VecDot(), STMInnerProduct()
@*/
PetscErrorCode STInnerProduct(ST st,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_SYMMETRIC:
    ierr = VecCopy(x,st->w);CHKERRQ(ierr);
    break;
  case STINNER_B_HERMITIAN:
  case STINNER_B_SYMMETRIC:
    ierr = STApplyB(st,x,st->w);CHKERRQ(ierr);
    break;
  }
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = VecDot(st->w,y,p);CHKERRQ(ierr);
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = VecTDot(st->w,y,p);CHKERRQ(ierr);
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STInnerProductBegin"
/*@
   STInnerProductBegin - Starts a split phase inner product computation.

   Input Parameters:
+  st - the spectral transformation context
.  x  - the first vector
.  y  - the second vector
-  p  - where the result will go

   Level: developer

   Notes:
   Each call to STInnerProductBegin() should be paired with a call to STInnerProductEnd().

.seealso: STInnerProductEnd(), STInnerProduct(), STNorm(), STNormBegin(), 
          STNormEnd(), STMInnerProduct() 

@*/
PetscErrorCode STInnerProductBegin(ST st,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_SYMMETRIC:
    ierr = VecCopy(x,st->w);CHKERRQ(ierr);
    break;
  case STINNER_B_HERMITIAN:
  case STINNER_B_SYMMETRIC:
    ierr = STApplyB(st,x,st->w);CHKERRQ(ierr);
    break;
  }
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = VecDotBegin(st->w,y,p);CHKERRQ(ierr);
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = VecTDotBegin(st->w,y,p);CHKERRQ(ierr);
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STInnerProductEnd"
/*@
   STInnerProductEnd - Ends a split phase inner product computation.

   Input Parameters:
+  st - the spectral transformation context
.  x  - the first vector
-  y  - the second vector

   Output Parameter:
.  p  - result of the inner product

   Level: developer

   Notes:
   Each call to STInnerProductBegin() should be paired with a call to STInnerProductEnd().

.seealso: STInnerProductBegin(), STInnerProduct(), STNorm(), STNormBegin(), 
          STNormEnd(), STMInnerProduct() 

@*/
PetscErrorCode STInnerProductEnd(ST st,Vec x,Vec y,PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,2);
  PetscValidHeaderSpecific(y,VEC_COOKIE,3);
  PetscValidScalarPointer(p,4);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = VecDotEnd(st->w,y,p);CHKERRQ(ierr);
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = VecTDotEnd(st->w,y,p);CHKERRQ(ierr);
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STMInnerProduct"
/*@
   STMInnerProduct - Computes the inner products a vector x with a set of
   vectors (columns of Y).

   Collective on ST and Vec

   Input Parameters:
+  st - the spectral transformation context
.  n  - number of vectors in y
.  x  - the first input vector
-  y  - array of vectors

   Output Parameter:
.  p - result of the inner products

   Notes:
   This function will usually compute the standard dot product of x and y_i, 
   (x,y_i)=y_i^H x, for each column of Y. However this behaviour may be different
   if changed via STSetBilinearForm(). This allows use of other inner products 
   such as the indefinite product y_i^T x for complex symmetric problems or the
   B-inner product for positive definite B, (x,y_i)_B=y_i^H Bx.

   Level: developer

.seealso: STSetBilinearForm(), STApplyB(), VecMDot(), STInnerProduct()
@*/
PetscErrorCode STMInnerProduct(ST st,PetscInt n,Vec x,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_SYMMETRIC:
    ierr = VecCopy(x,st->w);CHKERRQ(ierr);
    break;
  case STINNER_B_HERMITIAN:
  case STINNER_B_SYMMETRIC:
    ierr = STApplyB(st,x,st->w);CHKERRQ(ierr);
    break;
  }
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = VecMDot(st->w,n,y,p);CHKERRQ(ierr);
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = VecMTDot(st->w,n,y,p);CHKERRQ(ierr);
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STMInnerProductBegin"
/*@
   STMInnerProductBegin - Starts a split phase multiple inner product computation.

   Input Parameters:
+  st - the spectral transformation context
.  n  - number of vectors in y
.  x  - the first input vector
.  y  - array of vectors
-  p  - where the result will go

   Level: developer

   Notes:
   Each call to STMInnerProductBegin() should be paired with a call to STMInnerProductEnd().

.seealso: STMInnerProductEnd(), STMInnerProduct(), STNorm(), STNormBegin(), 
          STNormEnd(), STInnerProduct() 

@*/
PetscErrorCode STMInnerProductBegin(ST st,PetscInt n,Vec x,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;
  int            i;
  PetscTruth     mdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_SYMMETRIC:
    ierr = VecCopy(x,st->w);CHKERRQ(ierr);
    break;
  case STINNER_B_HERMITIAN:
  case STINNER_B_SYMMETRIC:
    ierr = STApplyB(st,x,st->w);CHKERRQ(ierr);
    break;
  }
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = PetscOptionsHasName(st->prefix,"-mdot",&mdot);CHKERRQ(ierr);
    if (mdot) {
      ierr = VecMDotBegin(st->w,n,y,p);CHKERRQ(ierr);
    } else {
      for (i=0;i<n;i++) {
	ierr = VecDotBegin(st->w,y[i],p+i);CHKERRQ(ierr);
      }
    }
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = PetscOptionsHasName(st->prefix,"-mdot",&mdot);CHKERRQ(ierr);
    if (mdot) {
      ierr = VecMTDotBegin(st->w,n,y,p);CHKERRQ(ierr);
    } else {
      for (i=0;i<n;i++) {
	ierr = VecTDotBegin(st->w,y[i],p+i);CHKERRQ(ierr);
      }
    }
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STMInnerProductEnd"
/*@
   STMInnerProductEnd - Ends a split phase multiple inner product computation.

   Input Parameters:
+  st - the spectral transformation context
.  n  - number of vectors in y
.  x  - the first input vector
-  y  - array of vectors

   Output Parameter:
.  p - result of the inner products

   Level: developer

   Notes:
   Each call to STMInnerProductBegin() should be paired with a call to STMInnerProductEnd().

.seealso: STMInnerProductBegin(), STMInnerProduct(), STNorm(), STNormBegin(), 
          STNormEnd(), STInnerProduct() 

@*/
PetscErrorCode STMInnerProductEnd(ST st,PetscInt n,Vec x,const Vec y[],PetscScalar *p)
{
  PetscErrorCode ierr;
  int            i;
  PetscTruth     mdot;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  PetscValidHeaderSpecific(x,VEC_COOKIE,3);
  PetscValidPointer(y,4);
  PetscValidHeaderSpecific(*y,VEC_COOKIE,4);
  PetscValidScalarPointer(p,5);
  
  ierr = PetscLogEventBegin(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  switch (st->bilinear_form) {
  case STINNER_HERMITIAN:
  case STINNER_B_HERMITIAN:
    ierr = PetscOptionsHasName(st->prefix,"-mdot",&mdot);CHKERRQ(ierr);
    if (mdot) {
      ierr = VecMDotEnd(st->w,n,y,p);CHKERRQ(ierr);
    } else {
      for (i=0;i<n;i++) {
	ierr = VecDotEnd(st->w,y[i],p+i);CHKERRQ(ierr);
      }
    }
    break;
  case STINNER_SYMMETRIC:
  case STINNER_B_SYMMETRIC:
    ierr = PetscOptionsHasName(st->prefix,"-mdot",&mdot);CHKERRQ(ierr);
    if (mdot) {
      ierr = VecMTDotEnd(st->w,n,y,p);CHKERRQ(ierr);
    } else {
      for (i=0;i<n;i++) {
	ierr = VecTDotEnd(st->w,y[i],p+i);CHKERRQ(ierr);
      }
    }
    break;
  }
  ierr = PetscLogEventEnd(ST_InnerProduct,st,x,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STSetUp"
/*@
   STSetUp - Prepares for the use of a spectral transformation.

   Collective on ST

   Input Parameter:
.  st - the spectral transformation context

   Level: advanced

.seealso: STCreate(), STApply(), STDestroy()
@*/
PetscErrorCode STSetUp(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);

  PetscInfo(st,"Setting up new ST\n");
  if (st->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  if (!st->A) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be set first");}
  if (!st->type_name) {
    ierr = STSetType(st,STSHIFT);CHKERRQ(ierr);
  }
  if (st->w) { ierr = VecDestroy(st->w);CHKERRQ(ierr); }
  if (st->Bx) { ierr = VecDestroy(st->Bx);CHKERRQ(ierr); }
  ierr = MatGetVecs(st->A,&st->w,&st->Bx);CHKERRQ(ierr);
  st->xid = 0; 
  if (st->ops->setup) {
    ierr = (*st->ops->setup)(st); CHKERRQ(ierr);
  }
  st->setupcalled = 1;
  ierr = PetscLogEventEnd(ST_SetUp,st,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STPostSolve"
/*@
   STPostSolve - Optional post-solve phase, intended for any actions that must 
   be performed on the ST object after the eigensolver has finished.

   Collective on ST

   Input Parameters:
.  st  - the spectral transformation context

   Level: developer

.seealso: EPSSolve()
@*/
PetscErrorCode STPostSolve(ST st)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->postsolve) {
    ierr = (*st->ops->postsolve)(st);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "STBackTransform"
/*@
   STBackTransform - Back-transformation phase, intended for 
   spectral transformations which require to transform the computed 
   eigenvalues back to the original eigenvalue problem.

   Collective on ST

   Input Parameters:
   st   - the spectral transformation context
   eigr - real part of a computed eigenvalue
   eigi - imaginary part of a computed eigenvalue

   Level: developer

.seealso: EPSBackTransform()
@*/
PetscErrorCode STBackTransform(ST st,PetscScalar* eigr,PetscScalar* eigi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(st,ST_COOKIE,1);
  if (st->ops->backtr) {
    ierr = (*st->ops->backtr)(st,eigr,eigi);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
