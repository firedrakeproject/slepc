/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc/private/slepcimpl.h>            /*I "slepcsys.h" I*/

/*@C
   SlepcConvMonitorCreate - Creates a SlepcConvMonitor context.

   Collective on PetscViewer

   Input Parameters:
+  viewer - the viewer where the monitor must send data
-  format - the format

   Output Parameter:
.  ctx - the created context

   Notes:
   The created context is used for EPS, SVD, PEP, and NEP monitor functions that just
   print the iteration numbers at which convergence takes place (XXXMonitorConverged).

   This function increases the reference count of the viewer so you can destroy the
   viewer object after this call.

   Level: developer

.seealso: SlepcConvMonitorDestroy()
@*/
PetscErrorCode SlepcConvMonitorCreate(PetscViewer viewer,PetscViewerFormat format,SlepcConvMonitor *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscNew(ctx);CHKERRQ(ierr);
  (*ctx)->viewer = viewer;
  (*ctx)->format = format;
  PetscFunctionReturn(0);
}

/*@C
   SlepcConvMonitorDestroy - Destroys a SlepcConvMonitor context.

   Collective on PetscViewer

   Input Parameters:
.  ctx - the SlepcConvMonitor context to be destroyed.

   Level: developer

.seealso: SlepcConvMonitorCreate()
@*/
PetscErrorCode SlepcConvMonitorDestroy(SlepcConvMonitor *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  ierr = PetscViewerDestroy(&(*ctx)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Given n vectors in V, this function gets references of them into W.
   If m<0 then some previous non-processed vectors remain in W and must be freed.
 */
PetscErrorCode SlepcBasisReference_Private(PetscInt n,Vec *V,PetscInt *m,Vec **W)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = SlepcBasisDestroy_Private(m,W);CHKERRQ(ierr);
  if (n>0) {
    ierr = PetscMalloc1(n,W);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      ierr = PetscObjectReference((PetscObject)V[i]);CHKERRQ(ierr);
      (*W)[i] = V[i];
    }
    *m = -n;
  }
  PetscFunctionReturn(0);
}

/*
   Destroys a set of vectors.
   A negative value of m indicates that W contains vectors to be destroyed.
 */
PetscErrorCode SlepcBasisDestroy_Private(PetscInt *m,Vec **W)
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (*m<0) {
    for (i=0;i<-(*m);i++) {
      ierr = VecDestroy(&(*W)[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(*W);CHKERRQ(ierr);
  }
  *m = 0;
  PetscFunctionReturn(0);
}

/*@C
   SlepcSNPrintfScalar - Prints a PetscScalar variable to a string of
   given length.

   Not Collective

   Input Parameters:
+  str - the string to print to
.  len - the length of str
.  val - scalar value to be printed
-  exp - to be used within an expression, print leading sign and parentheses
         in case of nonzero imaginary part

   Level: developer
@*/
PetscErrorCode SlepcSNPrintfScalar(char *str,size_t len,PetscScalar val,PetscBool exp)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      re,im;
#endif

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  if (exp) {
    ierr = PetscSNPrintf(str,len,"%+g",(double)val);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(str,len,"%g",(double)val);CHKERRQ(ierr);
  }
#else
  re = PetscRealPart(val);
  im = PetscImaginaryPart(val);
  if (im!=0.0) {
    if (exp) {
      ierr = PetscSNPrintf(str,len,"+(%g%+gi)",(double)re,(double)im);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(str,len,"%g%+gi",(double)re,(double)im);CHKERRQ(ierr);
    }
  } else {
    if (exp) {
      ierr = PetscSNPrintf(str,len,"%+g",(double)re);CHKERRQ(ierr);
    } else {
      ierr = PetscSNPrintf(str,len,"%g",(double)re);CHKERRQ(ierr);
    }
  }
#endif
  PetscFunctionReturn(0);
}

