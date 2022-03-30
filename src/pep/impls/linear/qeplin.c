/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Various types of linearization for quadratic eigenvalue problem
*/

#include <slepc/private/pepimpl.h>
#include "linear.h"

/*
    Given the quadratic problem (l^2*M + l*C + K)*x = 0 the linearization is
    A*z = l*B*z for z = [  x  ] and A,B defined as follows:
                        [ l*x ]

            N:
                     A = [-bK      aI    ]     B = [ aI+bC   bM    ]
                         [-aK     -aC+bI ]         [ bI      aM    ]

            S:
                     A = [ bK      aK    ]     B = [ aK-bC  -bM    ]
                         [ aK      aC-bM ]         [-bM     -aM    ]

            H:
                     A = [ aK     -bK    ]     B = [ bM      aK+bC ]
                         [ aC+bM   aK    ]         [-aM      bM    ]
 */

/* - - - N - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

PetscErrorCode MatCreateExplicit_Linear_NA(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  PetscInt       M,N,m,n;
  Mat            Id,T=NULL;
  PetscReal      a=ctx->alpha,b=ctx->beta;
  PetscScalar    scalt=1.0;

  PetscFunctionBegin;
  PetscCall(MatGetSize(ctx->M,&M,&N));
  PetscCall(MatGetLocalSize(ctx->M,&m,&n));
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)ctx->M),m,n,M,N,1.0,&Id));
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,-a*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatShift(T,b));
  } else {
    if (a==0.0) { T = Id; scalt = b; }
    else { T = ctx->C; scalt = -a*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(-b*ctx->dsfactor,ctx->K,a,Id,-ctx->dsfactor*a,ctx->K,scalt,T,A));
  PetscCall(MatDestroy(&Id));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateExplicit_Linear_NB(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  PetscInt       M,N,m,n;
  Mat            Id,T=NULL;
  PetscReal      a=ctx->alpha,b=ctx->beta;
  PetscScalar    scalt=1.0;

  PetscFunctionBegin;
  PetscCall(MatGetSize(ctx->M,&M,&N));
  PetscCall(MatGetLocalSize(ctx->M,&m,&n));
  PetscCall(MatCreateConstantDiagonal(PetscObjectComm((PetscObject)ctx->M),m,n,M,N,1.0,&Id));
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,b*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatShift(T,a));
  } else {
    if (b==0.0) { T = Id; scalt = a; }
    else { T = ctx->C; scalt = b*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(scalt,T,b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,b,Id,a*ctx->sfactor*ctx->sfactor*ctx->dsfactor,ctx->M,B));
  PetscCall(MatDestroy(&Id));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

/* - - - S - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

PetscErrorCode MatCreateExplicit_Linear_SA(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  Mat            T=NULL;
  PetscScalar    scalt=1.0;
  PetscReal      a=ctx->alpha,b=ctx->beta;

  PetscFunctionBegin;
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,a*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatAXPY(T,-b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,UNKNOWN_NONZERO_PATTERN));
  } else {
    if (a==0.0) { T = ctx->M; scalt = -b*ctx->dsfactor*ctx->sfactor*ctx->sfactor; }
    else { T = ctx->C; scalt = a*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(b*ctx->dsfactor,ctx->K,a*ctx->dsfactor,ctx->K,a*ctx->dsfactor,ctx->K,scalt,T,A));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateExplicit_Linear_SB(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  Mat            T=NULL;
  PetscScalar    scalt=1.0;
  PetscReal      a=ctx->alpha,b=ctx->beta;

  PetscFunctionBegin;
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,-b*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatAXPY(T,a*ctx->dsfactor,ctx->K,UNKNOWN_NONZERO_PATTERN));
  } else {
    if (b==0.0) { T = ctx->K; scalt = a*ctx->dsfactor; }
    else { T = ctx->C; scalt = -b*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(scalt,T,-b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,-b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,-a*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,B));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

/* - - - H - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

PetscErrorCode MatCreateExplicit_Linear_HA(MPI_Comm comm,PEP_LINEAR *ctx,Mat *A)
{
  Mat            T=NULL;
  PetscScalar    scalt=1.0;
  PetscReal      a=ctx->alpha,b=ctx->beta;

  PetscFunctionBegin;
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,a*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatAXPY(T,b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,UNKNOWN_NONZERO_PATTERN));
  } else {
    if (a==0.0) { T = ctx->M; scalt = b*ctx->dsfactor*ctx->sfactor*ctx->sfactor; }
    else { T = ctx->C; scalt = a*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(a*ctx->dsfactor,ctx->K,-b*ctx->dsfactor,ctx->K,scalt,T,a*ctx->dsfactor,ctx->K,A));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateExplicit_Linear_HB(MPI_Comm comm,PEP_LINEAR *ctx,Mat *B)
{
  Mat            T=NULL;
  PetscScalar    scalt=1.0;
  PetscReal      a=ctx->alpha,b=ctx->beta;

  PetscFunctionBegin;
  if (a!=0.0 && b!=0.0) {
    PetscCall(MatDuplicate(ctx->C,MAT_COPY_VALUES,&T));
    PetscCall(MatScale(T,b*ctx->dsfactor*ctx->sfactor));
    PetscCall(MatAXPY(T,a*ctx->dsfactor,ctx->K,UNKNOWN_NONZERO_PATTERN));
  } else {
    if (b==0.0) { T = ctx->K; scalt = a*ctx->dsfactor; }
    else { T = ctx->C; scalt = b*ctx->dsfactor*ctx->sfactor; }
  }
  PetscCall(MatCreateTile(b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,scalt,T,-a*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,b*ctx->dsfactor*ctx->sfactor*ctx->sfactor,ctx->M,B));
  if (a!=0.0 && b!=0.0) PetscCall(MatDestroy(&T));
  PetscFunctionReturn(0);
}
