/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Standard symmetric eigenproblem for the 3-D Laplacian built with the DM interface.\n\n"
"Use -seed <k> to modify the random initial vector.\n"
"Use -da_grid_x <nx> etc. to change the problem size.\n\n";

#include <slepceps.h>
#include <petscdmda.h>
#include <petsctime.h>

PetscErrorCode GetExactEigenvalues(PetscInt M,PetscInt N,PetscInt P,PetscInt nconv,PetscReal *exact)
{
  PetscInt       n,i,j,k,l;
  PetscReal      *evals,ax,ay,az,sx,sy,sz;

  PetscFunctionBeginUser;
  ax = PETSC_PI/2/(M+1);
  ay = PETSC_PI/2/(N+1);
  az = PETSC_PI/2/(P+1);
  n = PetscCeilReal(PetscPowReal((PetscReal)nconv,0.33333)+1);
  CHKERRQ(PetscMalloc1(n*n*n,&evals));
  l = 0;
  for (i=1;i<=n;i++) {
    sx = PetscSinReal(ax*i);
    for (j=1;j<=n;j++) {
      sy = PetscSinReal(ay*j);
      for (k=1;k<=n;k++) {
        sz = PetscSinReal(az*k);
        evals[l++] = 4.0*(sx*sx+sy*sy+sz*sz);
      }
    }
  }
  CHKERRQ(PetscSortReal(n*n*n,evals));
  for (i=0;i<nconv;i++) exact[i] = evals[i];
  CHKERRQ(PetscFree(evals));
  PetscFunctionReturn(0);
}

PetscErrorCode FillMatrix(DM da,Mat A)
{
  PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,idx;
  PetscScalar    v[7];
  MatStencil     row,col[7];

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm));

  for (k=zs;k<zs+zm;k++) {
    for (j=ys;j<ys+ym;j++) {
      for (i=xs;i<xs+xm;i++) {
        row.i=i; row.j=j; row.k=k;
        col[0].i=row.i; col[0].j=row.j; col[0].k=row.k;
        v[0]=6.0;
        idx=1;
        if (k>0) { v[idx]=-1.0; col[idx].i=i; col[idx].j=j; col[idx].k=k-1; idx++; }
        if (j>0) { v[idx]=-1.0; col[idx].i=i; col[idx].j=j-1; col[idx].k=k; idx++; }
        if (i>0) { v[idx]=-1.0; col[idx].i=i-1; col[idx].j=j; col[idx].k=k; idx++; }
        if (i<mx-1) { v[idx]=-1.0; col[idx].i=i+1; col[idx].j=j; col[idx].k=k; idx++; }
        if (j<my-1) { v[idx]=-1.0; col[idx].i=i; col[idx].j=j+1; col[idx].k=k; idx++; }
        if (k<mz-1) { v[idx]=-1.0; col[idx].i=i; col[idx].j=j; col[idx].k=k+1; idx++; }
        CHKERRQ(MatSetValuesStencil(A,1,&row,idx,col,v,INSERT_VALUES));
      }
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Mat            A;               /* operator matrix */
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  DM             da;
  Vec            v0;
  PetscReal      error,tol,re,im,*exact;
  PetscScalar    kr,ki;
  PetscInt       M,N,P,m,n,p,nev,maxit,i,its,nconv,seed;
  PetscLogDouble t1,t2,t3;
  PetscBool      flg,terse;
  PetscRandom    rctx;
  PetscErrorCode ierr;

  ierr = SlepcInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n3-D Laplacian Eigenproblem\n\n"));

  /* show detailed info unless -terse option is given by user */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-terse",&terse));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,10,10,10,
                      PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                      1,1,NULL,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /* print DM information */
  CHKERRQ(DMDAGetInfo(da,NULL,&M,&N,&P,&m,&n,&p,NULL,NULL,NULL,NULL,NULL,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Grid partitioning: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",m,n,p));

  /* create and fill the matrix */
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(FillMatrix(da,A));

  /* create random initial vector */
  seed = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-seed",&seed,NULL));
  PetscCheck(seed>=0,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Seed must be >=0");
  CHKERRQ(MatCreateVecs(A,&v0,NULL));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
  CHKERRQ(PetscRandomSetFromOptions(rctx));
  for (i=0;i<seed;i++) {   /* simulate different seeds in the random generator */
    CHKERRQ(VecSetRandom(v0,rctx));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create eigensolver context
  */
  CHKERRQ(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  CHKERRQ(EPSSetOperators(eps,A,NULL));
  CHKERRQ(EPSSetProblemType(eps,EPS_HEP));

  /*
     Set specific solver options
  */
  CHKERRQ(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  CHKERRQ(EPSSetTolerances(eps,1e-8,PETSC_DEFAULT));
  CHKERRQ(EPSSetInitialSpace(eps,1,&v0));

  /*
     Set solver parameters at runtime
  */
  CHKERRQ(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(PetscTime(&t1));
  CHKERRQ(EPSSetUp(eps));
  CHKERRQ(PetscTime(&t2));
  CHKERRQ(EPSSolve(eps));
  CHKERRQ(PetscTime(&t3));
  if (!terse) {
    CHKERRQ(EPSGetIterationNumber(eps,&its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

    /*
       Optional: Get some information from the solver and display it
    */
    CHKERRQ(EPSGetType(eps,&type));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
    CHKERRQ(EPSGetDimensions(eps,&nev,NULL,NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
    CHKERRQ(EPSGetTolerances(eps,&tol,&maxit));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (terse) CHKERRQ(EPSErrorView(eps,EPS_ERROR_RELATIVE,NULL));
  else {
    /*
       Get number of converged approximate eigenpairs
    */
    CHKERRQ(EPSGetConverged(eps,&nconv));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Number of converged approximate eigenpairs: %" PetscInt_FMT "\n\n",nconv));

    if (nconv>0) {
      CHKERRQ(PetscMalloc1(nconv,&exact));
      CHKERRQ(GetExactEigenvalues(M,N,P,nconv,exact));
      /*
         Display eigenvalues and relative errors
      */
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,
           "           k          ||Ax-kx||/||kx||   Eigenvalue Error \n"
           "   ----------------- ------------------ ------------------\n"));

      for (i=0;i<nconv;i++) {
        /*
          Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
          ki (imaginary part)
        */
        CHKERRQ(EPSGetEigenpair(eps,i,&kr,&ki,NULL,NULL));
        /*
           Compute the relative error associated to each eigenpair
        */
        CHKERRQ(EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error));

#if defined(PETSC_USE_COMPLEX)
        re = PetscRealPart(kr);
        im = PetscImaginaryPart(kr);
#else
        re = kr;
        im = ki;
#endif
        PetscCheck(im==0.0,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Eigenvalue should be real");
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"   %12g       %12g        %12g\n",(double)re,(double)error,(double)PetscAbsReal(re-exact[i])));
      }
      CHKERRQ(PetscFree(exact));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }
  }

  /*
     Show computing times
  */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-showtimes",&flg));
  if (flg) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Elapsed time: %g (setup), %g (solve)\n",(double)(t2-t1),(double)(t3-t2)));

  /*
     Free work space
  */
  CHKERRQ(EPSDestroy(&eps));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&v0));
  CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(DMDestroy(&da));
  ierr = SlepcFinalize();
  return ierr;
}

/*TEST

   testset:
      args: -eps_nev 8 -terse
      requires: double
      output_file: output/ex19_1.out
      test:
         suffix: 1_krylovschur
         args: -eps_type krylovschur -eps_ncv 64
      test:
         suffix: 1_lobpcg
         args: -eps_type lobpcg -eps_tol 1e-7
      test:
         suffix: 1_blopex
         args: -eps_type blopex -eps_tol 1e-7 -eps_blopex_blocksize 4
         requires: blopex

TEST*/
