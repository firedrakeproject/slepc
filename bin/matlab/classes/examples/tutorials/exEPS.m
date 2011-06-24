%%
%
%  Solves a simple eigenvalue problem with SLEPc
%  User creates directly a PETSc Mat
%

%%
%  Set the Matlab path and initialize SLEPc
%
path(path,'../../')
if exist('PetscInitialize')~=2
  PETSC_DIR = getenv('PETSC_DIR');
  if (length(PETSC_DIR) == 0) 
    error('Must set environment variable PETSC_DIR or add the appropriate dir to Matlab path')
  end
  path(path,[PETSC_DIR '/bin/matlab/classes'])
end
SlepcInitialize({'-eps_monitor','-malloc','-malloc_debug','-malloc_dump'});

%%
%  Create a tridiagonal matrix (1-D Laplacian)
%
n = 30;
mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(n,n,n,n);
for i=1:n
  mat.SetValues(i,i,2.0);
end
for i=1:n-1
  mat.SetValues(i+1,i,-1.0);
  mat.SetValues(i,i+1,-1.0);
end
mat.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
mat.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
%mat.View;

%%
%  Create the eigensolver, pass the matrix and solve the problem
%
eps = SlepcEPS();
eps.SetType('krylovschur');
eps.SetOperators(mat);
eps.SetProblemType(SlepcEPS.HEP);
eps.SetFromOptions();
eps.Solve();
nconv = eps.GetConverged();
fprintf('           k          ||Ax-kx||/||kx||\n')
fprintf('   ----------------- ------------------\n')
for i=1:nconv
  lambda = eps.GetEigenvalue(i);
  relerr = eps.ComputeRelativeError(i);
  fprintf('   %12f       %12g\n',lambda,relerr)
end
%eps.View();

%%
%   Free objects and shutdown SLEPc
%
mat.Destroy();
eps.Destroy();
SlepcFinalize();
