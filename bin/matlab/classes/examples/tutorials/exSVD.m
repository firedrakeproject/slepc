%%
%
%  Computes a partial SVD of a matrix with SLEPc
%  User creates directly a PETSc Mat
%

%%
%  Set the Matlab path and initialize SLEPc
%
path(path,'../../')
if ~exist('PetscInitialize','file')
  PETSC_DIR = getenv('PETSC_DIR');
  if isempty(PETSC_DIR) 
    error('Must set environment variable PETSC_DIR or add the appropriate dir to Matlab path')
  end
  path(path,[PETSC_DIR '/bin/matlab/classes'])
end
SlepcInitialize({'-malloc','-malloc_debug','-malloc_dump'});

%%
%  Create the Lauchli matrix
%
n = 100;
mu = 1e-7;
mat = PetscMat();
mat.SetType('seqaij');
mat.SetSizes(n+1,n,n+1,n);
for i=1:n
  mat.SetValues(1,i,1.0);
end
for i=2:n+1
  mat.SetValues(i,i-1,mu);
end
mat.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
mat.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
%mat.View;

%%
%  Create the solver, pass the matrix and solve the problem
%
svd = SlepcSVD();
svd.SetOperator(mat);
svd.SetType('trlanczos');
svd.SetFromOptions();
svd.Solve();
nconv = svd.GetConverged();
fprintf('         sigma         residual norm\n')
fprintf('   ----------------- ------------------\n')
for i=1:nconv
  sigma = svd.GetSingularTriplet(i);
  relerr = svd.ComputeRelativeError(i);
  fprintf('   %12f       %12g\n',sigma,relerr)
end
%svd.View();

%%
%   Free objects and shutdown SLEPc
%
mat.Destroy();
svd.Destroy();
SlepcFinalize();
