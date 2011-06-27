%%
%
%  Solves a quadratic eigenvalue problem with SLEPc
%  User creates directly the three PETSc Mat
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
%  Create a tridiagonal matrix (1-D Laplacian)
%
n = 10;
N = n*n;
K = PetscMat();
K.SetType('seqaij');
K.SetSizes(N,N,N,N);
for II=0:N-1
  i = floor(II/n);
  j = II-i*n;  
  if i>0,   K.SetValues(II+1,II+1-n,-1.0); end
  if i<n-1, K.SetValues(II+1,II+1+n,-1.0); end
  if j>0,   K.SetValues(II+1,II+1-1,-1.0); end
  if j<n-1, K.SetValues(II+1,II+1+1,-1.0); end
  K.SetValues(II+1,II+1,4.0);
end
K.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
K.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

C = PetscMat();
C.SetType('seqaij');
C.SetSizes(N,N,N,N);
C.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
C.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

M = PetscMat();
M.SetType('seqaij');
M.SetSizes(N,N,N,N);
for II=1:N
  M.SetValues(II,II,1.0);
end
M.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
M.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);

%%
%  Create the eigensolver, pass the matrices and solve the problem
%
qep = SlepcQEP();
qep.SetType('qarnoldi');
qep.SetOperators(M,C,K);
qep.SetProblemType(SlepcQEP.GENERAL);
qep.SetFromOptions();
qep.Solve();
nconv = qep.GetConverged();
fprintf('           k          ||Ax-kx||/||kx||\n')
fprintf('   ----------------- ------------------\n')
for i=1:nconv
  lambda = qep.GetEigenpair(i);
  relerr = qep.ComputeRelativeError(i);
  if isreal(lambda)
    fprintf('    %14.2f        %12g\n',lambda,relerr)
  else
    fprintf('  %9f%+9f      %12g\n',real(lambda),imag(lambda),relerr)
  end
end
%qep.View();

%%
%   Free objects and shutdown SLEPc
%
K.Destroy();
C.Destroy();
M.Destroy();
qep.Destroy();
SlepcFinalize();
