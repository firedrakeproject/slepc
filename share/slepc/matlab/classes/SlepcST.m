classdef SlepcST < PetscObject
%
%   SlepcST - a SLEPc spectral transformation object
%
%   Creation from an EPS:
%     st = eps.GetST();

%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
%  SLEPc - Scalable Library for Eigenvalue Problem Computations
%  Copyright (c) 2002-2014, Universitat Politecnica de Valencia, Spain
%
%  This file is part of SLEPc.
%
%  SLEPc is free software: you can redistribute it and/or modify it under  the
%  terms of version 3 of the GNU Lesser General Public License as published by
%  the Free Software Foundation.
%
%  SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
%  WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
%  FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
%  more details.
%
%  You  should have received a copy of the GNU Lesser General  Public  License
%  along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  properties (Constant)
    HEP=1;
    GHEP=2;
    NHEP=3;
    GNHEP=4;
    PGNHEP=5;
    GHIEP=6;
  end
  methods
    function obj = SlepcST(pid,flag)
      if (nargin > 1)
        %  SelpcST(pid,'pobj') uses an already existing SLEPc ST object
        obj.pobj = pid;
        return
      end
      comm =  PETSC_COMM_SELF();
      [err,obj.pobj] = calllib('libslepc', 'STCreate', comm,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libslepc', 'STSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libslepc', 'STSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetUp(obj)
      err = calllib('libslepc', 'STSetUp', obj.pobj);PetscCHKERRQ(err);
    end
    function err = SetOperators(obj,C)
      if ~iscell(C), error('Argument of ST.SetOperators must be a cell array'), end
      n = length(C);
      M = [];
      for i=1:n, M = [M, C{i}.pobj]; end
      err = calllib('libslepc', 'STSetOperators', obj.pobj, n, M);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      if (nargin == 1)
        err = calllib('libslepc', 'STView', obj.pobj,0);PetscCHKERRQ(err);
      else
        err = calllib('libslepc', 'STView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
      end
    end
    function [ksp,err] = GetKSP(obj)
      [err,pid] = calllib('libslepc', 'STGetKSP', obj.pobj,0);PetscCHKERRQ(err);
      ksp = PetscKSP(pid,'pobj');
    end
    function err = Destroy(obj)
      err = calllib('libslepc', 'STDestroy', obj.pobj);PetscCHKERRQ(err);
    end
  end
end

