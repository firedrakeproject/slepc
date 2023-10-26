
Notes for SLEPc developers and contributors
===========================================

This document is intended for pointing out the differences between SLEPc and PETSc that may be relevant for developers and contributors.

Additional information can be found in the [Developers section of PETSc's web site](https://petsc.org/release/developers).

Build system
------------

- The SLEPc `configure` script is independent of the PETSc counterpart, there is no overlap. It gets some PETSc configuration data from files `petscvariables` and `petscconf.h`.
- Makefiles in SLEPc follow a similar structure as in PETSc. When possible, relevant PETSc makefiles are included from SLEPc makefiles, in particular, files `rules`, `rules.util` and `variables` in `${PETSC_DIR}/lib/petsc/conf` are included.

Continuous integration
----------------------

- The SLEPc project uses its own gitlab-runners.
- As opposed to PETSc, merge requests do not generate a pipeline automatically. Pipelines must be run manually, with the `Run pipeline` button in the Pipelines menu item under CI/CD, except when running from a fork, in which case developers are advised to reach out to a maintainer.
- When running a pipeline, select the branch that you want to test. Note that this scheme may require rebasing the branch if the parent branch contains relevant changes (no merge is done automatically as in PETSc).
- In the `Run pipeline` screen, it is possible to select a PETSc branch by using `PETSC_BRANCH` as the variable key and the branch name as the variable value.
- The test harness is run with `DIFF_NUMBERS` enabled by default, as opposed to PETSc. When adding a new test, all the output must match, including floating point numbers. Use filters to remove potentially problematic values such as small residual norms.
- Using filters in tests is preferred to adding `alt` output files.

Code
----

- In SLEPc, code style is not enforced via `clang-format`. Still, most coding conventions should be followed. They are the same as the ones from PETSc prior to the switch to `clang-format`, see the [PETSc Style and Usage Guide](https://petsc.org/release/developers/style/).

Documentation
-------------

- SLEPc does not yet use the Sphinx-based documentation system implemented by PETSc.
- The SLEPc website is not included in the repository.
- The SLEPc users manual in PDF is generated directly from LaTeX source.
- The `alldoc` rule in the makefile uses the old rules, see file `slepc_rules_doc.mk`. In particular, Sowing is used to generate HTML man pages directly, not Markdown as in PETSc.
