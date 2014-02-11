Java Machine Learning Library
=============================

###About

This is machine learning library, written in Java and providing various machine learning algorithms. Mostly contributed by people form Applied Mathematics faculty of SPBSU.

###RTFM: How to check this out correctly

1. Use command line instead IDEA to checkout it in order to avoid problems with git submodule downloading.
To clone project run `git clone --recurse-submodules git@github.com:spbsu-ml-community/jmll.git`

2. `cd` to the {project_dir}/commons and run `git checkout master`. It will switch current branch in commons to `master`. Now if you update JMLL project, `commons` will be updated too.

3. When you import this project in IDEA please do:

  3.1. uncheck `libs` and `libs1` as libraries
    
  3.2. uncheck `tests` and `tests1` as modules

  3.3. choose `reuse` for module's '.iml' file

Remember that git submodule is just a pointer to particular commit of the another repository. And if you update 'commons' please update this pointer by committing 'commons' dir in JMLL directory.

Enjoy!
