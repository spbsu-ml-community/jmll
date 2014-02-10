Java Machine Learning Library

=============================================

RTFM:

1. Use command line instead IDEA to checkout it in order to avoid problems with git submodule downloading

2. After clonning project 'cd' to project dir and run 'git submodule update --init'

3. When you import this project please:

    3.1. uncheck 'libs' and 'libs1' as libraries
 
    3.2. uncheck 'tests' and 'tests1' as modules

    3.3. choose 'reuse' for '.iml' files



Remember that git submodule is just a pointer to particular commit of the another repository. And if you update 'commons' please update this pointer by committing 'commons' dir in JMLL directory.

=============================================

Enjoy!
