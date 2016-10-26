Java Machine Learning Library
=============================

[![Build Status](https://travis-ci.org/spbsu-ml-community/jmll.svg?branch=master)](https://travis-ci.org/spbsu-ml-community/jmll)

###About

This is machine learning library, written in Java and providing various machine learning algorithms. Mostly contributed by people form Applied Mathematics faculty of SPBSU.

###How to check this out correctly

1. Use command line instead IDEA to checkout it in order to avoid problems with git submodule downloading.
To clone project run `git clone --recurse-submodules git@github.com:spbsu-ml-community/jmll.git`

2. `cd` to the {project_dir}/commons and run `git checkout master`. It will switch current branch in commons to `master`. Now if you update JMLL project, `commons` will be updated too.

Remember that git submodule is just a pointer to particular commit of the another repository. And if you update 'commons' please update this pointer by committing 'commons' dir in JMLL directory.

###How to import the project

We use [Apache Maven](http://maven.apache.org/) for managing our project. You need Maven 3 to be installed on your computer. If you don't already have it (check with `mvn -version` command), [install](http://maven.apache.org/download.cgi) it. On Mac OS it can also be done with [Homebrew](http://brew.sh/) (`brew install maven`) or [MacPorts](http://www.macports.org/) (`sudo port install maven3`), on Linux -- with standard "apt-get" instllation tool (`sudo apt-get install maven`). Check Maven version after installation with `mvn -version` command.

1. Open IntelliJ Idea, select File -> Import Project. In file selection dialog box choose pom.xml file located in project root directory.
2. On the first page of "Import Project from Maven" window check "Import Maven projects automatically" and leave other settings by default. Click "Environment settings..." button and check that your Maven 3 installation is recognized by IntelliJ Idea. If not, specify the path to Maven home directory.
3. On SDK selection page select Java 1.7 SDK.
4. When clicking "Finish" on the last page of "Import Project from Maven" window IntelliJ Idea will warn you that ".idea" folder already exists and its content may be overwritten. Click "Yes" -- run configurations and other settings that we share won't be overwritten.
5. Wait until IntelliJ Idea downloads and resolves project dependencies. If it will ask you to reload project due to language level changes, do it.
6. Open "Maven Projects" tool window (View -> Tool Windows -> Maven Projects or just a "Maven Projects" button at the right side of IntelliJ Idea window) and check that there are no red-underlined modules. If there are some, click "Reimport All Maven Projects" button (button with blue cycled arrows at the left of the toolbar in the "Maven Projects" tool window).
7. Run tests to check that project is imported correctly, everything compiles and tests passes.

Enjoy!
