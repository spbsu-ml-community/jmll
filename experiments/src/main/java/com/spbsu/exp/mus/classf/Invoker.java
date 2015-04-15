package com.spbsu.exp.mus.classf;

import java.io.File;

/**
 * jmll
 * ksen
 * 11.April.2015 at 00:32
 */
public class Invoker {

  public static void main(final String[] args) {
    final Preprocessing preprocessing = new Preprocessing(
        new File("/home/ksen/Documents/data/TUD/data"),
        new File("/home/ksen/Documents/data/TUD/spectro"),
        new File("/home/ksen/Documents/data/TUD/lr-streams")
    );
    preprocessing.process();
  }
  
}
