package com.spbsu.ml;

import com.spbsu.commons.random.FastRandom;
import com.spbsu.ml.data.tools.DataTools;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by towelenee on 5/26/15.
 */
public class LetorTests extends MethodsTests {
  @Override
  protected void setUp() throws IOException {
    learn = DataTools.loadLetorFile("file", new FileReader("/home/towelenee/pool/Fold1/train.txt"));
    validate = DataTools.loadLetorFile("file", new FileReader("/home/towelenee/pool/Fold1/vali.txt"));
    rng = new FastRandom(0);
  }

}
