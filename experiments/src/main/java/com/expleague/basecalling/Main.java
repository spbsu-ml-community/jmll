package com.expleague.basecalling;

import com.expleague.commons.random.FastRandom;

import java.io.IOException;
import java.nio.file.Paths;

public class Main {
  public static void main(String[] args) throws IOException {
//    BasecallingDataset basecallingDataset = new BasecallingDataset();
//    basecallingDataset.prepareData(Paths.get("./dataset.txt"), Paths.get("rel3/chrM/part01/"),
//        200);

    PNFABasecall basecall = new PNFABasecall(Paths.get("./dataset.txt"), 0.04, 0.01, new
        FastRandom(239), true);
    basecall.trainOneVsRest();
  }
}
