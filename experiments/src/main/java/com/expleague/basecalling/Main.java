package com.expleague.basecalling;

import java.io.IOException;
import java.nio.file.Paths;

public class Main {
  public static void main(String[] args) throws IOException {
    BasecallingDataset basecallingDataset = new BasecallingDataset();
    basecallingDataset.prepareData(Paths.get("./dataset.txt"), Paths.get("rel3/chrM/part01/"), 1000);

//    PNFABasecall basecall = new PNFABasecall(Paths.get("./dataset.txt"), 0.05, 0.02, new FastRandom(239));
//    basecall.train();
  }
}
