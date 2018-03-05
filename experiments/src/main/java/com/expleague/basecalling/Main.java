package com.expleague.basecalling;

import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Paths;

public class Main {
  private static Options options = new Options();
  static {
    options.addOption("method", true, "method");
    options.addOption("stateCount", true, "stateCount");
    options.addOption("alphaShrink", true, "alphaShrink");
    options.addOption("lambda", true, "lambda");
    options.addOption("addToDiag", true, "addToDiad");
    options.addOption("boostStep", true, "boostStep");
    options.addOption("trainPart", true, "trainPart");
    options.addOption("testPart", true, "testPart");
    options.addOption("dataset", true, "datasetPath");
    options.addOption("checkpointFolder", true, "checkpointFolder");
  }

  public static void main(String[] args) throws IOException, ParseException {
//    BasecallingDataset basecallingDataset = new BasecallingDataset();
//    basecallingDataset.prepareData(Paths.get("./dataset.txt"), Paths.get("rel3/chrM/part01/"),
//        200);

    final CommandLineParser parser = new GnuParser();
    final CommandLine command = parser.parse(options, args);

    PNFABasecall basecall = new PNFABasecall(
        Paths.get(command.getOptionValue("dataset")),
        Paths.get(command.getOptionValue("checkpointFolder")),
        Integer.parseInt(command.getOptionValue("stateCount")),
        Integer.parseInt(command.getOptionValue("alphaShrink")),
        Double.parseDouble(command.getOptionValue("lambda")),
        Double.parseDouble(command.getOptionValue("addToDiag")),
        Double.parseDouble(command.getOptionValue("boostStep")),
        Double.parseDouble(command.getOptionValue("trainPart")),
        Double.parseDouble(command.getOptionValue("testPart")),
        239,
        false
    );
    final String method = command.getOptionValue("method");
    if (method.equals("gradFac")) {
      basecall.trainGradFac();
    } else {
      basecall.tranVecPNFA();
    }
  }
}
