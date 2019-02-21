package com.expleague.embedding;

import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.glove.GloVeBuilder;
import org.apache.commons.cli.*;

import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;

public class Glove {
  private static Options options = new Options();

  static {
    options.addOption(Option.builder()
            .longOpt("corpus_path")
            .desc("Path to the corpus of texts")
            .hasArg()
            .argName("CORPUS")
            .type(String.class)
            .required()
            .build());
    options.addOption(Option.builder()
            .longOpt("model_path")
            .desc("Path to the model to be built")
            .hasArg()
            .argName("MODEL")
            .type(String.class)
            .required()
            .build());
    options.addOption(Option.builder()
            .longOpt("dim")
            .desc("Dimension of vectors")
            .hasArg()
            .argName("DIMENSION")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("min_freq")
            .desc("Minimum frequency of words")
            .hasArg()
            .argName("MIN_FREQ")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("epochs")
            .desc("Number of epochs")
            .hasArg()
            .argName("EPOCHS")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("step")
            .desc("Learning rate")
            .hasArg()
            .argName("STEP")
            .type(Number.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("window_left")
            .desc("Number of words to the left of the current")
            .hasArg()
            .argName("WINDOW_LEFT")
            .type(Integer.class)
            .build());
    options.addOption(Option.builder()
            .longOpt("window_right")
            .desc("Number of words to the right of the current")
            .hasArg()
            .argName("WINDOW_RIGHT")
            .type(Integer.class)
            .build());
  }

  public static void main(String[] args) throws Exception {
    CommandLineParser parser = new DefaultParser();
    CommandLine cmd = parser.parse(options, args);

    final String corpus = cmd.getOptionValue("corpus_path");
    final String model = cmd.getOptionValue("model_path");
    final int dim = Integer.parseInt(cmd.getOptionValue("dim", "50"));
    final int minFreq = Integer.parseInt(cmd.getOptionValue("min_freq", "5"));
    final int epochs = Integer.parseInt(cmd.getOptionValue("epochs", "25"));
    final double step = Double.parseDouble(cmd.getOptionValue("step", "0.1"));
    final int windowLeft = Integer.parseInt(cmd.getOptionValue("window_left", "15"));
    final int windowRight = Integer.parseInt(cmd.getOptionValue("window_right", "15"));

    GloVeBuilder builder = (GloVeBuilder) Embedding.builder(Embedding.Type.GLOVE);
    final Embedding result = builder
            .dim(dim)
            .minWordCount(minFreq)
            .iterations(epochs)
            .step(step)
            .window(Embedding.WindowType.LINEAR, windowLeft, windowRight)
            .file(Paths.get(corpus))
            .build();
    try (Writer to = Files.newBufferedWriter(Paths.get(model))) {
      Embedding.write(result, to);
    }
  }
}
