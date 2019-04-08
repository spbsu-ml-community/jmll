package com.expleague.erc;

import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.data.LastFmDataReader;
import com.expleague.erc.data.OneTimeDataProcessor;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class DataMetrics {
    private static final Options options = new Options();
    private static final String OPT_DATA_PATH_LONG = "dataset";
    private static final String OPT_DATA_PATH_SHORT = "ds";
    private static final String OPT_DATA_SIZE_LONG = "size";
    private static final String OPT_DATA_SIZE_SHORT = "s";
    private static final String OPT_TRAIN_RATIO_LONG = "train_ratio";
    private static final String OPT_TRAIN_RATIO_SHORT = "tr";
    private static final String OPT_USER_NUM_LONG = "user_num";
    private static final String OPT_USER_NUM_SHORT = "un";
    private static final String OPT_ITEM_NUM_LONG = "item_num";
    private static final String OPT_ITEM_NUM_SHORT = "in";
    private static final String OPT_TOP_LONG = "top";
    private static final String OPT_TOP_SHORT = "t";
    private static final String OPT_OUT_SHORT = "o";
    private static final String OPT_OUT_LONG = "out";

    static {
        options.addOption(Option.builder(OPT_DATA_PATH_SHORT).longOpt(OPT_DATA_PATH_LONG).desc("Path to data").hasArg().build());
        options.addOption(Option.builder(OPT_DATA_SIZE_SHORT).longOpt(OPT_DATA_SIZE_LONG).desc("Num of lines read from data").hasArg().build());
        options.addOption(Option.builder(OPT_TRAIN_RATIO_SHORT).longOpt(OPT_TRAIN_RATIO_LONG).desc("Train data ratio to all data size").hasArg().build());
        options.addOption(Option.builder(OPT_USER_NUM_SHORT).longOpt(OPT_USER_NUM_LONG).desc("Num of users").hasArg().build());
        options.addOption(Option.builder(OPT_ITEM_NUM_SHORT).longOpt(OPT_ITEM_NUM_LONG).desc("Num of items").hasArg().build());
        options.addOption(Option.builder(OPT_TOP_SHORT).longOpt(OPT_TOP_LONG).desc("Is filter on top items").hasArg().build());
        options.addOption(Option.builder(OPT_OUT_SHORT).longOpt(OPT_OUT_LONG).desc("Output path").hasArg().build());
    }

    public static void main(String[] args) throws ParseException, IOException {
        final CommandLineParser parser = new DefaultParser();
        final CommandLine cliOptions = parser.parse(options, args);

        final String dataPath = cliOptions.getOptionValue(OPT_DATA_PATH_SHORT);
        final int dataSize = Integer.parseInt(cliOptions.getOptionValue(OPT_DATA_SIZE_SHORT));
        final int usersNum = Integer.parseInt(cliOptions.getOptionValue(OPT_USER_NUM_SHORT));
        final int itemsNum = Integer.parseInt(cliOptions.getOptionValue(OPT_ITEM_NUM_SHORT));
        final boolean isTop = Boolean.parseBoolean(cliOptions.getOptionValue(OPT_TOP_SHORT));
        final double trainRatio = Double.parseDouble(cliOptions.getOptionValue(OPT_TRAIN_RATIO_SHORT));
        final Path outPath = Paths.get(cliOptions.getOptionValue(OPT_OUT_SHORT));

        final LastFmDataReader lastFmDataReader = new LastFmDataReader();
        final List<Event> data = lastFmDataReader.readData(dataPath, dataSize);
        final DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(data, trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);

        final double startTime = dataset.getTrain().get(0).getTs();
        final double splitTime = dataset.getTest().get(0).getTs();
        final double endTime = dataset.getTest().get(dataset.getTest().size() - 1).getTs();
        System.out.println((endTime - splitTime) / (splitTime - startTime));

        Files.deleteIfExists(outPath);
        final MetricsCalculator metricsCalculator = new MetricsCalculator(dataset.getTrain(), dataset.getTest(), outPath);
        metricsCalculator.writeTrainSpus();
        metricsCalculator.writeTargetSpus();
    }
}
