package com.expleague.erc;

import com.expleague.erc.data.DataPreprocessor;
import com.expleague.erc.data.LastFmDataReader;
import com.expleague.erc.data.OneTimeDataProcessor;
import gnu.trove.map.TLongDoubleMap;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.LongFunction;
import java.util.stream.Collectors;

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
    private static final String OPT_OUT_DIR_SHORT = "o";
    private static final String OPT_OUT_DIR_LONG = "out";
    private static final String FILE_TRAIN = "train.txt";
    private static final String FILE_TEST = "test.txt";

    static {
        options.addOption(Option.builder(OPT_DATA_PATH_SHORT).longOpt(OPT_DATA_PATH_LONG).desc("Path to data").hasArg().build());
        options.addOption(Option.builder(OPT_DATA_SIZE_SHORT).longOpt(OPT_DATA_SIZE_LONG).desc("Num of lines read from data").hasArg().build());
        options.addOption(Option.builder(OPT_TRAIN_RATIO_SHORT).longOpt(OPT_TRAIN_RATIO_LONG).desc("Train data ratio to all data size").hasArg().build());
        options.addOption(Option.builder(OPT_USER_NUM_SHORT).longOpt(OPT_USER_NUM_LONG).desc("Num of users").hasArg().build());
        options.addOption(Option.builder(OPT_ITEM_NUM_SHORT).longOpt(OPT_ITEM_NUM_LONG).desc("Num of items").hasArg().build());
        options.addOption(Option.builder(OPT_TOP_SHORT).longOpt(OPT_TOP_LONG).desc("Is filter on top items").hasArg().build());
        options.addOption(Option.builder(OPT_OUT_DIR_SHORT).longOpt(OPT_OUT_DIR_LONG).desc("Output path").hasArg().build());
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
        final Path outDirPath = Paths.get(cliOptions.getOptionValue(OPT_OUT_DIR_SHORT));

        final LastFmDataReader lastFmDataReader = new LastFmDataReader();
        final List<Event> data = lastFmDataReader.readData(dataPath, dataSize);
        final DataPreprocessor preprocessor = new OneTimeDataProcessor();
//        final DataPreprocessor preprocessor = new TSRDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(data, trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);
        final List<Event> train = dataset.getTrain();
        final List<Event> test = dataset.getTest();

        final double splitTime = dataset.getTest().get(0).getTs();
        System.out.println(splitTime);

        final Path trainPath = outDirPath.resolve(FILE_TRAIN);
        final Path testPath = outDirPath.resolve(FILE_TEST);
        if (Files.isDirectory(outDirPath)) {
            Files.deleteIfExists(trainPath);
            Files.deleteIfExists(testPath);
        } else {
            Files.createDirectory(outDirPath);
        }
        final MetricsCalculator metricsCalculator = new MetricsCalculator(dataset.getTrain(), dataset.getTest(), outDirPath);
        final TLongDoubleMap trainPairSpus = metricsCalculator.pairwiseSessionsSpu(DataPreprocessor.groupToEventSeqs(train));
        final TLongDoubleMap testPairSpus = metricsCalculator.pairwiseSessionsSpu(DataPreprocessor.groupToEventSeqs(test));
        final long [] keys = Arrays.stream(trainPairSpus.keys())
                .boxed()
                .sorted(Comparator.comparingInt(Util::extractItemId).thenComparingInt(Util::extractUserId))
                .mapToLong(Long::longValue)
                .toArray();
        writeSeq(trainPath, keys, String::valueOf);
        writeSeq(testPath, keys, String::valueOf);
        writeSeq(trainPath, keys, key -> String.valueOf(trainPairSpus.get(key)));
        writeSeq(testPath, keys, key -> String.valueOf(testPairSpus.get(key)));

        Util.writeMap(Paths.get("items.txt"), lastFmDataReader.getReversedItemMap());
        Util.writeMap(Paths.get("users.txt"), lastFmDataReader.getReversedUserMap());
    }

    private static void writeSeq(Path filePath, long[] keys, LongFunction<String> keyToValue) throws IOException {
        final String strRep = Arrays.stream(keys)
                .mapToObj(keyToValue)
                .collect(Collectors.joining(" \t")) + '\n';
        Files.write(filePath, strRep.getBytes(), StandardOpenOption.APPEND, StandardOpenOption.CREATE);
    }
}
