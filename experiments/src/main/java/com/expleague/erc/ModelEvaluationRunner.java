package com.expleague.erc;

import com.expleague.commons.math.vectors.Vec;
import com.expleague.erc.data.*;
import com.expleague.erc.models.ApplicableModel;
import com.expleague.erc.models.Model;
import com.expleague.erc.models.ModelDays;
import gnu.trove.map.TIntDoubleMap;
import gnu.trove.map.TIntObjectMap;
import gnu.trove.map.hash.TIntDoubleHashMap;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

public class ModelEvaluationRunner {
    private static final String FILE_MODEL = "model";
    private static final String FILE_USER_MAP = "users_by_id.txt";
    private static final String FILE_ITEM_MAP = "items_by_id.txt";
    private static final String FILE_DELTAS = "deltas.txt";
    private static final String FILE_ITEM_EMBEDDINGS = "item_embeddings.txt";
    private static Options options = new Options();

    static {
        options.addOption(Option.builder("ds").longOpt("dataset").desc("Path to data").hasArg().build());
        options.addOption(Option.builder("s").longOpt("size").desc("Num of lines read from data").hasArg().build());
        options.addOption(Option.builder("tr").longOpt("train_ratio").desc("Train data ratio to all data size").hasArg().build());
        options.addOption(Option.builder("un").longOpt("user_num").desc("Num of users").hasArg().build());
        options.addOption(Option.builder("in").longOpt("item_num").desc("Num of items").hasArg().build());
        options.addOption(Option.builder("t").longOpt("top").desc("Is filter on top items").hasArg().build());
        options.addOption(Option.builder("mn").longOpt("model_name").desc("Name for statistics files").hasArg().build());
        options.addOption(Option.builder().longOpt("toloka").desc("Read data in Toloka format").hasArg(false).build());
        options.addOption(Option.builder("o").longOpt("operation").desc("Specify task").hasArg().build());
    }

    public static void main(String[] args) throws ParseException, IOException, ClassNotFoundException {
        final CommandLineParser parser = new DefaultParser();
        final CommandLine cliOptions = parser.parse(options, args);

//        final String dataPath = cliOptions.getOptionValue("ds", "../erc/data/lastfm/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv");
//        final int size = Integer.parseInt(cliOptions.getOptionValue("s", "1000000"));
//        final int usersNum = Integer.parseInt(cliOptions.getOptionValue("un", "1000"));
//        final int itemsNum = Integer.parseInt(cliOptions.getOptionValue("in", "1000"));
//        final double trainRatio = Double.parseDouble(cliOptions.getOptionValue("tr", "0.75"));
//        final boolean isTop = Boolean.parseBoolean(cliOptions.getOptionValue("t", "true"));
        final String modelName = cliOptions.getOptionValue("mn", "experiments/src/main/resources/com/expleague/erc/models/model");
//        final boolean toloka = cliOptions.hasOption("toloka");
        final String operation = cliOptions.getOptionValue("o");

        final Path modelDirPath = Paths.get(modelName);
        final ModelDays model = ModelDays.load(modelDirPath.resolve(FILE_MODEL));

        switch (operation) {
            case "deltas": {
                writeRealPredictedDeltas(model, modelDirPath, cliOptions);
                break;
            }
            case "embeddings": {
                writeEmbeddings(model, modelDirPath);
                break;
            }
        }
    }

    private static void writeRealPredictedDeltas(Model model, Path modelDirPath, CommandLine cliOptions)
            throws IOException {
        final String dataPath = cliOptions.getOptionValue("ds", "../erc/data/lastfm/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv");
        final int size = Integer.parseInt(cliOptions.getOptionValue("s", "1000000"));
        final int usersNum = Integer.parseInt(cliOptions.getOptionValue("un", "1000"));
        final int itemsNum = Integer.parseInt(cliOptions.getOptionValue("in", "1000"));
        final double trainRatio = Double.parseDouble(cliOptions.getOptionValue("tr", "0.75"));
        final boolean isTop = Boolean.parseBoolean(cliOptions.getOptionValue("t", "true"));
        final boolean toloka = cliOptions.hasOption("toloka");

        final BaseDataReader dataReader = toloka ? new TolokaDataReader() : new LastFmDataReader();
        final List<Event> history = dataReader.readData(dataPath, size);
        DataPreprocessor preprocessor = new OneTimeDataProcessor();
        DataPreprocessor.TrainTest dataset = preprocessor.splitTrainTest(history, trainRatio);
        dataset = preprocessor.filter(dataset, usersNum, itemsNum, isTop);

        final ApplicableModel applicable = model.getApplicable();
        final TIntDoubleMap prevTimes = new TIntDoubleHashMap();
        final List<String> deltas = new ArrayList<>();
        for (final Session session : DataPreprocessor.groupEventsToSessions(dataset.getTest())) {
            final int userId = session.userId();
            final double curTime = session.getStartTs();
            final double prevTime = prevTimes.get(userId);
            prevTimes.put(userId, curTime);
            final double actualReturnTime = session.getDelta();
            if (Util.forPrediction(session)) {
                final double expectedReturnTime = applicable.timeDelta(userId, prevTime);
                deltas.add(userId + "\t" + actualReturnTime + "\t" + expectedReturnTime);
            }
            applicable.accept(session);
        }
        Files.write(modelDirPath.resolve(FILE_DELTAS), String.join("\n", deltas).getBytes());
    }

    private static void writeEmbeddings(Model model, Path modelDirPath) throws IOException {
        final TIntObjectMap<Vec> itemEmbeddings = model.getItemEmbeddings();
        final String embeddingsStr = Files.lines(modelDirPath.resolve(FILE_ITEM_MAP))
                .map(line -> {
                    final String[] desc = line.split("\t");
                    final int number = Integer.parseInt(desc[0]);
                    final String name = desc[1];
                    final Vec embedding = itemEmbeddings.get(number);
                    if (embedding == null) {
                        return null;
                    }
                    return name + "\t" + embedding.stream()
                            .mapToObj(String::valueOf)
                            .collect(Collectors.joining(" "));
                })
                .filter(Objects::nonNull)
                .collect(Collectors.joining("\n"));

        Files.write(modelDirPath.resolve(FILE_ITEM_EMBEDDINGS), embeddingsStr.getBytes());
    }
}
