package com.expleague.text;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqArray;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class VGramBuilder {
    public static void main(String[] args) throws IOException {
        Options options = new Options();

        Option inputDatasetOption = new Option("i", "input-dataset-path", true, "path to the dataset");
        inputDatasetOption.setRequired(true);
        options.addOption(inputDatasetOption);

        Option outputOption = new Option("o", "output-path", true, "path to vgram dictionary");
        outputOption.setRequired(true);
        options.addOption(outputOption);

        Option alphabetOption = new Option("a", "alphabet-path", true, "path to the alphabet");
        alphabetOption.setRequired(true);
        options.addOption(alphabetOption);

        Option vgramsOption = new Option("n", "vgrams-count", true, "number of vgrams to fit");
        vgramsOption.setRequired(false);
        vgramsOption.setType(Integer.class);
        options.addOption(vgramsOption);

        Option iterationsOption = new Option("it", "iterations", true, "number of iterations to fit");
        iterationsOption.setRequired(false);
        iterationsOption.setType(Integer.class);
        options.addOption(iterationsOption);

        Option filterOption = new Option("f", "filter", false, "filter dataset symbols");
        filterOption.setRequired(false);
        options.addOption(filterOption);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd = null;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            System.out.println(e.getMessage());
            formatter.printHelp("vgram", options);
            System.exit(1);
        }

        final String datasetPath = cmd.getOptionValue("input-dataset-path");
        final String vgramDictionaryPath = cmd.getOptionValue("output-path");
        final String alphabetPath = cmd.getOptionValue("alphabet-path");
        final int vgrams = Integer.parseInt(cmd.getOptionValue("vgrams-count", "15000"));
        final int iterations = Integer.parseInt(cmd.getOptionValue("iterations", "50"));
        final boolean filter = cmd.hasOption("filter");

        final List<CharSeq> dataset = readDataset(Paths.get(datasetPath), filter);
        final Path vgramFile = Paths.get(vgramDictionaryPath);
        final List<Character> alphabet = readAlphabet(Paths.get(alphabetPath));

        DictExpansion<Character> de = new DictExpansion<>(alphabet, vgrams, System.err);
        for (int i = 0; i < iterations; i++) {
            List<CharSeq> nextIter = new ArrayList<>(dataset);
            Collections.shuffle(nextIter);
            IntStream.range(0, dataset.size()).parallel().forEach(idx -> de.accept(nextIter.get(idx)));
            System.err.println(i + "-th iter end");

            System.err.println("writing dict to " + vgramFile);
            try (Writer writer = new OutputStreamWriter(Files.newOutputStream(vgramFile), StandardCharsets.UTF_8)) {
                de.print(writer);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.err.println("END");
    }

    private static List<CharSeq> readDataset(final Path datasetPath, boolean filterDataset) throws IOException {
        long start = System.nanoTime();
        Stream<String> datasetStream = Files
                .readAllLines(datasetPath)
                .stream()
                .map(String::toLowerCase);

        if (filterDataset) {
            datasetStream = datasetStream
                    .map(str -> str.replaceAll("ё", "е"))
                    .map(str -> str.replaceAll("[^а-я -]", " "));
        }

        List<CharSeq> dataset = datasetStream
                .map(str -> str.replaceAll("\\s+", " "))
                .filter(str -> str.length() >= 5)
                .map(str -> new CharSeqArray(str.toCharArray()))
                .collect(Collectors.toList());

        System.err.printf("Data from " + datasetPath.toAbsolutePath() + " read in %.2f minutes\n", (System.nanoTime() - start) / 60e9);
        return dataset;
    }

    private static List<Character> readAlphabet(final Path alphabetPath) throws IOException {
        return Files.readAllLines(alphabetPath)
                .stream()
                .map(s -> s.charAt(0))
                .sorted()
                .collect(Collectors.toList());
    }
}
