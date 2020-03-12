package com.expleague.text;

import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqArray;

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

public class VGramBuilder {
    public static void main(String[] args) throws IOException {
        final String datasetPath = args[0];
        final String vgramDictionaryPath = args[1];
        final String alphabetPath = args[2];
        final int vgrams = Integer.parseInt(args[3]);

        final List<CharSeq> dataset = readDataset(Paths.get(datasetPath));
        final Path vgramFile = Paths.get(vgramDictionaryPath);
        final List<Character> alphabet = readAlphabet(Paths.get(alphabetPath));

        DictExpansion<Character> de = new DictExpansion<>(alphabet, vgrams, System.err);
        for (int i = 0; i < 40; i++) {
            List<CharSeq> nextIter = new ArrayList<>(dataset);
            Collections.shuffle(nextIter);
            IntStream.range(0, dataset.size()).parallel().forEach(idx -> de.accept(nextIter.get(idx)));
            System.err.println(i + "-th iter end");

            System.err.println("writing dict to " + vgramFile);
            try (Writer writer = new OutputStreamWriter(Files.newOutputStream(vgramFile), StandardCharsets.UTF_16)) {
                de.print(writer);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.err.println("END");
    }

    private static List<CharSeq> readDataset(final Path datasetPath) throws IOException {
        long start = System.nanoTime();
        List<CharSeq> dataset = Files.readAllLines(datasetPath)
                .stream()
                .map(String::toLowerCase)
                .map(str -> str.replaceAll("ё", "е"))
                .map(str -> str.replaceAll("[^а-я -]", " "))
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
