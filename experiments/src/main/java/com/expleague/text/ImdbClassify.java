package com.expleague.text;

import com.expleague.commons.func.Functions;
import com.expleague.commons.io.codec.seq.DictExpansion;
import com.expleague.commons.io.codec.seq.ListDictionary;
import com.expleague.commons.math.Trans;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.random.FastRandom;
import com.expleague.commons.seq.*;
import com.expleague.ml.GridTools;
import com.expleague.ml.ProgressHandler;
import com.expleague.ml.ScoreCalcer;
import com.expleague.ml.data.tools.DataTools;
import com.expleague.ml.data.tools.FeatureSet;
import com.expleague.ml.data.tools.Pool;
import com.expleague.ml.data.tools.PoolFSBuilder;
import com.expleague.ml.func.Ensemble;
import com.expleague.ml.loss.AccuracyLogit;
import com.expleague.ml.loss.L2;
import com.expleague.ml.loss.L2Reg;
import com.expleague.ml.loss.LLLogit;
import com.expleague.ml.meta.FeatureMeta;
import com.expleague.ml.meta.impl.JsonDataSetMeta;
import com.expleague.ml.meta.impl.TargetMetaImpl;
import com.expleague.ml.methods.BootstrapOptimization;
import com.expleague.ml.methods.GradientBoosting;
import com.expleague.ml.methods.trees.GreedyObliviousLinearTree;
import com.expleague.ml.methods.trees.GreedyObliviousTree;
import gnu.trove.list.array.TIntArrayList;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ImdbClassify {
  public static void main(String[] args) throws IOException {
    boolean linear = true;
    final String dataPrefix = "/Users/solar/data/text";

    final String expPrefix = "/Users/solar/data/text/embedding-classify";
    final Path trainPoolPath = Paths.get(expPrefix, "/train.pool");
    final Path testPoolPath = Paths.get(expPrefix, "/test.pool");
    final FastRandom rng = new FastRandom();
    if (!Files.exists(trainPoolPath) || !Files.exists(testPoolPath)) {
      final List<CharSeq> trainPos = readData(Paths.get(dataPrefix + "/aclImdb/train/pos"));
      final List<CharSeq> trainNeg = readData(Paths.get(dataPrefix + "/aclImdb/train/neg"));
      final List<CharSeq> testPos = readData(Paths.get(dataPrefix + "/aclImdb/test/pos"));
      final List<CharSeq> testNeg = readData(Paths.get(dataPrefix + "/aclImdb/test/neg"));
      final List<CharSeq> unlabeled = readData(Paths.get(dataPrefix + "/aclImdb/train/unsup"));

      final Path allTexts = Paths.get(expPrefix + "/all.txt");
      if (!Files.exists(allTexts)) {
        try (final Writer writer = Files.newBufferedWriter(allTexts)) {
          Stream.of(trainPos, trainNeg, unlabeled, testPos, testNeg).flatMap(Collection::stream)
              .forEach(Functions.rethrow(str -> {
                writer.append(str).append("\n");
              }));
        }
      }

      final Path vgramFile = Paths.get(expPrefix + "/vgram.txt");
      if (!Files.exists(vgramFile)) {
        final List<Character> alphabet = IntStream.concat(IntStream.range('a', 'z' + 1), IntStream.of(' ')).sorted()
            .mapToObj(ch -> (char) ch).collect(Collectors.toList());
        final List<CharSeq> allText = Stream.of(trainPos, trainNeg, unlabeled, testPos, testNeg)
            .flatMap(Collection::stream)
            .collect(Collectors.toList());

        DictExpansion<Character> de = new DictExpansion<>(alphabet, 15000, System.out);
        for (int i = 0; i < 40; i++) {
          List<CharSeq> nextIter = new ArrayList<>(allText);
          Collections.shuffle(nextIter);
          IntStream.range(0, allText.size()).parallel().forEach(idx -> de.accept(nextIter.get(idx)));
          System.out.println(i + "-th iter end");
        }

        System.out.println("writing dict to " + vgramFile);
        try (Writer writer = new OutputStreamWriter(Files.newOutputStream(vgramFile), StandardCharsets.UTF_8)){
          de.print(writer);
        }
        catch (IOException ioe) {
          ioe.printStackTrace();
        }
        System.out.println("END");
      }
      final VGramBM25FeatureSet vgfs;
      {
        //noinspection unchecked
        final ListDictionary<Character> dict = (ListDictionary<Character>) new ListDictionary(
            CharSeqTools.lines(Files.newBufferedReader(vgramFile))
                .map(line -> CharSeq.create(CharSeqTools.split(line, '\t')[0]))
                .filter(str -> str.length() > 0)
                .<Seq<Character>>toArray(Seq[]::new)
        );
        final TIntArrayList freqs = new TIntArrayList();
        freqs.fill(0, dict.size(), 0);
        int[] stat = new int[]{0};
        CharSeqTools.lines(Files.newBufferedReader(allTexts)).map(seq -> {
          if (stat[0] > 10000) {
            try {
              return dict.parse(seq, freqs, stat[0]);
            }
            catch (RuntimeException re) {
              if (!re.getMessage().equals(ListDictionary.DICTIONARY_INDEX_IS_CORRUPTED))
                throw re;
            }
          }
          return dict.parse(seq);
        }).flatMapToInt(IntSeq::stream).filter(idx -> idx >= 0).forEach(idx -> {
          stat[0]++;
          freqs.set(idx, freqs.get(idx) + 1);
        });

        vgfs = new VGramBM25FeatureSet(dict, freqs);
      }

//      final Path embeddingFile = Paths.get(expPrefix + "/embedding-vec.txt");
//      if (!Files.exists(embeddingFile)) {
//        final GloVeBuilder ebuilder = new GloVeBuilder();
//        final Embedding result = ebuilder
//            .dim(40)
//            .minWordCount(1)
//            .iterations(25)
//            .step(0.01)
//            .window(Embedding.WindowType.LINEAR, 15, 15)
//            .file(allTexts)
//            .build();
//        Embedding.write(result, Files.newBufferedWriter(embeddingFile));
//      }
//      final Embedding<CharSeq> embedding = EmbeddingImpl.read(Files.newBufferedReader(embeddingFile), CharSeq.class);
//
//      final EmbeddingFeatureSet embeddingFeatureSet = new EmbeddingFeatureSet("GloVe", "Glove embedding feature", embedding);
      final int[] targetHolder = new int[]{0};
      final FeatureSet.Stub<TextItem> target = new FeatureSet.Stub<TextItem>(new TargetMetaImpl("sentiment", "Positive or negative sentiment", FeatureMeta.ValueType.INTS)) {
        @Override
        public Vec advanceTo(Vec to) {
          to.set(0, targetHolder[0]);
          return to;
        }
      };
      { // train
        PoolFSBuilder<TextItem> poolBuilder = new PoolFSBuilder<>(
            new JsonDataSetMeta("imdb-train", "solar", new Date(), TextItem.class, "imdb-train-" + rng.nextBase64String(10)),
            FeatureSet.join(
//                embeddingFeatureSet,
                vgfs,
                target
            )
        );
        targetHolder[0] = 1;
        trainPos.forEach(text -> {
          poolBuilder.accept(new TextItem(text));
          poolBuilder.advance();
        });
        targetHolder[0] = -1;
        trainNeg.forEach(text -> {
          poolBuilder.accept(new TextItem(text));
          poolBuilder.advance();
        });
        DataTools.writePoolTo(poolBuilder.create(), Files.newBufferedWriter(trainPoolPath));
      }

      { // test
        PoolFSBuilder<TextItem> poolBuilder = new PoolFSBuilder<>(
            new JsonDataSetMeta("imdb-test", "solar", new Date(), TextItem.class, "imdb-test-" + rng.nextBase64String(10)),
            FeatureSet.join(
//                embeddingFeatureSet,
                vgfs,
                target
            )
        );
        targetHolder[0] = 1;
        testPos.forEach(text -> {
          poolBuilder.accept(new TextItem(text));
          poolBuilder.advance();
        });
        targetHolder[0] = -1;
        testNeg.forEach(text -> {
          poolBuilder.accept(new TextItem(text));
          poolBuilder.advance();
        });
        DataTools.writePoolTo(poolBuilder.create(), Files.newBufferedWriter(testPoolPath));
      }
    }
    final Pool<TextItem> train = DataTools.readPoolFrom(Files.newBufferedReader(trainPoolPath));
    final Pool<TextItem> test = DataTools.readPoolFrom(Files.newBufferedReader(testPoolPath));

    final GradientBoosting<LLLogit> boosting;

    if (linear) {
      boosting = new GradientBoosting<>(
          new BootstrapOptimization<>(
              new GreedyObliviousLinearTree<>(GridTools.medianGrid(train.vecData(), 2), 6),
              rng
          ),
          L2.class, 2000, 1
      );
    }
    else {
      boosting = new GradientBoosting<>(
          new BootstrapOptimization<>(
              new GreedyObliviousTree<>(GridTools.medianGrid(train.vecData(), 2), 6),
              rng
          ),
          L2Reg.class, 2000, 1
      );
    }

    final Consumer counter = new ProgressHandler() {
      int index = 0;
      @Override
      public void accept(final Trans partial) {
        System.out.print("\n" + index++);
      }
    };
    final ScoreCalcer learnListener = new ScoreCalcer(/*"\tlearn:\t"*/"\t", train.vecData(), train.target(LLLogit.class), false);
    final ScoreCalcer validateListener = new ScoreCalcer(/*"\ttest:\t"*/"\t", test.vecData(), test.target(AccuracyLogit.class), false);
//    final Consumer<Trans> modelPrinter = new ModelPrinter();
    boosting.addListener(counter);
    boosting.addListener(learnListener);
    boosting.addListener(validateListener);
//    boosting.addListener(modelPrinter);
    final Ensemble ans = boosting.fit(train.vecData(), train.target(LLLogit.class));

    System.out.println();
  }

  private static List<CharSeq> readData(final Path dataDir) throws IOException {
    long start = System.nanoTime();
    final List<CharSeq> data = Files.list(dataDir).map(path -> {
      try {
        return new CharSeqArray(Files.readAllLines(path)
            .stream()
            .map(String::toLowerCase)
            .map(str -> str.replaceAll("\\s+", " "))
            .map(str -> str.replaceAll("<[^>]+>", ""))
            .map(str -> str.replaceAll("[^a-z ]", ""))
            .map(str -> str.replaceAll("\\s+", " "))
            .collect(Collectors.joining(" "))
            .toCharArray());
      } catch (IOException e) {
        e.printStackTrace();
        return null;
      }
    }).filter(Objects::nonNull).collect(Collectors.toList());
    System.out.printf("Data from " + dataDir.toAbsolutePath() + " read in %.2f minutes\n", (System.nanoTime() - start) / 60e9);
    return data;
  }
}
