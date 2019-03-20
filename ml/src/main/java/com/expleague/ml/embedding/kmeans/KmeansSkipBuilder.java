package com.expleague.ml.embedding.kmeans;

import com.expleague.commons.math.vectors.Mx;
import com.expleague.commons.math.vectors.Vec;
import com.expleague.commons.math.vectors.VecTools;
import com.expleague.commons.math.vectors.impl.mx.VecBasedMx;
import com.expleague.commons.seq.CharSeq;
import com.expleague.commons.seq.CharSeqTools;
import com.expleague.commons.seq.LongSeq;
import com.expleague.ml.embedding.Embedding;
import com.expleague.ml.embedding.impl.EmbeddingBuilderBase;
import gnu.trove.list.TLongList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.list.array.TIntArrayList;
import gnu.trove.list.array.TLongArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;

public class KmeansSkipBuilder extends EmbeddingBuilderBase {
  private int dim = 50;
  private int clustersNumber = 20;
  TIntIntMap centIds = new TIntIntHashMap(); // номер id вектора : номер центроида
  Mx centroids = null;
  Mx residuals = null; // смещение всех векторов относительно центроидов, для центроидов - нули.
  TIntArrayList vec2centr = new TIntArrayList(); // принадлженость центроиду (по номеру центроида)
  TIntArrayList centroidsSizes = new TIntArrayList(); // размеры кластеров (по номеру центроида)

  public KmeansSkipBuilder dim(int dim) {
    this.dim = dim;
    return this;
  }

  public KmeansSkipBuilder clustersNumber(int num) {
    clustersNumber = num;
    return this;
  }

  @Override
  protected boolean isCoocNecessery() {
    return false;
  }

  @Override
  protected Embedding<CharSeq> fit() {
    initialize();

    log.info("Started scanning corpus." + this.path);
    final long startTime = System.nanoTime();
    final Lock[] rowLocks = IntStream.range(0, wordsList.size()).mapToObj(i -> new ReentrantLock()).toArray(Lock[]::new);
    final List<TLongList> accumulators = new ArrayList<>();
    final CharSeq newLine = CharSeq.create("777newline777");
    wordsIndex.put(newLine, Integer.MAX_VALUE);
    try {
      source().peek(new Consumer<CharSeq>() {
        long line = 0;
        long time = System.nanoTime();
        @Override
        public synchronized void accept(CharSeq l) {
          if ((++line) % 10000 == 0) {
            log.info(line + " lines processed for " + TimeUnit.NANOSECONDS.toSeconds(System.nanoTime() - time) + "s");
            time = System.nanoTime();
          }
        }
      }).map(line -> (CharSeq)CharSeqTools.concat(line, " ", newLine)).flatMap(CharSeqTools::words).map(this::normalize).mapToInt(wordsIndex::get).filter(idx -> idx >= 0).mapToObj(new IntFunction<LongStream>() {
        final TIntArrayList queue = new TIntArrayList(1000);
        int offset = 0;
        @Override
        public synchronized LongStream apply(int idx) {
          if (idx == Integer.MAX_VALUE) { // new line
            queue.resetQuick();
            offset = 0;
            return LongStream.empty();
          }
          int pos = queue.size();
          final long[] out = new long[windowLeft + windowRight];
          int outIndex = 0;
          for (int i = offset; i < pos; i++) {
            byte distance = (byte)(pos - i);
            if (distance == 0) {
              log.warn("Zero distance occured! pos: " + pos + " i: " + i);
              System.err.println("Zero distance occured! pos: " + pos + " i: " + i);
            }
            if (distance <= windowRight)
              out[outIndex++] = pack(queue.getQuick(i), idx, distance);
            if (distance <= windowLeft)
              out[outIndex++] = pack(idx, queue.getQuick(i), (byte)-distance);
          }
          queue.add(idx);
          if (queue.size() > Math.max(windowLeft, windowRight)) {
            offset++;
            if (offset > 1000 - Math.max(windowLeft, windowRight)) {
              queue.remove(0, offset);
              offset = 0;
            }
          }
          return Arrays.stream(out, 0, outIndex);
        }
      }).flatMapToLong(entries -> entries).forEach(tuple -> {
        move(unpackA(tuple), unpackB(tuple), unpackDist(tuple));
      });
    }
    catch (IOException e) {
      throw new RuntimeException("Error in source function occured\n" + e.getMessage());
    }

    return null;
  }

  private void move(int i, int j, int dist) {
    dist = dist > 126 ? -256 + dist : dist; // нормальное расстояние
    final int ciId = vec2centr.get(i);
    final int cjId = vec2centr.get(j);
    final Vec ci = centroids.row(ciId);
    final Vec cj = centroids.row(cjId);
    final Vec di = residuals.row(i);
    final Vec dj = residuals.row(j);
    final Vec vi = VecTools.sum(ci, di);
    final Vec vj = VecTools.sum(cj, dj);

    final double e = Math.exp(VecTools.multiply(vi, vj));
    double s = e;
    for (int id : centIds.keys()) {
      final int c = centIds.get(id);
      s += centroidsSizes.get(c) * Math.exp(VecTools.multiply(centroids.row(c), vi));
    }

    for (int k = 0; k < dim; k++) {
      /*double csk = 0;
      for (int id : centIds.keys()) {
        final int c = centIds.get(id);
        csk += centroids.get(c, k) * exps.get(c);
      }

      // moving residuals
      if (!centIds.containsKey(i)) {
        di.adjust(k, -step() * move_delta(vi.get(k), e, s, csk));
      }
      if (!centIds.containsKey(j)) {
        dj.adjust(k, -step() * move_delta(vj.get(k), e, s, 0d));
      }*/

      // moving centroids
      double dci = centroidsSizes.get(ciId) * (2 * ci.get(k) + di.get(k)) * Math.exp(VecTools.multiply(ci, vi));
      if (ciId == cjId) {
        final double v = vi.get(k) + vj.get(k);
        ci.adjust(k, -step() * (v - (dci + e * v) / s));
      } else {
        ci.adjust(k, -step() * (vi.get(k) - (dci + e * vi.get(k)) / s));
        final double v = vi.get(k);
        ci.adjust(k, -step() * (v - (centroidsSizes.get(cjId) * v * Math.exp(VecTools.multiply(cj, vi)) + e * v) / s));
      }

      // updating residuals
      di.set(k, vi.get(k) - ci.get(k));
      dj.set(k, vj.get(k) - cj.get(k));


      // TODO: update NeighbourhoodGraph
    }

    System.out.println(wordsList.get(i));
    System.out.println(wordsList.get(j));
    System.out.println(dist);
  }

  private double move_delta(double v, double e, double s, double cs) {
    return v - (cs + e * v) / s;
  }

  protected Stream<CharSeq> source() throws IOException {
    if (path.getFileName().toString().endsWith(".gz"))
      return CharSeqTools.lines(new InputStreamReader(new GZIPInputStream(Files.newInputStream(path)), StandardCharsets.UTF_8));
    return CharSeqTools.lines(Files.newBufferedReader(path));
  }

  private void initialize() {
    // TODO: initialize centIds, vec2centr, centroidsSize

    final int voc_size = dict().size();
    residuals = new VecBasedMx(voc_size, dim);
    for (int i = 0; i < voc_size; i++) {
      if (centIds.containsKey(i)) {
        for (int j = 0; j < dim; j++) {
          residuals.set(i, j, 0d);
        }
        continue;
      }
      for (int j = 0; j < dim; j++) {
        residuals.set(i, j, initializeValue());
      }
    }

    centroids = new VecBasedMx(clustersNumber, dim);
    for (int i = 0; i < clustersNumber; i++) {
      for (int j = 0; j < dim; j++) {
        residuals.set(i, j, initializeValue());
      }
    }

  }

  private double initializeValue() {
    return (Math.random() - 0.5) / dim;
  }
}
