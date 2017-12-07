package com.expleague.basecalling;

import com.expleague.commons.util.ArrayTools;
import com.expleague.commons.util.Pair;
import ncsa.hdf.object.Attribute;
import ncsa.hdf.object.HObject;
import ncsa.hdf.object.ScalarDS;
import ncsa.hdf.object.h5.H5File;
import ncsa.hdf.object.h5.H5Group;
import ncsa.hdf.object.h5.H5ScalarDS;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BasecallingDataset {
  private static final int NUM_OF_LEVELS = 8192;
  private static final int PREFIX_CONTEXT_LENGTH = 40;
  private static final int SUFFIX_CONTEXT_LENGTH = 40;

  public void prepareData(final Path dataSetFilePath, final Path readsPath, final int filesNum) throws IOException {
    final File dataSetFile = dataSetFilePath.toFile();
    dataSetFile.createNewFile();
    try (FileWriter writer = new FileWriter(dataSetFile)) {
      Files.list(readsPath).limit(filesNum).forEach(filePath -> {
        final H5File h5File = new H5File(filePath.toString(), H5File.READ);

        try {
          // parse nanopore read file
          final HObject metaData = h5File.get("UniqueGlobalKey/channel_id");
          final double samplingRate = getDoubleAttribute(metaData,"sampling_rate");
          final double range = getDoubleAttribute(metaData,"range");
          final int numLevels = (int) getDoubleAttribute(metaData, "digitisation");
          if (numLevels != NUM_OF_LEVELS) {
            throw new IllegalArgumentException("Unexpected number of levels: " + numLevels + ", Wexpected " + NUM_OF_LEVELS);
          }
          final double offset = getDoubleAttribute(metaData,"offset");

          final H5Group readsGroup = (H5Group) ((H5Group) h5File.get("Raw/Reads")).getMemberList().get(0);
          // Processing start time. Raw signal sequence starts from this time
          final long startTime = getLongAttribute(readsGroup,"start_time");
          final long duration = getIntAttribute(readsGroup,"duration");
          final ScalarDS rawSignalDataSet = new H5ScalarDS(h5File, "Signal", readsGroup.getFullName());
          rawSignalDataSet.init();

          final int[] rawSignal = shortArrayToIntArray((short []) rawSignalDataSet.getData());

          final List<SquashedNanoporeEvent> events = getSquashedEvents(filePath, samplingRate);

          System.err.println("Mean length of event: " + 1.0 * events.stream().mapToInt(it -> it.length).sum() / events.size());
          System.err.println("Median length of event: " + 1.0 * events.stream().mapToInt(it -> it
              .length).sorted().skip(events.size() / 2).findFirst().orElse(0));
          System.err.println("Max length of event: " + events.stream().mapToInt(it -> it.length).max().orElse(0));

          final String baseCalledNucleotides = getNucleotides(filePath);
          final Pair<String, String> cigarStringAndSeq = getCigarStringAndSeq(filePath);
          final String cigarString = cigarStringAndSeq.first;

          checkNucleotides(cigarStringAndSeq.second, baseCalledNucleotides);

          int cigarPos = 0;
          int eventPos = 0;

          // now I have a CIGAR string and a sequence of squashed events and I have to extract events that predicted
          // nucleotide correctly
          while (cigarPos < cigarString.length()) {
            // example of CIGAR string: 5S20M10I
            // it means that the first 5 nucleotides in our DNA sequence are soft-clipped,
            // the following 20 are matched against reference
            // and following 10 are extra nucleotides and should be removed from our sequence
            // CIGAR specification: http://samtools.github.io/hts-specs/SAMv1.pdf
            int cigarNextPos = cigarPos;
            while (Character.isDigit(cigarString.charAt(cigarNextPos))) {
              cigarNextPos++;
            }
            // number of nucleotides to which operation applies
            int cnt = Integer.parseInt(cigarString.substring(cigarPos, cigarNextPos));
            // type of operation
            char type = Character.toUpperCase(cigarString.charAt(cigarNextPos));
            cigarPos = cigarNextPos + 1;

            if (!Character.isLetter(type)) {
              throw new IllegalArgumentException("Invalid cigar character " + type);
            }

            // only these operations (aside from exact match) result in advancement in our DNA sequence
            if (type == 'I' || type == 'S') {
              eventPos += cnt;
              continue;
            }
            // we are interested only in matches
            else if (type != 'M') {
              continue;
            }

            // IMPORTANT!
            // I assume that the first nucleotide of the first event state corresponds to the
            // first nucleotide in the DNA sequence from reads
            // Probably should add some sanity checks there

            for (int i = 0; i < cnt; i++) {
              // sometimes I get OutOfBoundsException there
              final SquashedNanoporeEvent event = events.get(eventPos);
              final int signalStartPos = (int) (event.startTime - startTime);

              // extracting raw signal corresponding to a given event with context
              if (event.length != 0
                  && signalStartPos - PREFIX_CONTEXT_LENGTH >= 0
                  && signalStartPos + SUFFIX_CONTEXT_LENGTH < rawSignal.length) {
                final double meanSignalLevel = ArrayTools.sum(
                    rawSignal, signalStartPos, signalStartPos + event.length
                ) / ((double) event.length);

                final double meanEventFromLevel = (meanSignalLevel + offset) / NUM_OF_LEVELS * range;

                // some sanity check that mean in event is the same as mean in the raw signal
                // sometimes it's not true, about 10 times for a read in average
                // don't know why this happens, should not be important
                if (Math.abs(meanEventFromLevel - event.mean) > 1e-4) {
                  System.out.println("Mean event (" + event.mean + ") is not equal to mean from level ("
                      + meanEventFromLevel + ") at signal position " + signalStartPos
                      + " and event position " + eventPos
                  );
                }

                final int[] signal = Arrays.copyOfRange(
                      rawSignal, signalStartPos - PREFIX_CONTEXT_LENGTH, signalStartPos + SUFFIX_CONTEXT_LENGTH
                );
                writer
//                    .append(filePath.getFileName().toString()).append("-")
//                    .append(String.valueOf(event.startTime)).append(" ")
                    .append(baseCalledNucleotides.substring(eventPos, eventPos + 1)).append(" ")
                    .append(Arrays.stream(signal)
                        .mapToObj(Integer::toString)
                        .collect(Collectors.joining(","))
                    ).append("\n");
              }
              eventPos++;
            }
          }
          // Usually they differ by 5 that seems right
          System.out.println("Total squashed events: " + events.size() + ", eventPos after data extraction: " + eventPos);
        } catch (Exception e) {
          e.printStackTrace();
        }
      });
    }
  }

  private int[] shortArrayToIntArray(short[] data) {
    int[] array = new int[data.length];
    for (int i = 0; i < data.length; i++) {
      array[i] = data[i];
    }
    return array;
  }

  /**
   * Simple sanity check that sequence in alignment and sequence in read are the same
   * up to reversing and complementing
   * @param alignSeq DNA sequence from alignment
   * @param baseCalledNucleotides DNA sequence from reads
   */
  private void checkNucleotides(String alignSeq, String baseCalledNucleotides) {
    final Map<Character, Character> complementMap = new HashMap<>();
    complementMap.put('A', 'T');
    complementMap.put('T', 'A');
    complementMap.put('C', 'G');
    complementMap.put('G', 'C');

    if (alignSeq.length() != baseCalledNucleotides.length()) {
      throw new IllegalArgumentException("Lengths of cigar seq and string from poretools differ: " +
          baseCalledNucleotides.length() + " and " + alignSeq.length());
    }
    if (!alignSeq.equals(baseCalledNucleotides)) {
      for (int i = 0; i < alignSeq.length(); i++) {
        char c1 = alignSeq.charAt(i);
        char c2 = baseCalledNucleotides.charAt(alignSeq.length() - i - 1);
        if (c2 != complementMap.get(c1)) {
          throw new IllegalArgumentException("Cigar seq and string from poretools aren't equal");
        }
      }
    }

  }

  private String getNucleotides(Path readsPath) throws IOException, InterruptedException {
    final String command = "poretools fasta " + readsPath.toString() + " >reads.fasta";
    execCommand(command);
    return Files.readAllLines(Paths.get("reads.fasta")).get(1);
  }

  /**
   *
   * @param readsPath path to reads file
   * @return aligns reads against reference genome and returns pair of
   * (CIGAR string, DNA string extracted from reads, possibly reversed and complemented)
   */
  private Pair<String, String> getCigarStringAndSeq(Path readsPath) throws IOException, InterruptedException {
    final String command = "poretools fasta " + readsPath.toString() + " >reads.fasta && " +
        "minimap2 -ax map-ont genome.idx reads.fasta | samtools view - > align.sam" ;
    execCommand(command);
    final String[] tokens = Files.readAllLines(Paths.get("align.sam")).get(0).split("\t");

    if (tokens.length < 11) {
      throw new IllegalArgumentException("Found " + tokens.length + " tokens instead of 11");
    }

    String cigarString = tokens[5];
    String dnaString = tokens[9];
    return new Pair<>(cigarString, dnaString);
  }

  private List<SquashedNanoporeEvent> getSquashedEvents(Path readsPath, double samplingRate) throws IOException, InterruptedException {
    final String command = "poretools events " + readsPath.toString() + " >events.txt";
    execCommand(command);
    final List<String> events = Files.readAllLines(Paths.get("events.txt"));
    final String eventsHeadline = events.get(0);
    List<NanoporeEvent> nanoporeEvents = events
        .stream()
        .skip(1)
        .map(s -> new NanoporeEvent(s, eventsHeadline, samplingRate)).collect(Collectors.toList());
    List<SquashedNanoporeEvent> squashedEvents = new ArrayList<>();
    int eventBeginPos = 0;
    double mean = 0;
    int length = 0;
    for (int i = 0; i < nanoporeEvents.size(); i++) {
      final NanoporeEvent event = nanoporeEvents.get(i);
      mean += event.mean * event.length;
      length += event.length;
      if (event.move == 0) continue;
      squashedEvents.add(new SquashedNanoporeEvent(nanoporeEvents.get(eventBeginPos).startTime, length, mean / length));
      // there are events which has `move` greater that 1.
      // It means that model state has shifted by more than one character
      for (int j = 1; j < event.move; j++) {
        squashedEvents.add(new SquashedNanoporeEvent(0, 0, 0));
      }
      eventBeginPos = i + 1;
      length = 0;
      mean = 0;
    }
    return squashedEvents;
  }

  /**
   * Single Nanopore event from `poretools events`
   */
  private class NanoporeEvent {
    long startTime;
    int length;
    double mean;
    int move;

    NanoporeEvent(final String event, final String eventsHeadline, final double samplingRate) {
      final List<String> eventsTokens = Arrays.asList(event.split("\t"));
      final List<String> headlineTokens = Arrays.asList(eventsHeadline.split("\t"));
      Function<String, Double> getDouble = name -> Double.parseDouble(eventsTokens.get(headlineTokens.indexOf(name)));

      startTime = Math.round(getDouble.apply("start") * samplingRate);
      length = (int) Math.round(getDouble.apply("length") * samplingRate);
      mean = getDouble.apply("mean");
      move = (int) Math.round(getDouble.apply("move"));
    }
  }

  /**
   * Represents a continuous sequence of Nanopore events corresponding to the same state
   */
  private class SquashedNanoporeEvent {
    long startTime;
    int length; // if length is 0 then event was not present (move was >= 2)
    double mean; // for debug

    public SquashedNanoporeEvent(long startTime, int length, double mean) {
      this.startTime = startTime;
      this.length = length;
      this.mean = mean;
    }
  }

  @SuppressWarnings("unchecked")
  private Object getAttribute(HObject obj, String attributeName) throws Exception {
    return ((Stream<Attribute>) obj.getMetadata().stream())
        .filter(attr -> attr.getName().equals(attributeName))
        .findFirst().get().getValue();
  }

  private double getDoubleAttribute(HObject obj, String attributeName) throws Exception {
    return ((double[]) getAttribute(obj, attributeName))[0];
  }

  private long getLongAttribute(HObject obj, String attributeName) throws Exception {
    return ((long[]) getAttribute(obj, attributeName))[0];
  }

  private int getIntAttribute(HObject obj, String attributeName) throws Exception {
    return ((int[]) getAttribute(obj, attributeName))[0];
  }

  private void execCommand(final String command) throws IOException, InterruptedException {
    try (FileWriter writer = new FileWriter(new File("tmp.sh"))) {
      writer.write(command);
    }
    System.err.println("Executing " + command);
    Runtime.getRuntime().exec("bash tmp.sh").waitFor();

  }
}
