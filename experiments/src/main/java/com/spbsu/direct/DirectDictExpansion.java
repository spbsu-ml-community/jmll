package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.seq.ArraySeq;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.commons.util.ThreadTools;

import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;

import static com.spbsu.direct.Utils.convertToSeq;
import static com.spbsu.direct.Utils.normalizeQuery;


public class DirectDictExpansion {
    private final static int QUEUE_SIZE = 100_000;

    private final String outputFile;
    private final DictExpansion<CharSeq> expansion;
    private final ThreadPoolExecutor executor = ThreadTools.createBGExecutor("Creating DictExpansion", QUEUE_SIZE);

    public DirectDictExpansion(final Integer size, final String outputFile) {
        this.outputFile = outputFile;
        this.expansion = new DictExpansion<>(size, System.out);
        this.expansion.addListener(new DirectDictPrinter(outputFile));
    }

    public void process(final List<String> files) {
        files.forEach(
                fileName -> {
                    try {
                        CharSeqTools.processLines(StreamTools.openTextFile(fileName), new Normalizer());
                    } catch (Exception e) {
                        System.err.println(String.format("Failed to process %s: %s", fileName, e.toString()));
                    }
                });
    }

    private class Normalizer implements Action<CharSequence> {
        // TODO: why do we need to store it?
        private String current;

        @Override
        public void invoke(CharSequence line) {
            final String[] parts = line.toString().split("\\t");

            if (parts.length != 3) {
                throw new IllegalArgumentException("Each input line must contain <uid>\\t<ts>\\t<query> triplet. This one: [" + line + "]@" + outputFile + ":" + BroadMatch.index + " does not.");
            }

            if (parts[0].startsWith("uu/") || parts[0].startsWith("r")) {
                return;
            }

            if (current.equals(parts[2])) {
                return;
            }

            current = parts[2];

            final Runnable item = createQueryProcessor(parts[0], parts[2]);
            final BlockingQueue<Runnable> queue = executor.getQueue();

            //noinspection Duplicates
            if (queue.remainingCapacity() == 0) {
                try {
                    queue.put(item);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            } else {
                executor.execute(item);
            }
        }

        private Runnable createQueryProcessor(final String uid, final String query) {
            return () -> {
                final String normalizedQuery = normalizeQuery(query);
                final ArraySeq<CharSeq> seq = convertToSeq(normalizedQuery);

                // TODO
                if (BroadMatch.windex++ < 10) {
                    System.out.println(String.format("%s: %s -> %s", uid, normalizedQuery, seq));
                }

                expansion.accept(seq);
            };
        }
    }
}
