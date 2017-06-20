package com.spbsu.direct;

import com.spbsu.commons.func.Action;
import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.seq.CharSeq;
import java.io.FileWriter;

public class DirectDictPrinter implements Action<DictExpansion<CharSeq>> {
    private int dictionaryIndex;
    private final String outputFile;

    public DirectDictPrinter(final String outputFile) {
        this.outputFile = outputFile;
    }

    @Override
    public void invoke(DictExpansion<CharSeq> result) {
        try {
            System.out.println(String.format("Dump dictionary #%d", dictionaryIndex));
            result.print(new FileWriter(StreamTools.stripExtension(outputFile) + "-" + dictionaryIndex + ".dict"));
            dictionaryIndex++;

            // TODO
            BroadMatch.windex = 0;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
