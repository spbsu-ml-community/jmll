package com.spbsu.ml.cli.modes.impl;

import com.spbsu.commons.io.StreamTools;
import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.DynamicDictionary;
import com.spbsu.commons.seq.CharSeq;
import com.spbsu.commons.seq.CharSeqTools;
import com.spbsu.ml.cli.modes.AbstractMode;
import org.apache.commons.cli.CommandLine;

import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;

/**
 * Experts League
 * Created by solar on 11.09.17.
 */
public class CreateDictionary extends AbstractMode {
  int linesCount = 0;
  @Override
  public void run(CommandLine command) throws Exception {
    final InputStreamReader reader = new InputStreamReader(System.in, StreamTools.UTF);
    final LineNumberReader lnr = new LineNumberReader(reader);
    final String alphaStr = lnr.readLine();
    final DictExpansion<Character> result = new DictExpansion<Character>(
        new DynamicDictionary<>(CharSeq.create(alphaStr)),
        Integer.parseInt(command.getOptionValue('n', "1000"))
    );
    CharSeqTools.lines(reader, false)
        .forEach(line -> {
          linesCount++;
          result.accept(line);
          if (linesCount % 10000 == 0) {
            try {
              final FileWriter writer = new FileWriter(command.getOptionValue('o', "output.dict.temp"));
              writer.append("After ").append(Integer.toString(linesCount)).append(" lines\n");
              result.print(writer);
            }
            catch (IOException e) {
              throw new RuntimeException(e);
            }
          }
        });

    result.print(new FileWriter(command.getOptionValue('o', "output.dict")));
  }
}
