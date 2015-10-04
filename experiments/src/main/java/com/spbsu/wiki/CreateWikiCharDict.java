package com.spbsu.wiki;

import com.spbsu.commons.io.codec.seq.DictExpansion;
import com.spbsu.commons.io.codec.seq.Dictionary;
import com.spbsu.commons.seq.CharSeqAdapter;
import com.spbsu.commons.util.ThreadTools;
import se.lth.cs.nlp.mediawiki.model.WikipediaPage;
import se.lth.cs.nlp.mediawiki.parser.SinglestreamXmlDumpParser;
import se.lth.cs.nlp.pipeline.Filter;
import se.lth.cs.nlp.pipeline.PipelineBuilder;
import se.lth.cs.nlp.wikipedia.lang.RuConfig;
import se.lth.cs.nlp.wikipedia.parser.SwebleWikimarkupToText;

import javax.xml.parsers.SAXParserFactory;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.zip.GZIPInputStream;

/**
 * User: solar
 * Date: 01.10.15
 * Time: 12:52
 */
public class CreateWikiCharDict {
  public static void main(String[] args) throws Exception {
    //noinspection unchecked
    final DictExpansion<Character> expansion = new DictExpansion<>((Dictionary<Character>)Dictionary.EMPTY, 10000, System.out);
    final String fileName = args[0];
    final SAXParserFactory factory = SAXParserFactory.newInstance();
    factory.setValidating(false);
    final File output = new File(fileName.substring(0, fileName.lastIndexOf(".")) + ".dict");
//    final DictExpansion<Character> expansion = new DictExpansion<>(new HashSet<>(Arrays.asList('a')), 1000, System.out);
//    for (int i = 0; i < 1000; i++)

    final SinglestreamXmlDumpParser parser = new SinglestreamXmlDumpParser(new GZIPInputStream(new FileInputStream(fileName)));

    final ThreadPoolExecutor executor = ThreadTools.createBGExecutor("Creating DictExpansion", -1);
    PipelineBuilder.input(parser).pipe(new SwebleWikimarkupToText(new RuConfig())).pipe(new Filter<WikipediaPage>() {
      int index = 0;

      @Override
      protected boolean accept(WikipediaPage wikipediaPage) {
        String text = wikipediaPage.getText();
        text = text.replaceAll("\\s+", " ");
        text = text.replaceAll("«", "\"");
        text = text.replaceAll("»", "\"");
        String[] sentences = text.split("\\.\\s");
        for (int i = 0; i < sentences.length; i++) {
          final String sentence = sentences[i];
          if (sentence.length() < 100)
            continue;
//          System.out.println(sentence);
          executor.execute(() -> {
            expansion.accept(new CharSeqAdapter(sentence));
            if ((++index) % 10000 == 0) {
              try {
                expansion.print(new FileWriter(output.getAbsolutePath()));
              } catch (Exception e) {
                e.printStackTrace();
              }
            }
          });
        }
        return false;
      }
    }).build().run();
  }
}
