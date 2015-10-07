package com.spbsu.wiki;

import com.spbsu.commons.func.Processor;
import com.spbsu.commons.seq.CharSeqAdapter;
import com.spbsu.commons.seq.IntSeq;
import info.bliki.wiki.filter.PlainTextConverter;
import info.bliki.wiki.model.WikiModel;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.lang3.StringEscapeUtils;
import se.lth.cs.nlp.mediawiki.model.WikipediaPage;
import se.lth.cs.nlp.mediawiki.parser.SinglestreamXmlDumpParser;
import se.lth.cs.nlp.pipeline.Filter;
import se.lth.cs.nlp.pipeline.PipelineBuilder;
import se.lth.cs.nlp.wikipedia.lang.RuConfig;
import se.lth.cs.nlp.wikipedia.parser.SwebleWikimarkupToText;

import java.io.*;
import java.util.Arrays;

/**
 * Created by Юлиан on 05.10.2015.
 */
public class WikiforiaParser implements WikiParser {

    private final static WikiModel wikiModel = new WikiModel("", "");
    private final static String[] ends = new String[]{
            "==External links==",
            "==Further reading==",
            "==References==",
            "==See also==",
            "==Notes=="
    };

    private Processor<String> processor;

    public WikiforiaParser(){

    }

    public void setProcessor(final Processor<String> processor){
        this.processor = processor;
    }

    public void parse(final InputStream stream) throws Exception{
            final SinglestreamXmlDumpParser parser = new SinglestreamXmlDumpParser(stream);

            PipelineBuilder.input(parser).pipe(new SwebleWikimarkupToText(new RuConfig())).pipe(new Filter<WikipediaPage>() {
                @Override
                protected boolean accept(WikipediaPage wikipediaPage) {
                    try {
                        String text = wikipediaPage.getText();
                        text = plainText(text);
                        processor.process(text);
                        return false;
                    } catch (Exception ex){
                        ex.printStackTrace();
                    }
                    return false;
                }
            }).build().run();
    }

    public static String plainText(String text){
        //don't remove <ref> text </ref>
        StringBuilder result = new StringBuilder(text);
        int braceIndex = result.indexOf("{");

        while(braceIndex != -1){
            int closeBrace = fineMatchBrace(result, braceIndex);
            if(closeBrace != -1)
                result.delete(braceIndex, closeBrace + 1);
            else
                result.setCharAt(braceIndex, ' ');
            braceIndex = result.indexOf("{");
        }

        result = new StringBuilder(
                wikiModel.render(new PlainTextConverter(),
                        StringEscapeUtils.unescapeHtml4(
                                result.toString().replaceAll("\\\\'", "'")
                                        .replaceAll("\\\\\"", "\"")
                                        .replaceAll("<math>.*?</math>", " ")
                                        .replaceAll("]<", "] <"))
                ));


        int index;
        for(String end : ends){
            index = result.indexOf(end);
            if(index != -1)
                result.delete(index, result.length());
        }

        while((index = result.indexOf("[[")) != -1) {
            String str = wikiModel.render(new PlainTextConverter(),result.substring(index));
            if(!str.startsWith("[["))
                result.delete(index, result.length()).append(str);
            else {
                int matchBrace = fineMatchBrace(str, 0);
                if(matchBrace != -1)
                    result.delete(index, result.length()).append(str.substring(matchBrace + 1));
                else
                    result.delete(index, result.length()).append(str.substring(2));
            }
        }

        removeEquals(result);

        while(result.indexOf("[") != -1 && result.indexOf("]", result.indexOf("[")) != -1){
            result = new StringBuilder(result.toString().replaceAll("\\[[^\\[\\]]*?\\]", " "));
        }
        return StringEscapeUtils.unescapeHtml4(result.toString()).replaceAll("http://[^\\s]*", " ")
                .replaceAll("\\n", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static int fineMatchBrace(StringBuilder str, int index){
        int sum = 1;
        for(int i = index + 1; i < str.length(); i++){
            sum += signOfSymbol(str.charAt(i));
            if(sum == 0)
                return i;
        }
        sum = 1;
        for(int i = index + 1; i < str.length(); i++){
            sum += signOfSymbol(str.charAt(i));
            if(sum == 1 && str.charAt(i) == '}')
                return i;
        }
        return -1;
    }

    private static int signOfSymbol(char c){
        if(c == '{' || c == '[')
            return 1;
        if(c == '}' || c == ']')
            return -1;
        return 0;
    }

    private static int fineMatchBrace(String str, int index){
        int sum = 1;
        for(int i = index + 1; i < str.length(); i++){
            sum += signOfSymbol(str.charAt(i));
            if(sum == 0)
                return i;
        }
        sum = 1;
        for(int i = index + 1; i < str.length(); i++){
            sum += signOfSymbol(str.charAt(i));
            if(sum == 1 && str.charAt(i) == '}')
                return i;
        }
        return -1;
    }

    public static void removeEquals(StringBuilder sb){
        //Example: ==Section=====Subsection=======Subsubsection====
        int lastIndex;

        while ((lastIndex = sb.lastIndexOf("=")) != -1) {
            int length = 1;
            while (lastIndex - length >= 0 && sb.charAt(lastIndex - length) == '=')
                length++;
            if (length == 1) {
                sb.deleteCharAt(lastIndex);
            } else {
                char[] c = new char[length];
                Arrays.fill(c, '=');
                int firstIndex = sb.lastIndexOf(new String(c), lastIndex - length);
                if (lastIndex - firstIndex < 100 && firstIndex != -1)
                    sb.delete(firstIndex, lastIndex).setCharAt(firstIndex, ' ');
                else
                    sb.delete(lastIndex - length + 2, lastIndex + 1).setCharAt(lastIndex - length + 1, ' ');
            }
        }
    }

}
