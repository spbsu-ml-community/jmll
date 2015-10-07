package com.spbsu.exp.wikipedia;

import com.spbsu.commons.seq.IntSeq;
import com.spbsu.wiki.BlikiParser;
import com.spbsu.wiki.WikiUtils;
import com.spbsu.wiki.WikiforiaParser;
import junit.framework.TestCase;
import se.lth.cs.nlp.mediawiki.parser.XmlDumpParser;
import sun.misc.Regexp;

import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by Юлиан on 05.10.2015.
 */
public class ParserTests extends TestCase {



    public void testBlikiReading() throws Exception{
        XmlDumpParser parser = new XmlDumpParser(new FileInputStream("src/test/resources/wikipedia_page_example.xml"));
        assertEquals("{{text}}'''Simple''' \"[[Hello world|hello world]]\" \n" +
                        "<ref>http://yandex.ru</ref>\t  example [[File: file description.txt]]",
                parser.next().getContent());
    }

    public void testBlikiParser() throws Exception{

        BlikiParser parser = new BlikiParser();
        final String[] result = new String[1];
        parser.setProcessor((str) -> result[0] = str);
        parser.parse(new FileInputStream("src/test/resources/wikipedia_page_example.xml"));
        assertEquals("Simple \"hello world\" example", result[0]);

    }

    public void testWikiforiaParser() throws Exception{

        WikiforiaParser parser = new WikiforiaParser();
        final String[] result = new String[1];
        parser.setProcessor((str) -> result[0] = str);
        parser.parse(new FileInputStream("src/test/resources/wikipedia_page_example.xml"));
        assertEquals("Simple \"hello world\" example", result[0]);

    }

    public void testSplitting(){

        String[] sentences = WikiUtils.splitIntoSentences("It is the first sentence. It is the second U.S. sentence.And this is the third sentence.");
        assertEquals(3, sentences.length);
        assertEquals("It is the first sentence", sentences[0]);
        assertEquals("It is the second U.S. sentence", sentences[1]);
        assertEquals("And this is the third sentence", sentences[2]);
    }

    public void testRemovingPunctuations(){
        String result = WikiUtils.removePunctuation("There's!?/\" no: spoon-! 3.1415");
        assertEquals("There's no: spoon- 3.1415", result);

    }

    public void testStringToIntSeq(){
        IntSeq expected = new IntSeq(new int[]{1,2,1,3,1,4,2});
        IntSeq actual = WikiUtils.stringToIntSeq("a b a c a d b");
        assertEquals(expected, actual);
        Map<String, Integer> indexes = WikiUtils.getIndexes();
        HashMap<String, Integer> expectedMap = new HashMap<>();
        expectedMap.put("a", 1);
        expectedMap.put("b", 2);
        expectedMap.put("c", 3);
        expectedMap.put("d", 4);
        assertEquals(expectedMap, indexes);
    }


}
