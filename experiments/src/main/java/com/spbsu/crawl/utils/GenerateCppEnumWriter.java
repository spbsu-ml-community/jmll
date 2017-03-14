package com.spbsu.crawl.utils;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * User: Noxoomo
 * Date: 30.04.16
 * Time: 21:47
 */
public class GenerateCppEnumWriter {

  public static void main(String[] args) {
    try {
      Scanner reader = new Scanner(new BufferedInputStream(new FileInputStream("enum_lines")));
      Pattern enumNamePatter = Pattern.compile("([A-Z_0-9]+)");
      Set<String> enumValues = new HashSet<>();

      while (reader.hasNext()) {
        String line = reader.nextLine();
        if (line.length() == 0 || line.charAt(0) == '#' || !line.contains("_")) {
          continue;
        }

        Matcher matcher = enumNamePatter.matcher(line);

        if (matcher.find()) {
          enumValues.add(matcher.group(0));
        }
      }


      for (String enumValue : enumValues) {
        System.out.println("out << \"" + enumValue + " : \" << " + enumValue + " << std::endl;");
      }


    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
}
