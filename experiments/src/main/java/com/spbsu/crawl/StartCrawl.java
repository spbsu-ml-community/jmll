package com.spbsu.crawl;

import com.spbsu.commons.system.RuntimeUtils;
import com.spbsu.crawl.data.GameSession;
import com.spbsu.crawl.data.Hero;

import javax.websocket.DeploymentException;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;

/**
 * Experts League
 * Created by solar on 23/03/16.
 */
public class StartCrawl {
  public static void main(String[] args) throws IOException, InterruptedException, URISyntaxException, DeploymentException {
    final File socketFile = File.createTempFile("crawl", ".socket");
    final RuntimeUtils.BashProcess bash = new RuntimeUtils.BashProcess("experiments/crawl");
    try {
      //noinspection ResultOfMethodCallIgnored
      socketFile.delete();
      bash.exec("bash ./run_server.sh");
      Thread.sleep(1000);
      final WSEndpoint endpoint = new WSEndpoint(new URI("ws://localhost:8080/socket"));
      final GameProcess gameProcess = new GameProcess(endpoint, new GameSession() {
        @Override
        public Hero.Race race() {
          return null;
        }

        @Override
        public Hero.Spec spec() {
          return null;
        }

        @Override
        public void heroPosition(int x, int y) {

        }

        @Override
        public void setDungeonLevel(int level) {

        }

      });
      gameProcess.start();
    }
    catch (Exception e) {
      e.printStackTrace();
      bash.destroy();
    }
    finally {
      bash.waitFor();
      //noinspection ResultOfMethodCallIgnored
      socketFile.delete();
    }
  }
}
